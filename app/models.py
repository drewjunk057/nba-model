"""Training, evaluation, versioning and inference for the three models.

  win     LightGBM classifier  -> home_win_prob   (logistic regression baseline reported)
  spread  LightGBM classifier  -> home_covers_prob
  total   LightGBM regressor   -> predicted_total_points (+ over/under via residual sigma)

Evaluation is walk-forward: hold out the most recent completed season,
train on everything before it, early-stop on the season before that.
The production model is then refit on all completed games.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR
from .features import FEATURE_SETS

TARGETS = {"win": "home_win", "spread": "home_covers", "total": "total_points"}

LGB_PARAMS = dict(
    learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, random_state=7,
)

# key features kept for drift monitoring (PSI)
DRIFT_FEATURES = ["net_rtg_diff", "closing_home_spread", "closing_total", "pace_avg",
                  "home_missing_minutes", "spread_gap", "total_gap"]


# ------------------------------------------------------------- metrics
def calibration(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 10:
            out.append(dict(bin=float((edges[b] + edges[b + 1]) / 2), predicted=float(p[m].mean()),
                            actual=float(y[m].mean()), n=int(m.sum())))
    return out


def clf_metrics(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return dict(n=int(len(y)), accuracy=float(((p > 0.5) == (y == 1)).mean()),
                brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
                calibration=calibration(y, p))


def betting_record(y, p, threshold=0.04, price=-110):
    """Flat 1u bets when |p - 0.5| > threshold, at a standard -110 price."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    pick = np.abs(p - 0.5) > threshold
    if not pick.any():
        return dict(bets=0, win_rate=None, units=0.0, roi=None)
    won = ((p[pick] > 0.5) == (y[pick] == 1))
    win_pay = 100 / -price
    units = float(won.sum() * win_pay - (~won).sum())
    return dict(bets=int(pick.sum()), win_rate=float(won.mean()), units=round(units, 2),
                roi=float(units / pick.sum()))


def psi_bins(values):
    qs = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    counts, _ = np.histogram(values, bins=qs)
    return dict(edges=qs.tolist(), freq=(counts / counts.sum()).tolist())


def psi(ref, values):
    edges = np.array(ref["edges"])
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
    cur = counts / max(counts.sum(), 1)
    exp = np.array(ref["freq"])
    cur, exp = np.clip(cur, 1e-4, None), np.clip(exp, 1e-4, None)
    return float(((cur - exp) * np.log(cur / exp)).sum())


# --------------------------------------------------------------- fitting
def _fit_lgb(kind, X, y, X_val=None, y_val=None, rounds=None):
    params = dict(LGB_PARAMS, n_estimators=rounds or 2000)
    model = lgb.LGBMRegressor(**params) if kind == "total" else lgb.LGBMClassifier(**params)
    if X_val is not None:
        import warnings
        with warnings.catch_warnings():          # eval_set works on every LightGBM 4.x
            warnings.simplefilter("ignore")
            model.fit(X, y, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(X, y)
    return model


def _predict(kind, model, X):
    return model.predict(X) if kind == "total" else model.predict_proba(X)[:, 1]


def _train_frame(gf, kind):
    tgt = TARGETS[kind]
    d = gf[gf[tgt].notna()]
    return d[FEATURE_SETS[kind]], d[tgt].astype(float), d


def evaluate(gf: pd.DataFrame) -> dict:
    """Walk-forward evaluation on the most recent completed season."""
    played = gf[gf.home_score.notna()]
    seasons = sorted(played.season.unique())
    counts = played.season.value_counts()
    full = [s for s in seasons if counts.get(s, 0) >= 0.6 * counts.max()]   # skip partial seasons
    if len(full) < 3:
        raise ValueError(f"Need at least 3 complete seasons to train and evaluate; have {full}")
    test_season, es_season = full[-1], full[-2]
    train_seasons = [s for s in seasons if s < es_season]

    report = dict(test_season=test_season, train_seasons=train_seasons + [es_season],
                  early_stop_season=es_season, models={})
    best_rounds, sigma = {}, None
    preds = played[played.season == test_season][
        ["game_id", "date", "home_team", "away_team", "home_score", "away_score",
         "closing_home_spread", "closing_total", "mkt_home_prob"]].copy()

    for kind in ("win", "spread", "total"):
        X, y, d = _train_frame(gf, kind)
        tr, es, te = d.season.isin(train_seasons), d.season == es_season, d.season == test_season
        m = _fit_lgb(kind, X[tr], y[tr], X[es], y[es])
        best_rounds[kind] = int(m.best_iteration_ or m.n_estimators)
        # refit on train + early-stop season with the chosen number of rounds
        trall = tr | es
        m = _fit_lgb(kind, X[trall], y[trall], rounds=best_rounds[kind])
        p = _predict(kind, m, X[te])
        preds.loc[d[te].index, f"{kind}_pred"] = p

        if kind == "total":
            resid = y[te] - p
            sigma = float(np.std(resid))   # out-of-sample residual spread
            mkt = d.loc[te, "closing_total"]
            ou_y = d.loc[te, "over"]
            over_p = 1 - norm.cdf((mkt.values - p) / sigma)
            mask = ou_y.notna()
            hl = mkt.notna().values                      # games with a market total
            report["models"][kind] = dict(
                n=int(te.sum()), n_with_line=int(hl.sum()),
                rmse=float(np.sqrt(np.mean(resid ** 2))), mae=float(np.mean(np.abs(resid))),
                model_rmse_with_line=float(np.sqrt(np.mean(resid[hl] ** 2))),
                market_rmse=float(np.sqrt(np.mean((y[te][hl] - mkt[hl]) ** 2))),
                market_mae=float(np.mean(np.abs(y[te][hl] - mkt[hl]))),
                mean_edge_vs_market=float(np.mean(np.abs(p[hl] - mkt[hl]))),
                over_under=clf_metrics(ou_y[mask], over_p[mask.values]),
                over_under_bets=betting_record(ou_y[mask], over_p[mask.values]),
                sigma=sigma, rounds=best_rounds[kind])
            preds.loc[d[te].index, "over_prob"] = over_p
        else:
            r = clf_metrics(y[te], p)
            r["rounds"] = best_rounds[kind]
            if kind == "win":
                mk = d.loc[te, "mkt_home_prob"]
                ok = mk.notna()
                r["market_baseline"] = clf_metrics(y[te][ok], mk[ok])
                r["market_baseline"]["model_brier_same_games"] = float(brier_score_loss(y[te][ok], p[ok.values]))
                lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=0.1, max_iter=2000))
                lr.fit(X[trall], y[trall])
                r["logistic_baseline"] = clf_metrics(y[te], lr.predict_proba(X[te])[:, 1])
            else:
                r["coin_flip_brier"] = 0.25
                r["bets"] = betting_record(y[te], p)
            report["models"][kind] = r

    report["best_rounds"] = best_rounds
    report["sigma"] = sigma
    preds = preds.rename(columns={"win_pred": "home_win_prob", "spread_pred": "home_covers_prob",
                                  "total_pred": "predicted_total_points"})
    report["backtest"] = json.loads(preds.to_json(orient="records"))
    return report


def train_production(gf: pd.DataFrame, report: dict, as_of: str | None = None) -> str:
    """Fit on all completed games, save a new versioned bundle, mark it current."""
    played = gf[gf.home_score.notna()]
    existing = sorted(p.name for p in MODEL_DIR.glob("v*") if p.is_dir())
    n = 1 + max([int(p.split("-")[0][1:]) for p in existing] or [0])
    stamp = (as_of or datetime.now().date().isoformat()).replace("-", "")
    version = f"v{n}-{stamp}"
    vdir = MODEL_DIR / version
    vdir.mkdir(parents=True, exist_ok=True)

    importances, n_train = {}, {}
    scale = len(played) / max(1, len(played[played.season.isin(report["train_seasons"])]))
    for kind in ("win", "spread", "total"):
        X, y, _ = _train_frame(gf, kind)
        rounds = max(50, int(report["best_rounds"][kind] * min(scale, 1.3)))
        m = _fit_lgb(kind, X, y, rounds=rounds)
        m.booster_.save_model(str(vdir / f"{kind}.txt"))
        gain = pd.Series(m.booster_.feature_importance("gain"), index=FEATURE_SETS[kind])
        importances[kind] = (gain / gain.sum()).sort_values(ascending=False).head(20).round(4).to_dict()
        n_train[kind] = int(len(y))

    meta = dict(
        model_version=version, trained_at=datetime.now().isoformat(timespec="seconds"),
        data_source=_data_source(),
        trained_through=str(played.date.max()), trained_as_of=as_of or datetime.now().date().isoformat(),
        n_train=n_train,
        features=FEATURE_SETS, sigma_total=report["sigma"],
        importances=importances,
        evaluation={k: v for k, v in report.items() if k != "backtest"},
        drift_reference={f: psi_bins(played[f].dropna().values) for f in DRIFT_FEATURES},
    )
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
    (vdir / "backtest.json").write_text(json.dumps(report["backtest"]))
    (MODEL_DIR / "current.json").write_text(json.dumps({"model_version": version}))
    return version


def _data_source() -> str:
    from .pipeline import data_source
    return data_source()


# --------------------------------------------------------------- inference
class ModelBundle:
    def __init__(self, version: str | None = None):
        if version is None:
            version = json.loads((MODEL_DIR / "current.json").read_text())["model_version"]
        self.dir = MODEL_DIR / version
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.version = version
        self.boosters = {k: lgb.Booster(model_file=str(self.dir / f"{k}.txt"))
                         for k in ("win", "spread", "total")}
        self.sigma = self.meta["sigma_total"]

    def backtest(self):
        p = self.dir / "backtest.json"
        return json.loads(p.read_text()) if p.exists() else []

    def predict(self, rows: pd.DataFrame, n_drivers: int = 5) -> list[dict]:
        out = [dict() for _ in range(len(rows))]
        for kind, booster in self.boosters.items():
            feats = FEATURE_SETS[kind]
            X = rows[feats].astype(float)
            pred = booster.predict(X)
            contrib = booster.predict(X, pred_contrib=True)[:, :-1]   # last col = bias
            if kind != "total":
                # log-odds contributions -> approx. probability points
                contrib = contrib * (pred * (1 - pred))[:, None]
            for i in range(len(rows)):
                order = np.argsort(-np.abs(contrib[i]))[:n_drivers]
                drivers = [dict(feature=feats[j], value=_num(X.iat[i, j]),
                                impact=round(float(contrib[i, j]), 4)) for j in order]
                if kind == "win":
                    out[i]["home_win_prob"] = round(float(pred[i]), 4)
                elif kind == "spread":
                    out[i]["home_covers_prob"] = round(float(pred[i]), 4)
                else:
                    tot = float(pred[i])
                    out[i]["predicted_total_points"] = round(tot, 1)
                    line = rows["closing_total"].iat[i]
                    if pd.notna(line):
                        over = 1 - norm.cdf((line - tot) / self.sigma)
                        out[i]["over_prob"] = round(float(over), 4)
                        out[i]["under_prob"] = round(1 - float(over), 4)
                out[i].setdefault("drivers", {})[kind] = drivers
        for o in out:
            o["model_version"] = self.version
        return out


def _num(v):
    v = float(v)
    return None if math.isnan(v) else round(v, 3)
