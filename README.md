# NBA Predictor

Three connected models fed by one leak-free game dataset:

| Model | Type | Output |
|---|---|---|
| Win probability | LightGBM classifier (logistic regression baseline reported) | `home_win_prob` |
| Spread (ATS) | LightGBM classifier, pushes excluded | `home_covers_prob` |
| Total points | LightGBM regressor | `predicted_total_points`, `over_prob`, `under_prob` |

FastAPI serves the predictions and a dashboard; a nightly job ingests results, settles and logs predictions, retrains weekly, and checks drift.

## Using it (Windows)

1. Install **Python 3.10 or newer** from python.org. Check **Add Python to PATH** during the install.
2. Download this repo: the green **Code** button, then **Download ZIP**, and unzip it. (Or `git clone` it.)
3. Double-click **`start.bat`**.

The first run installs the packages (a few minutes), sets up the saved models and five seasons of games from `seed/`, downloads today's lines, injuries and rosters, and opens the dashboard at http://localhost:8000. After that, `start.bat` opens in seconds and refreshes the data whenever it's more than 3 hours old. Keep the black window open while you use the app.

**Automatic updates.** The first time `start.bat` runs, it sets up a Windows scheduled task ("NBA Predictor update") that runs hidden at 4:00 AM and every 30 minutes from 8 AM to 8 PM, and catches up if the PC was asleep. It grades finished picks, logs picks for the day's games with the latest lines and injuries, and retrains weekly. The app doesn't need to be open. To turn it off, delete the task in Task Scheduler.

**Our record.** The History tab keeps a running record of every pick: winners, against the spread and over/under, graded against the line posted when the pick was made, with units at −110.

To get model updates later, download the repo again (or `git pull`). Your local data in `data/` and `models/` isn't touched.

**For the owner:** `Publish to GitHub.bat` packs the current models and game history into `seed/` and pushes everything to `github.com/drewjunk057/nba-model`.

## What the model can and can't do (tested on 2024-25 and 2025-26)

- **Win probability** starts from the betting market's own probability and lets the model nudge it. That matches the market (Brier 0.194) instead of trailing it; the old standalone model was slightly worse. Winner picks will usually be the favorite.
- **Spread and total picks are labeled "No edge".** Six different approaches all landed at 49-52% against closing lines (52.4% breaks even). They're shown as leans only.
- **Injury alerts are the one supported edge.** A stats model that knows who's missing beat early lines 54-56% in both seasons, with the line then moving its way 61-69% of the time; without injury information that edge disappeared. So the app checks injury reports every 30 minutes from 8 AM to 8 PM and flags a game (on the card, in the History tab, and with a Windows notification) when a player worth 1+ point is newly ruled out, the line hasn't caught up, and the model disagrees with the line by 2+ points. Player worth comes from a minutes-weighted adjusted plus-minus (`app/player_value.py`), refit monthly on past games only.
- **Line movement tracking.** The History tab shows how often the line moved toward each pick after it was made. Over 200+ picks, well above 50% means a real edge; around 50% means none.

These alert rules are a reasonable starting point, not a proven system: the backtest knew exactly who sat out, while live alerts depend on how fast the news arrives. Judge them by their own record and line movement over a few hundred alerts.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
go_live.bat   (or: python -m scripts.update_data --go-live)   # real NBA data + train
uvicorn app.api:app --reload                           # open http://localhost:8000
python -m pytest tests -q                              # leakage test
```

## Real data (ESPN)

`go_live.bat` (or `python -m scripts.update_data --go-live`) switches the app from demo data to real NBA data and retrains. It runs once and takes 10–25 minutes.

| What | Source | Notes |
|---|---|---|
| Schedule, tip times, scores | ESPN scoreboard | regular season, 2021-22 onward |
| Team box scores | ESPN game summary | FG, 3P, FT, OREB, DREB, turnovers; possessions estimated |
| Closing lines | ESPN game summary (DraftKings) | spread, total, moneylines; open lines kept too |
| Who didn't play, and why | ESPN game summary | every inactive player per game |
| Today's injury report | ESPN injuries | Out → OUT; Day-To-Day → QUESTIONABLE |
| Rosters | ESPN team rosters | joined to each player's real minutes and scoring |

**Availability in history** is taken from box scores: a player who didn't play for any reason except a coach's decision counts as OUT. His minutes share and star status (top-two scorer among players averaging 24+ minutes) come only from games before that one. **Live availability** uses ESPN's injury report with the same definitions.

**Keeping it current:**
- `nightly.bat` downloads only new games, lines, injuries and rosters, scores yesterday's predictions, predicts today and retrains weekly. Schedule it daily in Task Scheduler. Output goes to `data\update.log`.
- The **Update data** button in the dashboard does the same refresh on demand without retraining. Use it before tip-off, when lines and injury news are freshest.

Parsed ESPN data is cached in `data/raw/`, and the CSVs the app reads are in `data/real/`. The demo generator (`python -m scripts.setup`) still works; models trained on demo data are moved to `models/demo_synthetic/` at go-live.

**Caveats.** ESPN's endpoints are public but undocumented, so they can change without notice; the connector logs and skips anything it can't parse. Lines are DraftKings' as ESPN records them, close to but not exactly the market close. Players listed as questionable in history aren't captured (only who actually sat), so the model learns from confirmed absences.

## How it fits together

```
feed CSVs ──► ingest (as of date) ──► SQLite: games, team_box, injuries, teams
                                          │
                       features.py  ◄─────┘   pre-game only (shifted), per team
                          │                   joined twice → home_* / away_*
                          ▼
           models.py: evaluate (walk-forward) → train_production → models/vN-YYYYMMDD/
                          │
          api.py (FastAPI) ──► /api/games, /api/predict, /api/model, /api/backtest, /api/log
          pipeline.py (nightly) ──► settle → retrain if due → log predictions → drift
```

### Features (`app/features.py`)

- **Season to date:** off/def/net rating, pace, four factors (eFG%, TOV%, ORB%, FT rate), opponent eFG%, strength of schedule. Early season is blended with last season's regressed rating: weight = games / (games + 10).
- **Rolling (last 5/10/20 played games):** off/def/net rating, last-10 pace, last-10 win %, win streak.
- **Fatigue and travel:** days rest, back-to-back, games in last 7 days, road trip length, km travelled from the previous venue.
- **Availability:** star out / questionable, missing minutes share, rotation continuity (minutes share of players whose status changed since the last game).
- **Market:** closing spread, total, vig-free moneyline probability.
- **Spread extras:** rating-based spread, its gap to the market, favorite's rating edge.
- **Total extras:** expected offensive ratings (offense × defense), pace interaction, expected possessions, rating-based total and its gap to the market, fatigue load.

`tests/test_leakage.py` builds the database before and after a day's games are played and asserts every feature for those games is identical.

### Targets

- `home_win = home_score > away_score`
- `home_covers = (home_score − away_score + closing_home_spread) > 0`, pushes dropped
- `total_points = home_score + away_score`; `over` vs `closing_total`, pushes dropped

### Evaluation (`/api/model`, Model tab)

The most recent completed season is held out entirely. Models train on earlier seasons, early-stop on the season before the holdout, then refit. Reported: accuracy, Brier, log loss and calibration for win and ATS (vs the market and a coin flip); RMSE/MAE for totals vs the closing total; over/under calibration; flat-bet records at −110. The production model is then refit on everything.

Past dates in the Games tab are scored by the production model, which trained on them — the dashboard flags this. Use the Model tab for honest numbers.

## API

```bash
curl "localhost:8000/api/games?date=2026-10-20"
curl -X POST localhost:8000/api/predict -H 'Content-Type: application/json' \
  -d '{"home_team":"DEN","away_team":"BOS","date":"2026-10-22","closing_home_spread":-4.5,"closing_total":226.5,
       "injuries":[{"team":"BOS","player":"BOS Star","is_star":true,"minutes_share":0.3}]}'
```

`/api/predict` accepts a `game_id` or any hypothetical matchup; for a matchup it rebuilds features with the game inserted into the schedule, so rest and travel are correct for that date. Each response includes the spec's fields plus the top 5 drivers per model (SHAP contributions from LightGBM, shown in probability points or total points), market comparison, and edges.

Other routes: `/api/meta`, `/api/model`, `/api/backtest`, `/api/log`, `POST /api/pipeline/run?as_of=YYYY-MM-DD&retrain=false`.

## Nightly job

```bash
python -m scripts.nightly                     # as of today
# cron, 4:15 am daily:
15 4 * * * cd /path/to/nba_predictor && .venv/bin/python -m scripts.nightly >> data/nightly.log 2>&1
```

Each run: ingest results and the next 7 days of schedule → rebuild features → settle logged predictions → retrain if the model is 7+ days old (`RETRAIN_EVERY_DAYS` in `app/config.py`) → log predictions for today's games → compute drift. Drift is PSI on key features over the last 300 games (alert above 0.2) plus live Brier vs the evaluation Brier once 30+ predictions are settled. Expect a `pace_avg` alert in the first weeks of a season: early ratings are shrunk toward last year, so the distribution is narrower than the full-season reference.

Every model version is kept in `models/` with its metrics, feature importances, backtest and drift reference; `models/current.json` points at the live one.
