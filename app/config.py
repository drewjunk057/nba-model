"""Central configuration. Override any path with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("NBA_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("NBA_MODEL_DIR", ROOT / "models"))
DB_PATH = Path(os.getenv("NBA_DB_PATH", DATA_DIR / "nba.db"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Predictions are made this many hours before tip-off. Only information
# available at that moment (results, injury reports) may be used.
PREDICTION_LEAD_HOURS = 2

# Early-season blending: season-to-date ratings are shrunk toward last
# season's (regressed) ratings until a team has played enough games.
PRIOR_GAMES = 10          # weight of prior = PRIOR_GAMES / (gp + PRIOR_GAMES)
PRIOR_REGRESSION = 0.6    # last season's rating is multiplied by this

ROLLING_WINDOWS = (5, 10, 20)

# Retrain automatically in the nightly pipeline if the live model is older
RETRAIN_EVERY_DAYS = 7

LEAGUE_AVG_RTG = 114.0
LEAGUE_AVG_PACE = 99.0
