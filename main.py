import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import re
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
import requests
from scipy.stats import norm
import xgboost
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import nfl_data_py as nfl

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

DEFAULT_NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAR", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
]

DEFAULT_CAROUSEL_GAMES = [
    {"id": "1", "homeTeam": "KC", "awayTeam": "BAL", "hasStarted": False, "isLive": False, "status": "Week 1 - Scheduled", "homeScore": None, "awayScore": None, "weekNumber": 1},
    {"id": "2", "homeTeam": "PHI", "awayTeam": "GB", "hasStarted": False, "isLive": False, "status": "Week 1 - Scheduled", "homeScore": None, "awayScore": None, "weekNumber": 1},
    {"id": "3", "homeTeam": "SF", "awayTeam": "NYJ", "hasStarted": False, "isLive": False, "status": "Week 1 - Scheduled", "homeScore": None, "awayScore": None, "weekNumber": 1},
    {"id": "4", "homeTeam": "DET", "awayTeam": "LAR", "hasStarted": False, "isLive": False, "status": "Week 1 - Scheduled", "homeScore": None, "awayScore": None, "weekNumber": 1},
    {"id": "5", "homeTeam": "MIA", "awayTeam": "BUF", "hasStarted": False, "isLive": False, "status": "Week 1 - Scheduled", "homeScore": None, "awayScore": None, "weekNumber": 1},
    {"id": "6", "homeTeam": "DAL", "awayTeam": "CLE", "hasStarted": False, "isLive": False, "status": "Week 1 - Scheduled", "homeScore": None, "awayScore": None, "weekNumber": 1},
]

LIVE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1Df05-UIAO2kp5AomHXnhcrTAzGqcQgkKCaEIrexEwKA/gviz/tq"
    "?tqx=out:csv&gid=1227961915"
)
NFL_TZ = ZoneInfo("America/New_York")
# Exhibition slates in the sheet that are not NFL weeks.
EXHIBITION_WEEK_LABELS = {"pro bowl", "hof game"}

LEAGUE_AVERAGE_TEAM_SCORE = 22.0
NFL_SPREAD_STD_DEV = 13.8
MIN_PROJECTED_SCORE = 0.0
MAX_PROJECTED_SCORE = 75.0

FEATURE_COLUMNS = [
    "diff_off_epa",
    "diff_off_success_rate",
    "diff_off_cpoe",
    "diff_def_epa",
    "diff_def_success_rate",
]

# State globals
model = xgboost.XGBRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
)

team_stats = {}
LEAGUE_AVERAGES = {}
LEAGUE_STDS = {}
games_cache = list(DEFAULT_CAROUSEL_GAMES)

training_state = {
    "ready": False,
    "progress": 0,
    "message": "Initializing...",
    "val_mae": None,
    "val_r2": None,
}

# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class PredictionRequest(BaseModel):
    home_team: str = Field(..., validation_alias=AliasChoices("home_team", "homeTeam"))
    away_team: str = Field(..., validation_alias=AliasChoices("away_team", "awayTeam"))

class MetricsContext(BaseModel):
    home_off_epa_ranking: str
    away_off_epa_ranking: str
    home_def_epa_ranking: str
    away_def_epa_ranking: str

class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    home_team: str
    away_team: str
    predicted_winner: str
    home_win_probability: float
    away_win_probability: float
    home_projected_score: float
    away_projected_score: float
    metrics_context: MetricsContext

class StatusResponse(BaseModel):
    ready: bool
    progress: int
    message: str
    val_mae: float | None = None
    val_r2: float | None = None

# ==============================================================================
# GOOGLE SHEETS & PARSING HELPERS
# ==============================================================================

def parse_sheet_datetime(date_str, time_str) -> datetime | None:
    date_str = str(date_str).strip()
    if not date_str or date_str == "nan":
        return None
    try:
        game_dt = datetime.strptime(date_str, "%a %m/%d/%Y")
    except ValueError:
        return None

    time_str = str(time_str).strip()
    if time_str and time_str != "nan":
        try:
            t = datetime.strptime(time_str, "%I:%M %p")
            game_dt = game_dt.replace(hour=t.hour, minute=t.minute)
        except ValueError:
            pass
    return game_dt.replace(tzinfo=NFL_TZ)


def parse_week_label(week_str) -> str | None:
    """Keep the sheet's NFL week label intact (Pre Week 3 ≠ Week 1)."""
    label = str(week_str).strip()
    if not label or label.lower() in {"nan", "none"}:
        return None
    if label.lower() in EXHIBITION_WEEK_LABELS:
        return None
    return label


def parse_week_number(week_str) -> int | None:
    match = re.search(r"\d+", str(week_str))
    return int(match.group()) if match else None


def determine_weeks_to_show(rows_with_meta: list, now: datetime | None = None) -> set[str]:
    """Return the prior NFL week and the upcoming (or in-progress) NFL week.

    rows_with_meta: list of (week_label, game_dt) tuples. Weeks are keyed by
    the sheet label so preseason and regular season never collapse together.
    """
    if now is None:
        now = datetime.now(NFL_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=NFL_TZ)

    week_ranges: dict[str, tuple[datetime, datetime]] = {}
    for week_label, game_dt in rows_with_meta:
        if week_label is None or game_dt is None:
            continue
        if game_dt.tzinfo is None:
            game_dt = game_dt.replace(tzinfo=NFL_TZ)
        lo, hi = week_ranges.get(week_label, (game_dt, game_dt))
        week_ranges[week_label] = (min(lo, game_dt), max(hi, game_dt))

    if not week_ranges:
        return set()

    weeks_sorted = sorted(week_ranges.items(), key=lambda item: item[1][0])

    in_progress_idx = next(
        (
            i
            for i, (_, (lo, hi)) in enumerate(weeks_sorted)
            if lo <= now <= hi + timedelta(days=1)
        ),
        None,
    )
    if in_progress_idx is not None:
        upcoming_idx = in_progress_idx
    else:
        upcoming_idx = next(
            (i for i, (_, (lo, _)) in enumerate(weeks_sorted) if lo > now),
            len(weeks_sorted) - 1,
        )

    labels = {weeks_sorted[upcoming_idx][0]}
    if upcoming_idx > 0:
        labels.add(weeks_sorted[upcoming_idx - 1][0])
    return labels


def parse_sheet_games(df: pd.DataFrame) -> list:
    if df.empty:
        return []

    week_col = df.columns[0]  # Column A holds week identifiers ("Week 1", etc.)
    parsed_rows = []

    for _, row in df.iterrows():
        away = str(row.get("Away", "")).strip()
        home = str(row.get("Home", "")).strip()
        if not away or not home or away == "nan" or home == "nan":
            continue

        week_label = parse_week_label(row[week_col])
        if week_label is None:
            continue

        game_dt = parse_sheet_datetime(row.get("Date"), row.get("Time"))
        parsed_rows.append((row, week_label, game_dt))

    weeks_to_show = determine_weeks_to_show([(label, dt) for _, label, dt in parsed_rows])

    games = []
    for row, week_label, game_dt in parsed_rows:
        if week_label not in weeks_to_show:
            continue

        away = str(row.get("Away", "")).strip()
        home = str(row.get("Home", "")).strip()
        qtr = str(row.get("Qtr", "")).strip().lower()
        clock = str(row.get("Clock", "")).strip()
        is_final = qtr in ("final", "f")
        is_live = not is_final and qtr not in ("", "pre", "nan")
        has_started = is_live or is_final
        week_num = parse_week_number(week_label)

        if is_final:
            status = f"{week_label} · Final"
        elif is_live:
            status = f"{week_label} · Q{qtr} {clock}".strip()
        elif game_dt:
            status = f"{week_label} · {game_dt.strftime('%a %m/%d %I:%M %p')}"
        else:
            status = f"{week_label} - Scheduled"

        games.append({
            "id": f"{away}_{home}_{row.get('Date', '')}",
            "homeTeam": home,
            "awayTeam": away,
            "homeScore": row.get("Home Score") if pd.notna(row.get("Home Score")) else None,
            "awayScore": row.get("Away Score") if pd.notna(row.get("Away Score")) else None,
            "status": status,
            "isLive": is_live,
            "hasStarted": has_started,
            "weekNumber": week_num,
            "weekLabel": week_label,
        })
    return games

# ==============================================================================
# ML DATA PREPARATION & PROCESSING
# ==============================================================================

def calculate_team_features_weighted(pbp_df: pd.DataFrame, half_life_weeks: float = 18.0) -> pd.DataFrame:
    valid_pbp = pbp_df.dropna(subset=["posteam", "defteam", "season", "week"]).copy()
    valid_pbp = valid_pbp[valid_pbp["play_type"].isin(["pass", "run"])]

    for col in ["epa", "success", "cpoe"]:
        if col in valid_pbp.columns:
            valid_pbp[col] = pd.to_numeric(valid_pbp[col], errors="coerce")

    max_season = valid_pbp["season"].max()
    max_week = valid_pbp[valid_pbp["season"] == max_season]["week"].max()

    valid_pbp["weeks_ago"] = (
        (max_season - valid_pbp["season"]) * 18 + (max_week - valid_pbp["week"])
    )
    valid_pbp["weight"] = 0.5 ** (valid_pbp["weeks_ago"] / half_life_weeks)

    def weighted_avg(group, col):
        sub = group.dropna(subset=[col, "weight"])
        if sub.empty or sub["weight"].sum() == 0:
            return np.nan
        return float(np.average(sub[col], weights=sub["weight"]))

    teams = sorted(valid_pbp["posteam"].unique())
    records = []

    for team in teams:
        off_plays = valid_pbp[valid_pbp["posteam"] == team]
        def_plays = valid_pbp[valid_pbp["defteam"] == team]

        records.append({
            "team": team,
            "off_epa": weighted_avg(off_plays, "epa"),
            "off_success_rate": weighted_avg(off_plays, "success"),
            "off_cpoe": weighted_avg(off_plays, "cpoe"),
            "def_epa": weighted_avg(def_plays, "epa"),
            "def_success_rate": weighted_avg(def_plays, "success"),
        })

    team_df = pd.DataFrame(records).set_index("team")
    team_df = team_df.fillna(team_df.mean())
    
    # Offense: higher EPA is better
    team_df["off_epa_rank"] = team_df["off_epa"].rank(ascending=False, method="min").astype(int)
    # Defense: lower EPA allowed is better
    team_df["def_epa_rank"] = team_df["def_epa"].rank(ascending=True, method="min").astype(int)
    
    return team_df


def build_game_level_dataset(schedules_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    games = schedules_df.dropna(subset=["home_team", "away_team", "home_score", "away_score"]).copy()
    games["margin"] = games["home_score"] - games["away_score"]

    games = games.merge(team_df.add_prefix("home_"), left_on="home_team", right_index=True, how="inner")
    games = games.merge(team_df.add_prefix("away_"), left_on="away_team", right_index=True, how="inner")

    games["diff_off_epa"] = games["home_off_epa"] - games["away_off_epa"]
    games["diff_off_success_rate"] = games["home_off_success_rate"] - games["away_off_success_rate"]
    games["diff_off_cpoe"] = games["home_off_cpoe"] - games["away_off_cpoe"]
    games["diff_def_epa"] = games["home_def_epa"] - games["away_def_epa"]
    games["diff_def_success_rate"] = games["home_def_success_rate"] - games["away_def_success_rate"]

    return games


def project_scores_and_probabilities(home_team: str, away_team: str, predicted_spread: float):
    home_data = team_stats.get(home_team, LEAGUE_AVERAGES)
    away_data = team_stats.get(away_team, LEAGUE_AVERAGES)

    epa_std = LEAGUE_STDS.get("off_epa", 0.1) if LEAGUE_STDS else 0.1
    if epa_std == 0:
        epa_std = 0.1
        
    MATCHUP_SCALE = 10.0

    home_off = home_data.get("off_epa", 0.0)
    away_def = away_data.get("def_epa", 0.0)
    away_off = away_data.get("off_epa", 0.0)
    home_def = home_data.get("def_epa", 0.0)

    home_off_matchup = (home_off - away_def) / epa_std
    away_off_matchup = (away_off - home_def) / epa_std

    raw_home_points = LEAGUE_AVERAGE_TEAM_SCORE + (home_off_matchup * MATCHUP_SCALE)
    raw_away_points = LEAGUE_AVERAGE_TEAM_SCORE + (away_off_matchup * MATCHUP_SCALE)
    projected_total = raw_home_points + raw_away_points

    home_score = (projected_total / 2.0) + (predicted_spread / 2.0)
    away_score = (projected_total / 2.0) - (predicted_spread / 2.0)

    home_score = float(np.clip(home_score, MIN_PROJECTED_SCORE, MAX_PROJECTED_SCORE))
    away_score = float(np.clip(away_score, MIN_PROJECTED_SCORE, MAX_PROJECTED_SCORE))

    home_win_prob = float(norm.cdf(predicted_spread / NFL_SPREAD_STD_DEV))
    away_win_prob = 1.0 - home_win_prob

    return home_score, away_score, home_win_prob, away_win_prob


def normalize_team_abbr(abbr: str) -> str:
    t = abbr.upper().strip()
    aliases = {"LAR": "LA", "LA": "LAR", "LV": "OAK", "OAK": "LV", "SD": "LAC", "STL": "LAR"}
    if t in team_stats:
        return t
    if aliases.get(t) in team_stats:
        return aliases[t]
    return t


def parse_espn_events(data: dict) -> list:
    """Parses ESPN raw JSON event payload into app-compatible game objects."""
    games = []
    for event in data.get("events", []):
        status_container = event.get("status", {})
        status_type = status_container.get("type", {})

        status_text = status_type.get("detail", "Scheduled")
        raw_status = status_type.get("name", "STATUS_SCHEDULED")

        competitions = event.get("competitions", [])
        if not competitions:
            continue

        competitors = competitions[0].get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        h_abbr = home.get("team", {}).get("abbreviation", "UNK")
        a_abbr = away.get("team", {}).get("abbreviation", "UNK")

        is_live = raw_status == "STATUS_IN_PROGRESS"
        is_finished = raw_status == "STATUS_FINAL"
        has_started = is_live or is_finished

        games.append({
            "id": str(event.get("id")),
            "homeTeam": h_abbr,
            "awayTeam": a_abbr,
            "homeScore": home.get("score", "0"),
            "awayScore": away.get("score", "0"),
            "status": status_text,
            "rawStatus": raw_status,
            "isLive": is_live,
            "hasStarted": has_started
        })
    return games

# ==============================================================================
# BACKGROUND TASKS
# ==============================================================================

async def train_model_task():
    global team_stats, LEAGUE_AVERAGES, LEAGUE_STDS, games_cache, training_state
    try:
        training_state["message"] = "Downloading play-by-play data..."
        training_state["progress"] = 10
        await asyncio.sleep(0.1)

        current_year = datetime.now().year
        training_seasons = [current_year - 2, current_year - 1]

        pbp_data = await asyncio.to_thread(nfl.import_pbp_data, training_seasons)
        schedules_training = await asyncio.to_thread(nfl.import_schedules, training_seasons)

        training_state["message"] = f"Loading {current_year} season schedule..."
        training_state["progress"] = 25
        
        try:
            current_schedule = await asyncio.to_thread(nfl.import_schedules, [current_year])
            unplayed = current_schedule[current_schedule["home_score"].isna()]
            if not unplayed.empty:
                upcoming_week = int(unplayed["week"].min())
            else:
                upcoming_week = 1

            # Build fallback games cache for previous week + current week
            weeks_to_show = {max(1, upcoming_week - 1), upcoming_week}
            filtered_schedule = current_schedule[
                current_schedule["week"].isin(weeks_to_show)
            ].dropna(subset=["home_team", "away_team"])

            formatted_games = []
            for _, row in filtered_schedule.iterrows():
                h_score = float(row["home_score"]) if pd.notna(row.get("home_score")) else None
                a_score = float(row["away_score"]) if pd.notna(row.get("away_score")) else None
                has_started = h_score is not None and a_score is not None
                wk = int(row.get("week", upcoming_week))

                formatted_games.append({
                    "id": str(row.get("game_id", f"{row.get('home_team')}_{row.get('away_team')}")),
                    "homeTeam": str(row.get("home_team")).strip(),
                    "awayTeam": str(row.get("away_team")).strip(),
                    "hasStarted": has_started,
                    "isLive": False,
                    "status": "Final" if has_started else f"Week {wk} - Scheduled",
                    "homeScore": h_score,
                    "awayScore": a_score,
                    "weekNumber": wk,
                })
            
            if formatted_games:
                games_cache = formatted_games
        except Exception as sched_err:
            print(f"Schedule cache warning: {sched_err}")

        training_state["message"] = "Calculating weighted recency features..."
        training_state["progress"] = 40
        await asyncio.sleep(0.1)

        team_df = calculate_team_features_weighted(pbp_data, half_life_weeks=18.0)
        
        team_stats = team_df.to_dict(orient="index")
        LEAGUE_AVERAGES = team_df.mean().to_dict()
        LEAGUE_STDS = team_df.std().to_dict()

        training_state["message"] = "Building spread target dataset..."
        training_state["progress"] = 60
        await asyncio.sleep(0.1)

        dataset = build_game_level_dataset(schedules_training, team_df)

        X = dataset[FEATURE_COLUMNS]
        y = dataset["margin"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        training_state["message"] = "Training XGBoost Regressor..."
        training_state["progress"] = 80
        await asyncio.sleep(0.1)

        await asyncio.to_thread(model.fit, X_train, y_train)

        val_preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        r2 = r2_score(y_val, val_preds)

        training_state.update({
            "ready": True,
            "progress": 100,
            "message": "Model training complete and ready for inference.",
            "val_mae": round(float(mae), 3),
            "val_r2": round(float(r2), 3),
        })

    except Exception as e:
        training_state.update({
            "ready": False,
            "progress": 0,
            "message": f"Training failed: {str(e)}",
        })

# ==============================================================================
# FASTAPI APP LIFECYCLE & ROUTES
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(train_model_task())
    yield

app = FastAPI(title="NFL Game Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    return training_state

@app.get("/teams")
@app.get("/api/teams")
@app.get("/api/scores/active-teams")
async def get_teams():
    teams_list = sorted(list(team_stats.keys())) if team_stats else DEFAULT_NFL_TEAMS
    return {"teams": teams_list}

@app.get("/scores")
@app.get("/games")
@app.get("/api/scores")
@app.get("/api/scores/games")
async def get_scores():
    try:
        df = await asyncio.to_thread(pd.read_csv, LIVE_SHEET_URL)
        return parse_sheet_games(df)
    except Exception as e:
        print(f"CRITICAL TICKER ERROR: {str(e)} — Falling back to cached schedule.")
        return games_cache

@app.post("/predict")
@app.post("/api/predict", response_model=PredictionResponse)
async def predict_game(request: PredictionRequest):
    if not training_state["ready"]:
        raise HTTPException(
            status_code=503, 
            detail=f"Model training in progress. Status: {training_state['message']}"
        )

    home_input = request.home_team.upper().strip()
    away_input = request.away_team.upper().strip()

    if home_input == away_input:
        raise HTTPException(status_code=400, detail="Home team and Away team must be different.")

    home = normalize_team_abbr(home_input)
    away = normalize_team_abbr(away_input)

    home_data = team_stats.get(home, LEAGUE_AVERAGES)
    away_data = team_stats.get(away, LEAGUE_AVERAGES)

    features = pd.DataFrame([{
        "diff_off_epa": home_data.get("off_epa", 0.0) - away_data.get("off_epa", 0.0),
        "diff_off_success_rate": home_data.get("off_success_rate", 0.0) - away_data.get("off_success_rate", 0.0),
        "diff_off_cpoe": home_data.get("off_cpoe", 0.0) - away_data.get("off_cpoe", 0.0),
        "diff_def_epa": home_data.get("def_epa", 0.0) - away_data.get("def_epa", 0.0),
        "diff_def_success_rate": home_data.get("def_success_rate", 0.0) - away_data.get("def_success_rate", 0.0),
    }])[FEATURE_COLUMNS]

    predicted_spread = float(model.predict(features)[0])

    proj_home, proj_away, home_prob, away_prob = project_scores_and_probabilities(
        home, away, predicted_spread
    )

    winner = home_input if predicted_spread >= 0 else away_input

    home_off_rank = home_data.get("off_epa_rank")
    away_off_rank = away_data.get("off_epa_rank")
    home_def_rank = home_data.get("def_epa_rank")
    away_def_rank = away_data.get("def_epa_rank")

    home_off_epa = home_data.get("off_epa", 0.0)
    away_off_epa = away_data.get("off_epa", 0.0)
    home_def_epa = home_data.get("def_epa", 0.0)
    away_def_epa = away_data.get("def_epa", 0.0)

    home_off_str = f"#{int(home_off_rank)} ({home_off_epa:+.2f} EPA)" if home_off_rank is not None else f"{home_off_epa:+.2f} EPA"
    away_off_str = f"#{int(away_off_rank)} ({away_off_epa:+.2f} EPA)" if away_off_rank is not None else f"{away_off_epa:+.2f} EPA"
    home_def_str = f"#{int(home_def_rank)} ({home_def_epa:+.2f} EPA)" if home_def_rank is not None else f"{home_def_epa:+.2f} EPA"
    away_def_str = f"#{int(away_def_rank)} ({away_def_epa:+.2f} EPA)" if away_def_rank is not None else f"{away_def_epa:+.2f} EPA"

    return PredictionResponse(
        home_team=home_input,
        away_team=away_input,
        predicted_winner=winner,
        home_win_probability=round(home_prob * 100, 2),
        away_win_probability=round(away_prob * 100, 2),
        home_projected_score=int(round(proj_home, 1)),
        away_projected_score=int(round(proj_away, 1)),
        metrics_context=MetricsContext(
            home_off_epa_ranking=home_off_str,
            away_off_epa_ranking=away_off_str,
            home_def_epa_ranking=home_def_str,
            away_def_epa_ranking=away_def_str,
        ),
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)