from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost
import nfl_data_py as nfl
import asyncio
import requests

model = xgboost.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
team_stats = {}
training_state = {
    "ready": False,
    "progress": 0,
    "message": "Starting the model...",
}


def normalize_team_code(team_str):
    mapping = {
        "OAK": "LV",
        "SD": "LAC",
        "STL": "LAR",
        "PHO": "ARI",
    }
    return mapping.get(team_str, team_str)


def set_progress(progress, message):
    training_state["progress"] = progress
    training_state["message"] = message
    print(f"[{progress}%] {message}")


def background_training_task():
    global model, team_stats
    seasons = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

    try:
        set_progress(2, "Preparing play-by-play downloads...")
        frames = []
        for index, season in enumerate(seasons):
            percent = 5 + int((index / len(seasons)) * 60)
            set_progress(percent, f"Loading {season} play-by-play data...")
            frames.append(nfl.import_pbp_data([season]))

        set_progress(68, "Combining season data...")
        pbp_df = pd.concat(frames, ignore_index=True)
        pbp_df = pbp_df.dropna(subset=["epa", "posteam", "defteam"])
        pbp_df = pbp_df[pbp_df["play_type"].isin(["pass", "run"])]

        set_progress(75, "Building team offense and defense profiles...")
        off_epa = (
            pbp_df.groupby("posteam")["epa"]
            .mean()
            .reset_index()
            .rename(columns={"posteam": "team", "epa": "off_epa"})
        )
        def_epa = (
            pbp_df.groupby("defteam")["epa"]
            .mean()
            .reset_index()
            .rename(columns={"defteam": "team", "epa": "def_epa"})
        )
        team_profiles = pd.merge(off_epa, def_epa, on="team")

        for _, row in team_profiles.iterrows():
            normalized_name = normalize_team_code(row["team"])
            team_stats[normalized_name] = {
                "off_epa": float(row["off_epa"]),
                "def_epa": float(row["def_epa"]),
            }

        set_progress(82, "Loading regular-season schedules...")
        sched_df = nfl.import_schedules(seasons)
        sched_df = sched_df[sched_df["game_type"] == "REG"].dropna(subset=["home_score", "away_score"])

        set_progress(88, "Assembling training examples...")
        training_data = []
        for _, game in sched_df.iterrows():
            home = game["home_team"]
            away = game["away_team"]
            if home in team_stats and away in team_stats:
                h_profile = team_stats[home]
                a_profile = team_stats[away]
                features = [
                    h_profile["off_epa"],
                    h_profile["def_epa"],
                    a_profile["off_epa"],
                    a_profile["def_epa"],
                ]
                target = 1 if game["home_score"] > game["away_score"] else 0
                training_data.append(features + [target])

        dataset = pd.DataFrame(training_data, columns=["h_off", "h_def", "a_off", "a_def", "label"])
        X = dataset[["h_off", "h_def", "a_off", "a_def"]]
        y = dataset["label"]

        set_progress(94, "Fitting the XGBoost model...")
        model.fit(X, y)

        training_state["ready"] = True
        set_progress(100, "Model ready.")
    except Exception as error:
        training_state["ready"] = False
        set_progress(training_state["progress"], f"Training failed: {error}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI started. Training the model in the background...")
    task = asyncio.create_task(asyncio.to_thread(background_training_task))
    yield
    task.cancel()
    print("Clearing model state on shutdown...")
    team_stats.clear()
    training_state["ready"] = False


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok"}


# Aliased to handle both /api/status and /status
@app.get("/api/status")
@app.get("/status")
def model_status():
    return training_state


class PredictionRequest(BaseModel):
    homeTeam: str
    awayTeam: str


# Aliased to handle both /api/predict and /predict
@app.post("/api/predict")
@app.post("/predict")
def predict(req: PredictionRequest):
    if not training_state["ready"]:
        raise HTTPException(status_code=503, detail="Model is still training")

    team_one_res = team_stats.get(req.homeTeam, {"off_epa": 0.0, "def_epa": 0.0})
    team_two_res = team_stats.get(req.awayTeam, {"off_epa": 0.0, "def_epa": 0.0})

    input_features = np.array(
        [[team_one_res["off_epa"], team_one_res["def_epa"], team_two_res["off_epa"], team_two_res["def_epa"]]]
    )
    probabilities = model.predict_proba(input_features)[0]

    return {
        "home_team": req.homeTeam,
        "away_team": req.awayTeam,
        "home_win_probability": round(float(probabilities[1]) * 100, 2),
        "away_win_probability": round(float(probabilities[0]) * 100, 2),
        "predicted_winner": req.homeTeam if probabilities[1] > probabilities[0] else req.awayTeam,
        "metrics_context": {
            "home_off_epa_ranking": "Above Average" if team_one_res["off_epa"] > 0 else "Below Average",
            "away_off_epa_ranking": "Above Average" if team_two_res["off_epa"] > 0 else "Below Average",
        },
    }


# Cleaned score ticker endpoint (Returns 1 object per game instead of duplicate team objects)
@app.get("/api/scores")
@app.get("/api/scores/active-teams")
def get_scores():
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        data = requests.get(url, timeout=10).json()
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
                "id": event.get("id"),
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
    except Exception as e:
        print(f"CRITICAL TICKER ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch scoreboard: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)