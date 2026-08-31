from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost
import nfl_data_py as nfl
import asyncio

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
        "STL": "LA",
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


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.get("/status")
def model_status():
    return training_state


class PredictionRequest(BaseModel):
    Team_one: str
    Team_two: str


@app.post("/predict")
def predict(req: PredictionRequest):
    if not training_state["ready"]:
        raise HTTPException(status_code=503, detail="Model is still training")

    team_one_res = team_stats.get(req.Team_one, {"off_epa": 0.0, "def_epa": 0.0})
    team_two_res = team_stats.get(req.Team_two, {"off_epa": 0.0, "def_epa": 0.0})

    input_features = np.array(
        [[team_one_res["off_epa"], team_one_res["def_epa"], team_two_res["off_epa"], team_two_res["def_epa"]]]
    )
    probabilities = model.predict_proba(input_features)[0]

    return {
        "home_team": req.Team_one,
        "away_team": req.Team_two,
        "home_win_probability": round(float(probabilities[1]) * 100, 2),
        "away_win_probability": round(float(probabilities[0]) * 100, 2),
        "predicted_winner": req.Team_one if probabilities[1] > probabilities[0] else req.Team_two,
        "metrics_context": {
            "home_off_epa_ranking": "Above Average" if team_one_res["off_epa"] > 0 else "Below Average",
            "away_off_epa_ranking": "Above Average" if team_two_res["off_epa"] > 0 else "Below Average",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
