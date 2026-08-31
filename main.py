import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost 
import nfl_data_py as nfl

app = FastAPI()
model = xgboost.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
team_stats = {}

@app.on_event("startup")
def train():
    global model, team_stats
    pbp_df = nfl.import_pbp_data([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    pbp_df = pbp_df.dropna(subset=['epa', 'posteam', 'defteam'])
    pbp_df = pbp_df[pbp_df['play_type'].isin(['pass', 'run'])]
    off_epa = pbp_df.groupby('posteam')['epa'].mean().reset_index().rename(columns={'posteam': 'team', 'epa': 'off_epa'})
    def_epa = pbp_df.groupby('defteam')['epa'].mean().reset_index().rename(columns={'defteam': 'team', 'epa': 'def_epa'})
    team_profiles = pd.merge(off_epa, def_epa, on='team')
    for _, row in team_profiles.iterrows():
        team_stats[row['team']] = {
            "off_epa": row['off_epa'],
            "def_epa": row['def_epa']
        }
    sched_df = nfl.import_schedules([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    sched_df = sched_df[sched_df['game_type'] == 'REG'].dropna(subset=['home_score', 'away_score'])
    training_data = []
    
    for _, game in sched_df.iterrows():
        home = game['home_team']
        away = game['away_team']
        if home in team_stats and away in team_stats:
            h_profile = team_stats[home]
            a_profile = team_stats[away]
            
            features = [
                h_profile['off_epa'], 
                h_profile['def_epa'],
                a_profile['off_epa'],
                a_profile['def_epa']
            ]
            target = 1 if game['home_score'] > game['away_score'] else 0
            
            training_data.append(features + [target])
    dataset = pd.DataFrame(training_data, columns=['h_off', 'h_def', 'a_off', 'a_def', 'label'])
    
    X = dataset[['h_off', 'h_def', 'a_off', 'a_def']]
    y = dataset['label']
    model.fit(X, y)

@app.get("/")
def health_check():
    return {"status": "ok"}

class PredictionRequest(BaseModel):
    Team_one:str
    Team_two:str

@app.get("/predict")
def predict(req: PredictionRequest):
    team_one_res = team_stats.get(req.Team_one, {0.0, 0.0})
    team_two_res = team_stats.get(req.Team_two, {0.0, 0.0})
    input_features = np.array([[team_one_res['off_epa'], team_one_res['def_epa'], team_two_res['off_epa'], team_two_res['def_epa']]])
    probabilities = model.predict_proba(input_features)[0]
    return {
        "home_team": req.Team_one,
        "away_team": req.Team_two,
        "home_win_probability": round(float(probabilities[1]) * 100, 2),
        "away_win_probability": round(float(probabilities[0]) * 100, 2),
        "predicted_winner": req.Team_one if probabilities[1] > probabilities[0] else req.Team_two,
        "metrics_context": {
            "home_off_epa_ranking": "Above Average" if team_one_res['off_epa'] > 0 else "Below Average",
            "away_off_epa_ranking": "Above Average" if team_two_res['off_epa'] > 0 else "Below Average"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)