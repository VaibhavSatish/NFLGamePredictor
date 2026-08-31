# NFL Game Predictor

FastAPI trains an XGBoost model on NFL play-by-play data. Express serves a simple HTML UI and forwards predictions to FastAPI.

## Run locally

```bash
npm install
npm install --prefix frontend
uv sync
npm run dev
```

- UI: http://localhost:5001
- FastAPI: http://localhost:8000

## TODOs
- [ ] Add Team Logos
- [ ] Tune the model to provide better results
- [ ] Add scores from current games to UI
- [ ] Deploy Predictor
- [ ] Add Auth system