const path = require("path");
const express = require("express");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 5001;
const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://127.0.0.1:8000";

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// FastAPI only binds its port once the model has finished training, so an
// unreachable health check means the model is still warming up.
app.get("/api/status", async (req, res) => {
  try {
    await axios.get(`${PYTHON_API_URL}/`, { timeout: 2000 });
    res.json({ ready: true });
  } catch {
    res.json({ ready: false });
  }
});

app.post("/api/predict", async (req, res) => {
  try {
    const { homeTeam, awayTeam } = req.body;
    const response = await axios.post(`${PYTHON_API_URL}/predict`, {
      Team_one: homeTeam,
      Team_two: awayTeam,
    });
    res.json(response.data);
  } catch (error) {
    console.error("Error communicating with ML engine:", error.message);
    res.status(500).json({ error: "Failed to generate prediction" });
  }
});

app.listen(PORT, () => {
  console.log(`NFL predictor UI running at http://localhost:${PORT}`);
});
