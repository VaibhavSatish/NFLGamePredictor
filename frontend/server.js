const path = require("path");
const express = require("express");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 5001;
const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://127.0.0.1:8000";

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.get("/api/status", async (req, res) => {
  res.set("Cache-Control", "no-store");
  try {
    const response = await axios.get(`${PYTHON_API_URL}/status`, { timeout: 2000 });
    res.json({
      ready: response.data?.ready === true,
      progress: Number(response.data?.progress) || 0,
      message: response.data?.message || "Training the model...",
    });
  } catch {
    res.json({
      ready: false,
      progress: 0,
      message: "Waiting for the model server...",
    });
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

// Express 5 runs this callback even when the bind failed, so confirm the
// socket is actually listening before claiming the server is up.
const server = app.listen(PORT, () => {
  if (server.listening) {
    console.log(`NFL predictor UI running at http://localhost:${PORT}`);
  }
});

server.on("error", (error) => {
  if (error.code === "EADDRINUSE") {
    console.error(
      `Port ${PORT} is already in use. Stop the other server, or start this one with a different PORT.`
    );
    process.exit(1);
  }
  throw error;
});
