const express = require("express");
const axios = require("axios");
const cors = require("cors");
const app = express();
app.use(cors({ origin: "http://localhost:3000" }));
app.use(express.json())

const PYTHON_API_URL = 'http://0.0.0.0:8000';
app.post("/api/express/predict", async (req, res) => {
    try {
        const {homeTeam, awayTeam} = req.body; 
        const response = await axios.post(`${PYTHON_API_URL}/predict`, {Team_one: homeTeam, Team_two: awayTeam});
        res.json(response.data); 
    } catch (error) {
        console.error("Error communicating with ML engine:", error.message);
        res.status(500).json({ error: "Failed to generate prediction" });
    }
});

app.listen(5001, () => console.log('🚀 Node.js gateway running on port 5000')); 