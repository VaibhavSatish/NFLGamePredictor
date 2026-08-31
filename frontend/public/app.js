const NFL_TEAMS = [
  "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
  "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
  "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
  "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
].sort();

const homeSelect = document.getElementById("homeTeam");
const awaySelect = document.getElementById("awayTeam");
const predictBtn = document.getElementById("predictBtn");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const loadingEl = document.getElementById("loading");
const appEl = document.getElementById("app");

function fillSelect(select, selected) {
  select.innerHTML = NFL_TEAMS.map(
    (team) => `<option value="${team}" ${team === selected ? "selected" : ""}>${team}</option>`
  ).join("");
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
  resultEl.hidden = true;
}

function showResult(data) {
  errorEl.hidden = true;
  resultEl.hidden = false;
  resultEl.innerHTML = `
    <h3>Predicted Winner: <span class="winner">${data.predicted_winner}</span></h3>
    <p><strong>${data.home_team} Win Chance:</strong> ${data.home_win_probability}%</p>
    <p><strong>${data.away_team} Win Chance:</strong> ${data.away_win_probability}%</p>
  `;
}

async function isModelReady() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    return data.ready === true;
  } catch {
    return false;
  }
}

async function waitForModel() {
  while (!(await isModelReady())) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  loadingEl.hidden = true;
  appEl.hidden = false;
}

fillSelect(homeSelect, "KC");
fillSelect(awaySelect, "SF");
waitForModel();

predictBtn.addEventListener("click", async () => {
  const homeTeam = homeSelect.value;
  const awayTeam = awaySelect.value;

  if (homeTeam === awayTeam) {
    showError("Teams must be different.");
    return;
  }

  predictBtn.disabled = true;
  predictBtn.innerHTML = '<span class="btn-spinner"></span>Calculating...';
  errorEl.hidden = true;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ homeTeam, awayTeam }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }
    showResult(data);
  } catch (err) {
    showError(err.message || "Failed to connect to the predictor.");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Calculate Game Odds";
  }
});
