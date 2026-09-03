const NFL_TEAMS = [
  "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
  "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
  "LAR", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
  "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
].sort();

const homeSelect = document.getElementById("homeTeam");
const awaySelect = document.getElementById("awayTeam");
const predictBtn = document.getElementById("predictBtn");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const loadingEl = document.getElementById("loading");
const appEl = document.getElementById("app");
const progressBar = document.getElementById("progressBar");
const loadingPercent = document.getElementById("loadingPercent");
const loadingHint = document.getElementById("loadingHint");
const progressEl = document.querySelector(".progress");

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

function updateProgress(data) {
  const progress = Math.max(0, Math.min(100, Number(data.progress) || 0));
  progressBar.style.width = `${progress}%`;
  loadingPercent.textContent = `${progress}%`;
  loadingHint.textContent = data.message || "Training the model...";
  if (progressEl) {
    progressEl.setAttribute("aria-valuenow", String(progress));
  }
}

async function getModelStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    return await response.json();
  } catch {
    return { ready: false, progress: 0, message: "Waiting for the model server..." };
  }
}

function revealApp() {
  loadingEl.hidden = true;
  appEl.hidden = false;
}

async function waitForModel() {
  let status = await getModelStatus();
  updateProgress(status);
  while (!status.ready) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    status = await getModelStatus();
    updateProgress(status);
  }
  updateProgress({ progress: 100, message: "Model ready." });
  revealApp();
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

// --- LIVE TICKER INTEGRATION WITH DEDUPLICATION ---
async function initScoreTicker() {
  const tickerContainer = document.getElementById("ticker-content");
  if (!tickerContainer) return;

  try {
    const response = await fetch("/api/scores");
    const rawData = await response.json();

    if (!rawData || rawData.length === 0) {
      tickerContainer.innerHTML = "<div class='game-card'>No active games scheduled this week</div>";
      return;
    }

    // 1. Filter out duplicate entries for the same matchup
    const seenMatchups = new Set();
    const uniqueGames = rawData.filter((item) => {
      // If server provides a game ID, use it directly
      if (item.id) {
        if (seenMatchups.has(item.id)) return false;
        seenMatchups.add(item.id);
        return true;
      }

      // Fallback: Parse teams and build a unique pair key (e.g. "NE-SEA")
      const rawTeam = item.team || item.homeTeam || "";
      const contextStr = item.context || "";
      const otherTeam = item.awayTeam || contextStr.replace(/^(vs|@|\s)+/i, "").trim();

      const matchupKey = [rawTeam, otherTeam].filter(Boolean).sort().join("-");
      if (!matchupKey || seenMatchups.has(matchupKey)) {
        return false;
      }
      seenMatchups.add(matchupKey);
      return true;
    });

    // 2. Build ticker HTML from deduplicated games list
    const tickerHTML = uniqueGames.map((item) => {
      let home = item.homeTeam;
      let away = item.awayTeam;

      // Extract home/away from legacy team/context structure if necessary
      if (!home || !away) {
        const rawTeam = item.team || "TBD";
        const contextStr = item.context || "";
        const otherTeam = contextStr.replace(/^(vs|@|\s)+/i, "").trim();

        if (contextStr.trim().startsWith("@")) {
          away = rawTeam;
          home = otherTeam || "TBD";
        } else {
          home = rawTeam;
          away = otherTeam || "TBD";
        }
      }

      const isLive = Boolean(item.isLive || item.rawStatus === "STATUS_IN_PROGRESS");
      const hasStarted = Boolean(item.hasStarted || (item.rawStatus && item.rawStatus !== "STATUS_SCHEDULED"));
      const statusText = item.status || "Scheduled";

      let scoreMarkup = "";
      if (hasStarted) {
        if (item.awayScore !== undefined && item.homeScore !== undefined) {
          scoreMarkup = `<span class="ticker-score">${away} ${item.awayScore} - ${item.homeScore} ${home}</span>`;
        } else {
          scoreMarkup = `<span class="ticker-score">${home} ${item.score || 0}</span>`;
        }
      } else {
        scoreMarkup = `<span class="ticker-matchup">${away} @ ${home}</span>`;
      }

      return `
        <div class="game-card ${isLive ? "is-live" : ""}">
          ${scoreMarkup}
          <span class="ticker-status">(${statusText})</span>
        </div>
      `;
    }).join('');

    // 3. Smooth, readable animation duration
    const computedDuration = Math.max(35, uniqueGames.length * 6); 
    tickerContainer.style.animationDuration = `${computedDuration}s`;

    // Clone HTML once for smooth infinite loop
    tickerContainer.innerHTML = tickerHTML + tickerHTML;

  } catch (error) {
    console.error("Failed to load live score ticker:", error);
    tickerContainer.innerHTML = "<div class='game-card'>Scores temporarily unavailable</div>";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initScoreTicker();
  setInterval(initScoreTicker, 30000);
});