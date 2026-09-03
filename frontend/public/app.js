const NFL_TEAMS = [
  "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
  "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
  "LAR", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
  "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
].sort();

// Used only for client-side styling (bar colors, select tinting).
// The API never returns team colors, so this stays local to the UI.
const TEAM_COLORS = {
  ARI: "#97233F", ATL: "#A71930", BAL: "#241773", BUF: "#00338D",
  CAR: "#0085CA", CHI: "#0B162A", CIN: "#FB4F14", CLE: "#311D00",
  DAL: "#003594", DEN: "#FB4F14", DET: "#0076B6", GB: "#203731",
  HOU: "#03202F", IND: "#002C5F", JAX: "#006778", KC: "#E31837",
  LAR: "#003594", LAC: "#0080C6", LV: "#A5ACAF", MIA: "#008E97",
  MIN: "#4F2683", NE: "#002244", NO: "#D3BC8D", NYG: "#0B2265",
  NYJ: "#125740", PHI: "#004C54", PIT: "#FFB612", SEA: "#002244",
  SF: "#AA0000", TB: "#D50A0A", TEN: "#4B92DB", WAS: "#5A1414",
};
function teamColor(abbr) {
  return TEAM_COLORS[abbr] || "#d7a13b";
}

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
const predictBtnLabel = predictBtn.textContent;

function fillSelect(select, selected) {
  select.innerHTML = NFL_TEAMS.map(
    (team) => `<option value="${team}" ${team === selected ? "selected" : ""}>${team}</option>`
  ).join("");
}

function tintSelect(select) {
  select.style.borderColor = teamColor(select.value);
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
  resultEl.hidden = true;
}

function rankingLabel(text) {
  return text || "No data";
}

function showResult(data) {
  errorEl.hidden = true;
  resultEl.hidden = false;

  const homeAbbr = data.home_team;
  const awayAbbr = data.away_team;
  const ctx = data.metrics_context || {};

  const hasProjection =
    data.home_projected_score !== undefined && data.away_projected_score !== undefined;

  resultEl.innerHTML = `
    <h3>Predicted winner: <span class="winner">${data.predicted_winner}</span></h3>
    ${hasProjection ? buildScoreLine(homeAbbr, data.home_projected_score, awayAbbr, data.away_projected_score) : ""}
    ${buildProbRow(homeAbbr, data.home_win_probability, `Offense: ${rankingLabel(ctx.home_off_epa_ranking)}`)}
    ${buildProbRow(awayAbbr, data.away_win_probability, `Offense: ${rankingLabel(ctx.away_off_epa_ranking)}`)}
    <p class="result-note">Based on scoring-play efficiency (EPA) trends, not a guarantee of outcome.</p>
  `;

  requestAnimationFrame(() => {
    resultEl.querySelectorAll(".prob-fill").forEach((fill) => {
      fill.style.width = `${fill.getAttribute("data-target")}%`;
    });
  });
}

function buildScoreLine(homeAbbr, homeScore, awayAbbr, awayScore) {
  return `
    <p class="result-score">
      ${homeAbbr} ${homeScore}<span class="sep">&ndash;</span>${awayScore} ${awayAbbr}
    </p>
  `;
}

function buildProbRow(abbr, pctValue, metricText) {
  const pct = Number(pctValue) || 0;
  return `
    <div class="prob-row">
      <div class="prob-row-top">
        <span class="team-name">${abbr}</span>
        <div class="prob-track">
          <div class="prob-fill" data-target="${pct}" style="background:${teamColor(abbr)}"></div>
        </div>
        <span class="prob-value">${pct}%</span>
      </div>
      <span class="metric-tag">${metricText}</span>
    </div>
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
tintSelect(homeSelect);
tintSelect(awaySelect);
homeSelect.addEventListener("change", () => tintSelect(homeSelect));
awaySelect.addEventListener("change", () => tintSelect(awaySelect));

waitForModel();

predictBtn.addEventListener("click", async () => {
  const homeTeam = homeSelect.value;
  const awayTeam = awaySelect.value;

  if (homeTeam === awayTeam) {
    showError("Pick two different teams to run a simulation.");
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
      throw new Error(data.error || data.detail || "Prediction failed");
    }
    showResult(data);
  } catch (err) {
    showError(err.message || "Failed to connect to the predictor.");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = predictBtnLabel;
  }
});

// --- Live score ticker ---
async function initScoreTicker() {
  const tickerContainer = document.getElementById("ticker-content");
  if (!tickerContainer) return;

  try {
    const response = await fetch("/api/scores");
    const games = await response.json();

    if (!Array.isArray(games) || games.length === 0) {
      tickerContainer.innerHTML = "<div class='game-card'>No active games scheduled this week</div>";
      return;
    }

    const tickerHTML = games
      .map((game) => {
        const isLive = Boolean(game.isLive);
        const hasStarted = Boolean(game.hasStarted);
        const statusText = game.status || "Scheduled";

        const scoreMarkup = hasStarted
          ? `<span>${game.awayTeam} ${game.awayScore} &ndash; ${game.homeScore} ${game.homeTeam}</span>`
          : `<span>${game.awayTeam} @ ${game.homeTeam}</span>`;

        return `
          <div class="game-card ${isLive ? "is-live" : ""}">
            ${scoreMarkup}
            <span class="ticker-status">${statusText}</span>
          </div>
        `;
      })
      .join("");

    const computedDuration = Math.max(35, games.length * 6);
    tickerContainer.style.animationDuration = `${computedDuration}s`;

    // Duplicate once for a seamless loop.
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