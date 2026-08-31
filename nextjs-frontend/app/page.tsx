"use client";
import React, { useState, ChangeEvent } from 'react';

interface PredictionResult {
  home_team: string;
  away_team: string;
  home_win_probability: number;
  away_win_probability: number;
  predicted_winner: string;
}

// All 32 active NFL teams mapped to standard data abbreviations
const NFL_TEAMS: string[] = [
  'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
  'DAL', 'DEN', 'DET', 'GB',  'HOU', 'IND', 'JAX', 'KC',
  'LA',  'LAC', 'LV',  'MIA', 'MIN', 'NE',  'NO',  'NYG',
  'NYJ', 'PHI', 'PIT', 'SEA', 'SF',  'TB',  'TEN', 'WAS'
].sort(); // Kept organized alphabetically for easier UI dropdown scanning

export default function Home() {
  const [homeTeam, setHomeTeam] = useState<string>('KC');
  const [awayTeam, setAwayTeam] = useState<string>('SF');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handlePredict = async (): Promise<void> => {
    if (homeTeam === awayTeam) {
      alert("Teams must be different!");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5001/api/express/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ homeTeam, awayTeam }),
      });
      
      if (!response.ok) {
        throw new Error('Express API network failure');
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Failed to connect to Express server:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
      <div className="bg-white max-w-xl w-full p-8 rounded-2xl shadow-2xl">
        <h1 className="text-2xl font-black text-center text-slate-800 mb-6">🔀 Next.js + Express NFL Predictor</h1>
        
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-xs font-bold uppercase text-gray-500 mb-1">Home Team</label>
            <select 
              value={homeTeam} 
              onChange={(e: ChangeEvent<HTMLSelectElement>) => setHomeTeam(e.target.value)} 
              className="w-full p-3 border rounded-lg bg-gray-50 outline-none text-slate-800"
            >
              {NFL_TEAMS.map(team => <option key={team} value={team}>{team}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-bold uppercase text-gray-500 mb-1">Away Team</label>
            <select 
              value={awayTeam} 
              onChange={(e: ChangeEvent<HTMLSelectElement>) => setAwayTeam(e.target.value)} 
              className="w-full p-3 border rounded-lg bg-gray-50 outline-none text-slate-800"
            >
              {NFL_TEAMS.map(team => <option key={team} value={team}>{team}</option>)}
            </select>
          </div>
        </div>

        <button 
          onClick={handlePredict} 
          disabled={loading} 
          className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl transition duration-150 disabled:bg-gray-300"
        >
          {loading ? 'Querying Express Backend...' : 'Calculate Game Odds'}
        </button>

        {result && (
          <div className="mt-6 p-5 bg-indigo-50 rounded-xl border border-indigo-100 text-slate-700">
            <h3 className="text-lg font-bold mb-2">🏆 Predicted Winner: <span className="text-indigo-600">{result.predicted_winner}</span></h3>
            <div className="space-y-1">
              <p><strong>{result.home_team} Win Chance:</strong> {result.home_win_probability}%</p>
              <p><strong>{result.away_team} Win Chance:</strong> {result.away_win_probability}%</p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
