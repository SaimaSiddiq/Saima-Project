import React, { useState, useEffect } from 'react';
import DiagnosisPage from './pages/DiagnosisPage';
import ModelComparisonPage from './pages/ModelComparisonPage';
import XAIReportPage from './pages/XAIReportPage';
import './App.css';

export default function App() {
  const [activePage, setActivePage] = useState('diagnosis');
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');

  useEffect(() => {
    try {
      document.body.classList.toggle('theme-dark', theme === 'dark');
      localStorage.setItem('theme', theme);
    } catch (e) {
      // ignore (server-side rendering / restricted envs)
    }
  }, [theme]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/>
              <path d="M8 12h8M12 8v8"/>
            </svg>
          </div>
          <div>
            <div className="brand-name">MammoXplain</div>
            <div className="brand-tagline">Explainable Multimodal Breast Cancer Diagnosis</div>
          </div>
        </div>
        <nav className="topnav">
          <button
            className={`nav-btn ${activePage === 'diagnosis' ? 'active' : ''}`}
            onClick={() => setActivePage('diagnosis')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M9 12l2 2 4-4"/></svg>
            Diagnosis
          </button>
          <button
            className={`nav-btn ${activePage === 'comparison' ? 'active' : ''}`}
            onClick={() => setActivePage('comparison')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            Model Comparison
          </button>
          <button
            className={`nav-btn ${activePage === 'xai' ? 'active' : ''}`}
            onClick={() => setActivePage('xai')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35M11 8v6M8 11h6"/></svg>
            XAI Report
          </button>
        </nav>
        <button
          className="theme-toggle"
          onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
          )}
        </button>
        <div className="topbar-meta">
          <span className="status-dot"></span>
          <span className="status-label">Flask API Connected</span>
        </div>
      </header>

      <main className="page-content">
        {activePage === 'diagnosis' && <DiagnosisPage />}
        {activePage === 'comparison' && <ModelComparisonPage />}
        {activePage === 'xai' && <XAIReportPage />}
      </main>
    </div>
  );
}
