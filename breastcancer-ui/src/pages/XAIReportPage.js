import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const shapData = [
  { feature: 'Image Pattern (CNN)', shap: 0.42, direction: 'positive' },
  { feature: 'Mass Margin', shap: 0.31, direction: 'positive' },
  { feature: 'Breast Density', shap: 0.13, direction: 'positive' },
  { feature: 'Calcification Type', shap: 0.09, direction: 'positive' },
  { feature: 'Patient Age', shap: 0.06, direction: 'positive' },
  { feature: 'Mass Shape', shap: -0.04, direction: 'negative' },
  { feature: 'Calc Distribution', shap: -0.07, direction: 'negative' },
];

const GradCamMock = () => (
  <div style={{ position: 'relative', background: 'var(--slate-100)', borderRadius: 10, overflow: 'hidden', aspectRatio: '1/1', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <svg width="100%" height="100%" viewBox="0 0 200 200">
      <defs>
        <radialGradient id="heatmap1" cx="45%" cy="40%" r="35%">
          <stop offset="0%" stopColor="#e8365d" stopOpacity="0.85"/>
          <stop offset="50%" stopColor="#f59e0b" stopOpacity="0.5"/>
          <stop offset="100%" stopColor="#1a8c6a" stopOpacity="0.1"/>
        </radialGradient>
        <radialGradient id="heatmap2" cx="60%" cy="55%" r="20%">
          <stop offset="0%" stopColor="#e8365d" stopOpacity="0.7"/>
          <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.1"/>
        </radialGradient>
      </defs>
      <rect width="200" height="200" fill="var(--slate-200)"/>
      <ellipse cx="100" cy="100" rx="75" ry="85" fill="var(--slate-400)" opacity="0.4"/>
      <ellipse cx="90" cy="80" rx="25" ry="30" fill="var(--slate-500)" opacity="0.5"/>
      <circle cx="90" cy="78" r="12" fill="var(--slate-600)" opacity="0.4"/>
      <rect width="200" height="200" fill="url(#heatmap1)" opacity="0.7"/>
      <rect width="200" height="200" fill="url(#heatmap2)" opacity="0.6"/>
      <text x="100" y="190" textAnchor="middle" fontSize="10" fill="var(--slate-700)" fontFamily="Inter, sans-serif" fontWeight="500">Grad-CAM Activation Map</text>
    </svg>
    <div style={{ position: 'absolute', bottom: 24, right: 8, display: 'flex', flexDirection: 'column', gap: 2, alignItems: 'center' }}>
      {['#e8365d','#f59e0b','#22c55e','#3b82f6'].map((c,i) => (
        <div key={i} style={{ width: 12, height: 12, background: c, borderRadius: 2, opacity: 0.85 }}/>
      ))}
      <span style={{ fontSize: 9, color: 'var(--slate-500)', marginTop: 2 }}>High→Low</span>
    </div>
  </div>
);

export default function XAIReportPage() {
  const [showRaw, setShowRaw] = useState(false);

  return (
    <div>
      <div className="section-heading">Explainability Report</div>
      <div className="section-sub">SHAP feature attributions and Grad-CAM activation maps for model transparency</div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
        {/* SHAP Panel */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
              SHAP Feature Importance
            </div>
            <div style={{ fontSize: 11, color: 'var(--slate-400)', background: 'var(--slate-100)', padding: '3px 8px', borderRadius: 6 }}>
              Prediction: Malignant
            </div>
          </div>
          <div className="panel-body">
            <div style={{ fontSize: 12, color: 'var(--slate-500)', marginBottom: 14 }}>
              SHAP values quantify each feature's contribution to moving the prediction away from the base rate.
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={shapData} layout="vertical" margin={{ left: 0, right: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--slate-100)" horizontal={false} />
                <XAxis type="number" domain={[-0.1, 0.5]} tick={{ fontSize: 11, fill: 'var(--slate-500)' }} tickFormatter={v => v.toFixed(1)} />
                <YAxis type="category" dataKey="feature" width={145} tick={{ fontSize: 11, fill: 'var(--slate-500)' }} />
                <Tooltip
                  formatter={(v) => [v.toFixed(3), 'SHAP Value']}
                  contentStyle={{ borderRadius: 8, border: '1px solid var(--slate-200)', fontSize: 12 }}
                />
                <Bar dataKey="shap" radius={[0, 4, 4, 0]}>
                  {shapData.map((entry, i) => (
                    <Cell key={i} fill={entry.direction === 'positive' ? '#e8365d' : '#1a8c6a'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', gap: 16, marginTop: 10 }}>
              {[['#e8365d', 'Increases malignancy risk'], ['#1a8c6a', 'Decreases risk']].map(([c, l]) => (
                <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--slate-500)' }}>
                  <div style={{ width: 10, height: 10, borderRadius: 2, background: c }} />
                  {l}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Grad-CAM Panel */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
              Grad-CAM Activation
            </div>
            <button
              onClick={() => setShowRaw(v => !v)}
              style={{ fontSize: 11, background: 'var(--slate-100)', border: 'none', borderRadius: 6, padding: '4px 10px', color: 'var(--slate-600)', cursor: 'pointer', fontFamily: 'Inter' }}
            >
              {showRaw ? 'Show Overlay' : 'Show Raw'}
            </button>
          </div>
          <div className="panel-body">
            <div style={{ fontSize: 12, color: 'var(--slate-500)', marginBottom: 14 }}>
              Gradient-weighted Class Activation Maps highlight image regions driving the malignancy prediction.
            </div>
            <GradCamMock />
            <div style={{ marginTop: 14, padding: '10px 14px', background: 'var(--rose-50)', border: '1px solid var(--rose-100)', borderRadius: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--rose-600)', marginBottom: 4 }}>High-activation region detected</div>
              <div style={{ fontSize: 12, color: 'var(--rose-700)' }}>
                Spiculated mass at upper-outer quadrant (coordinates: ~45%, 40%) — primary driver of malignant classification.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* LIME Panel */}
      <div className="panel" style={{ marginBottom: '1.5rem' }}>
        <div className="panel-header">
          <div className="panel-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>
            LIME Local Explanation
          </div>
          <span style={{ fontSize: 11, background: 'var(--amber-50)', color: 'var(--amber-600)', border: '1px solid var(--amber-100)', padding: '3px 8px', borderRadius: 6, fontWeight: 600 }}>
            Patch-level Analysis
          </span>
        </div>
        <div className="panel-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
            {[
              { label: 'Superpixels', val: '50', sub: 'image segments' },
              { label: 'Perturb Samples', val: '1000', sub: 'perturbations' },
              { label: 'Top Features', val: '8', sub: 'selected' },
              { label: 'Local Accuracy', val: '94.2%', sub: 'explanation fit' },
            ].map(m => (
              <div className="metric-card" key={m.label}>
                <div className="metric-value">{m.val}</div>
                <div className="metric-name">{m.label}</div>
                <div style={{ fontSize: 10, color: 'var(--slate-400)', marginTop: 2 }}>{m.sub}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 16, fontSize: 12, color: 'var(--slate-500)', background: 'var(--slate-50)', borderRadius: 8, padding: '12px 14px', lineHeight: 1.8, borderLeft: '3px solid var(--teal-300)' }}>
            LIME explanation: The model focuses on <strong style={{ color: 'var(--slate-700)' }}>superpixels 12, 18, 31</strong> (mass boundary region) as the primary indicators of malignancy. Clinical metadata — particularly <strong style={{ color: 'var(--slate-700)' }}>mass margin (Spiculated)</strong> and <strong style={{ color: 'var(--slate-700)' }}>breast density (ACR C)</strong> — reinforce this through the cross-attention fusion pathway.
          </div>
        </div>
      </div>

      {/* Export */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">Export Report</div>
        </div>
        <div className="panel-body" style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            ['PDF Clinical Report', 'var(--teal-500)', 'white'],
            ['SHAP Summary JSON', 'var(--slate-100)', 'var(--slate-700)'],
            ['Grad-CAM Images (ZIP)', 'var(--slate-100)', 'var(--slate-700)'],
            ['Full XAI Bundle', 'var(--slate-100)', 'var(--slate-700)'],
          ].map(([label, bg, color]) => (
            <button key={label} style={{
              padding: '10px 18px', background: bg, color, borderRadius: 8,
              border: '1px solid var(--slate-200)', fontSize: 13, fontWeight: 600,
              cursor: 'pointer', fontFamily: 'Inter', display: 'flex', alignItems: 'center', gap: 8
            }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
