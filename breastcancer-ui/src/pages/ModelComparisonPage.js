import React, { useState } from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const models = [
  { name: 'Cross-Attention\nFusion', short: 'CAF', acc: 96.8, auc: 0.981, sens: 95.2, spec: 97.1, f1: 96.4, prec: 97.6, color: '#1a8c6a' },
  { name: 'DenseNet201', short: 'DN201', acc: 91.4, auc: 0.942, sens: 88.7, spec: 93.2, f1: 90.8, prec: 92.9, color: '#3b82f6' },
  { name: 'ResNet50', short: 'RN50', acc: 89.7, auc: 0.921, sens: 86.3, spec: 92.4, f1: 88.9, prec: 91.5, color: '#8b5cf6' },
  { name: 'InceptionV3', short: 'IV3', acc: 90.1, auc: 0.930, sens: 87.4, spec: 92.1, f1: 89.5, prec: 91.6, color: '#f59e0b' },
  { name: 'MobileNetV2', short: 'MNV2', acc: 87.3, auc: 0.905, sens: 83.9, spec: 89.8, f1: 86.7, prec: 89.5, color: '#e8365d' },
];

const radarData = [
  { metric: 'Accuracy', CAF: 96.8, DN201: 91.4, RN50: 89.7 },
  { metric: 'AUC-ROC', CAF: 98.1, DN201: 94.2, RN50: 92.1 },
  { metric: 'Sensitivity', CAF: 95.2, DN201: 88.7, RN50: 86.3 },
  { metric: 'Specificity', CAF: 97.1, DN201: 93.2, RN50: 92.4 },
  { metric: 'F1-Score', CAF: 96.4, DN201: 90.8, RN50: 88.9 },
];

const barData = models.map(m => ({
  name: m.short,
  Accuracy: m.acc,
  'AUC-ROC': +(m.auc * 100).toFixed(1),
  'F1-Score': m.f1,
}));

export default function ModelComparisonPage() {
  const [activeTab, setActiveTab] = useState('table');

  return (
    <div>
      <div className="section-heading">Model Performance Comparison</div>
      <div className="section-sub">Benchmarking proposed Cross-Attention Fusion against image-only baselines on CBIS-DDSM</div>

      <div className="metrics-row" style={{ gridTemplateColumns: 'repeat(5, 1fr)', marginBottom: '1.5rem' }}>
        {[
          { label: 'Best Accuracy', val: '96.8%', note: 'Cross-Attention Fusion' },
          { label: 'Best AUC-ROC', val: '0.981', note: 'Cross-Attention Fusion' },
          { label: 'Best Sensitivity', val: '95.2%', note: 'Cross-Attention Fusion' },
          { label: 'Improvement', val: '+5.4%', note: 'vs DenseNet201' },
          { label: 'Models Tested', val: '5', note: 'CBIS-DDSM dataset' },
        ].map(m => (
          <div className="metric-card" key={m.label}>
            <div className="metric-value" style={{ color: 'var(--teal-600)' }}>{m.val}</div>
            <div className="metric-name">{m.label}</div>
            <div style={{ fontSize: 10, color: 'var(--slate-400)', marginTop: 3 }}>{m.note}</div>
          </div>
        ))}
      </div>

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            Performance Metrics
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {['table', 'bar', 'radar'].map(t => (
              <button key={t} onClick={() => setActiveTab(t)} style={{
                padding: '5px 12px', borderRadius: 6, border: '1px solid var(--slate-200)',
                background: activeTab === t ? 'var(--teal-50)' : 'var(--white)',
                color: activeTab === t ? 'var(--teal-700)' : 'var(--slate-500)',
                fontSize: 12, fontWeight: 500, cursor: 'pointer', fontFamily: 'Inter',
                textTransform: 'capitalize'
              }}>
                {t === 'bar' ? 'Bar Chart' : t === 'radar' ? 'Radar' : 'Table'}
              </button>
            ))}
          </div>
        </div>
        <div className="panel-body" style={{ padding: 0 }}>
          {activeTab === 'table' && (
            <table className="compare-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy (%)</th>
                  <th>AUC-ROC</th>
                  <th>Sensitivity (%)</th>
                  <th>Specificity (%)</th>
                  <th>F1-Score (%)</th>
                  <th>Precision (%)</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr key={m.short} className={i === 0 ? 'best-row' : ''}>
                    <td>
                      <span style={{ fontWeight: 600, color: 'var(--slate-800)' }}>{m.name.replace('\n', ' ')}</span>
                      {i === 0 && <span className="best-badge">Proposed</span>}
                    </td>
                    <td style={{ fontWeight: i === 0 ? 700 : 400, color: i === 0 ? 'var(--teal-600)' : 'inherit' }}>{m.acc}</td>
                    <td style={{ fontWeight: i === 0 ? 700 : 400, color: i === 0 ? 'var(--teal-600)' : 'inherit' }}>{m.auc}</td>
                    <td>{m.sens}</td>
                    <td>{m.spec}</td>
                    <td>{m.f1}</td>
                    <td>{m.prec}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {activeTab === 'bar' && (
            <div style={{ padding: '24px 8px' }}>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={barData} barGap={4} barSize={20}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--slate-100)" />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: 'var(--slate-500)' }} />
                  <YAxis domain={[80, 100]} tick={{ fontSize: 12, fill: 'var(--slate-500)' }} />
                  <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid var(--slate-200)', fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Accuracy" fill="#1a8c6a" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="AUC-ROC" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="F1-Score" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeTab === 'radar' && (
            <div style={{ padding: '24px 8px' }}>
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="var(--slate-200)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: 'var(--slate-500)' }} />
                  <Radar name="Cross-Attention Fusion" dataKey="CAF" stroke="#1a8c6a" fill="#1a8c6a" fillOpacity={0.15} strokeWidth={2} />
                  <Radar name="DenseNet201" dataKey="DN201" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} strokeWidth={2} />
                  <Radar name="ResNet50" dataKey="RN50" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.1} strokeWidth={2} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginTop: '1.5rem' }}>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">Architecture Summary</div>
          </div>
          <div className="panel-body" style={{ fontSize: 13, lineHeight: 1.8, color: 'var(--slate-600)' }}>
            <div style={{ marginBottom: 10 }}>
              <strong style={{ color: 'var(--teal-600)' }}>Cross-Attention Fusion (Proposed)</strong>
            </div>
            <ul style={{ paddingLeft: 16 }}>
              <li>Dual-stream: CNN image branch + ANN clinical branch</li>
              <li>Bidirectional cross-attention fusion layer</li>
              <li>Modality dropout for robustness</li>
              <li>Sigmoid binary classification head</li>
              <li>SHAP + Grad-CAM explainability integration</li>
            </ul>
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">Training Configuration</div>
          </div>
          <div className="panel-body">
            <table style={{ width: '100%', fontSize: 13 }}>
              {[
                ['Optimizer', 'Adam (lr=1e-4)'],
                ['Batch Size', '32'],
                ['Epochs', '100 (early stopping)'],
                ['Loss', 'Binary Cross-Entropy'],
                ['Train/Val/Test', '70% / 15% / 15%'],
                ['Data Augmentation', 'Flip, Rotate, Zoom'],
              ].map(([k, v]) => (
                <tr key={k} style={{ borderBottom: '1px solid var(--slate-100)' }}>
                  <td style={{ padding: '8px 0', color: 'var(--slate-500)', fontWeight: 500 }}>{k}</td>
                  <td style={{ padding: '8px 0', color: 'var(--slate-800)', fontWeight: 600, textAlign: 'right' }}>{v}</td>
                </tr>
              ))}
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
