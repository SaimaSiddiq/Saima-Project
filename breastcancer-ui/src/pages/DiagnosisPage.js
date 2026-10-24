import React, { useState, useRef } from 'react';

const MODELS = [
  { id: 'crossattn', name: 'Cross-Attention Fusion', desc: 'Multimodal (Proposed)', acc: 96.8 },
  { id: 'densenet', name: 'DenseNet201', desc: 'Image-only baseline', acc: 91.4 },
  { id: 'resnet', name: 'ResNet50', desc: 'Image-only baseline', acc: 89.7 },
  { id: 'mobilenet', name: 'MobileNetV2', desc: 'Lightweight CNN', acc: 87.3 },
  { id: 'inception', name: 'InceptionV3', desc: 'Deep CNN baseline', acc: 90.1 },
];

const FEATURE_COLORS = {
  high: '#e8365d',
  medium: '#f59e0b',
  low: '#1a8c6a',
};

function ConfidenceArc({ score, type }) {
  const radius = 52;
  const circumference = Math.PI * radius;
  const fill = (score / 100) * circumference;
  const color = type === 'malignant' ? '#e8365d' : '#1a8c6a';

  return (
    <div className="conf-arc-wrap">
      <svg width="140" height="80" viewBox="0 0 140 80">
        <path
          d={`M 14 76 A ${radius} ${radius} 0 0 1 126 76`}
          fill="none" stroke="var(--slate-200)" strokeWidth="10" strokeLinecap="round"
        />
        <path
          d={`M 14 76 A ${radius} ${radius} 0 0 1 126 76`}
          fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={`${fill} ${circumference}`}
          style={{ transition: 'stroke-dasharray 1s ease' }}
        />
        <text x="70" y="62" textAnchor="middle" fontSize="22" fontWeight="700" fill="var(--slate-900)">
          {score}%
        </text>
      </svg>
      <div className="conf-label">Diagnostic Confidence</div>
    </div>
  );
}

export default function DiagnosisPage() {
  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';
  const [selectedModel, setSelectedModel] = useState('crossattn');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef();

  const [clinicalData, setClinicalData] = useState({
    massMargin: '',
    massShape: '',
    breastDensity: '',
    calcType: '',
    calcDistribution: '',
    patientAge: '',
  });

  const handleImageDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) loadImage(file);
  };

  const loadImage = (file) => {
    setImageFile(file);
    setError('');
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleDiagnose = async () => {
    if (!imageFile) return;
    setLoading(true);

    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('model', selectedModel);
      formData.append('clinical', JSON.stringify(clinicalData));

      const res = await fetch(`${API_BASE_URL}/api/diagnose`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.error || 'Diagnosis request failed');
      }

      setResult(data);
    } catch (err) {
      setError(err?.message || 'Unable to connect to backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="section-heading">Diagnostic Workstation</div>
      <div className="section-sub">Upload mammogram and enter clinical metadata for real-time AI-assisted diagnosis</div>

      <div className="dashboard-grid">
        {/* ── LEFT: Model Selector ── */}
        <div>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93A10 10 0 0 0 12 2v2a8 8 0 0 1 8 8h2a10 10 0 0 0-2.93-7.07z"/></svg>
                Active Model
              </div>
            </div>
            <div className="panel-body">
              {MODELS.map((m) => (
                <div
                  key={m.id}
                  className={`model-card ${selectedModel === m.id ? 'selected' : ''}`}
                  onClick={() => setSelectedModel(m.id)}
                >
                  <div className="model-card-left">
                    <div className="model-name">{m.name}</div>
                    <div className="model-desc">{m.desc}</div>
                    <div className="model-acc-bar">
                      <div className="model-acc-fill" style={{ width: `${m.acc}%` }} />
                    </div>
                  </div>
                  <div className="model-acc">{m.acc}%</div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel" style={{ marginTop: '1rem' }}>
            <div className="panel-header">
              <div className="panel-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                Dataset
              </div>
            </div>
            <div className="panel-body">
              <div style={{ fontSize: 12, color: 'var(--slate-500)', lineHeight: 1.8 }}>
                <div><strong style={{ color: 'var(--slate-700)' }}>CBIS-DDSM</strong></div>
                <div>Mass Images: <strong>1,592</strong></div>
                <div>Calc Images: <strong>1,511</strong></div>
                <div style={{ marginTop: 8 }}><strong style={{ color: 'var(--slate-700)' }}>Format</strong></div>
                <div>DICOM → PNG via TCIA</div>
              </div>
            </div>
          </div>
        </div>

        {/* ── CENTER: Upload + Form ── */}
        <div>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                Mammogram Upload
              </div>
              {imagePreview && (
                <button
                  onClick={() => { setImageFile(null); setImagePreview(null); setResult(null); }}
                  style={{ background: 'var(--slate-100)', border: 'none', borderRadius: 6, padding: '4px 10px', fontSize: 12, color: 'var(--slate-600)', cursor: 'pointer' }}
                >
                  Clear
                </button>
              )}
            </div>
            <div className="panel-body">
              {imagePreview ? (
                <div className="upload-preview">
                  <img src={imagePreview} alt="Mammogram preview" />
                  <div className="upload-preview-label">
                    Mammogram: {imageFile?.name || 'Selected image'}
                  </div>
                </div>
              ) : (
                <div
                  className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleImageDrop}
                  onClick={() => fileInputRef.current.click()}
                >
                  <div className="upload-icon">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#1a8c6a" strokeWidth="1.5">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                  </div>
                  <div className="upload-title">Drop mammogram here</div>
                  <div className="upload-sub">PNG, JPG, or DICOM · up to 50MB</div>
                  <input
                    ref={fileInputRef} type="file" accept="image/*,.dcm"
                    style={{ display: 'none' }}
                    onChange={(e) => e.target.files[0] && loadImage(e.target.files[0])}
                  />
                </div>
              )}

              <div className="divider" />

              {/* Clinical Metadata Form */}
              <div className="form-section">
                <div className="form-section-label">Clinical Metadata</div>
                <div className="form-grid">
                  <div className="form-group">
                    <label className="form-label">Mass Margin</label>
                    <select className="form-select" value={clinicalData.massMargin} onChange={e => setClinicalData(p => ({ ...p, massMargin: e.target.value }))}>
                      <option value="">Select…</option>
                      <option>Circumscribed</option>
                      <option>Spiculated</option>
                      <option>Ill-defined</option>
                      <option>Microlobulated</option>
                      <option>Obscured</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Mass Shape</label>
                    <select className="form-select" value={clinicalData.massShape} onChange={e => setClinicalData(p => ({ ...p, massShape: e.target.value }))}>
                      <option value="">Select…</option>
                      <option>Round</option>
                      <option>Oval</option>
                      <option>Irregular</option>
                      <option>Lobular</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Breast Density</label>
                    <select className="form-select" value={clinicalData.breastDensity} onChange={e => setClinicalData(p => ({ ...p, breastDensity: e.target.value }))}>
                      <option value="">Select…</option>
                      <option>ACR A – Fatty</option>
                      <option>ACR B – Scattered</option>
                      <option>ACR C – Heterogeneous</option>
                      <option>ACR D – Dense</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Calcification Type</label>
                    <select className="form-select" value={clinicalData.calcType} onChange={e => setClinicalData(p => ({ ...p, calcType: e.target.value }))}>
                      <option value="">Select…</option>
                      <option>Amorphous</option>
                      <option>Fine Pleomorphic</option>
                      <option>Coarse Heterogeneous</option>
                      <option>Fine Linear</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Calc Distribution</label>
                    <select className="form-select" value={clinicalData.calcDistribution} onChange={e => setClinicalData(p => ({ ...p, calcDistribution: e.target.value }))}>
                      <option value="">Select…</option>
                      <option>Clustered</option>
                      <option>Linear</option>
                      <option>Segmental</option>
                      <option>Regional</option>
                      <option>Diffuse</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Patient Age</label>
                    <input className="form-input" type="number" placeholder="e.g. 52" value={clinicalData.patientAge} onChange={e => setClinicalData(p => ({ ...p, patientAge: e.target.value }))} />
                  </div>
                </div>
              </div>

              <button className="diagnose-btn" onClick={handleDiagnose} disabled={!imageFile || loading}>
                {loading ? (
                  <><div className="spinner" /> Analyzing with AI…</>
                ) : (
                  <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>
                    Run Diagnosis
                  </>
                )}
              </button>

              {!imageFile && (
                <div className="alert info" style={{ marginTop: 10 }}>
                  Upload a mammogram image to enable diagnosis
                </div>
              )}

              {error && (
                <div className="alert" style={{ marginTop: 10, color: '#b91c1c', background: '#fee2e2', border: '1px solid #fecaca' }}>
                  {error}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── RIGHT: Results ── */}
        <div>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                Diagnostic Result
              </div>
              {result && (
                <span className={`result-badge ${result.type}`}>
                  {result.type === 'malignant' ? '⚠' : '✓'} {result.label}
                </span>
              )}
            </div>
            <div className="panel-body">
              {!result ? (
                <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--slate-400)' }}>
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" style={{ marginBottom: 12, opacity: 0.3 }}>
                    <circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/>
                  </svg>
                  <div style={{ fontSize: 13 }}>Results will appear here after diagnosis</div>
                </div>
              ) : (
                <>
                  <ConfidenceArc score={result.confidence} type={result.type} />
                  <div className="divider" />

                  <div style={{ marginBottom: 12 }}>
                    <div className="form-section-label" style={{ marginBottom: 10 }}>Feature Importance (SHAP)</div>
                    {result.features.map((f) => (
                      <div className="feature-row" key={f.name}>
                        <div className="feature-top">
                          <span className="feature-name">{f.name}</span>
                          <span className="feature-val">{(f.value * 100).toFixed(0)}%</span>
                        </div>
                        <div className="feature-bar">
                          <div className="feature-fill" style={{ width: `${f.value * 100 / 0.45 * 100}%`, background: FEATURE_COLORS[f.level] }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="divider" />
                  <div style={{ fontSize: 12, color: 'var(--slate-600)', lineHeight: 1.7, padding: '6px 0' }}>
                    <strong style={{ color: 'var(--slate-800)' }}>Clinical Notes:</strong><br />
                    {result.notes}
                  </div>

                  <div className="divider" />
                  <div className="form-section-label">XAI Indicators</div>
                  <div className="xai-tag-row">
                    <span className="xai-tag high">Spiculated Margin</span>
                    <span className="xai-tag high">High Density</span>
                    <span className="xai-tag medium">ACR C</span>
                    <span className="xai-tag low">Oval Shape</span>
                  </div>

                  <button
                    style={{ marginTop: 16, width: '100%', padding: '10px', border: '1px solid var(--teal-200)', background: 'var(--teal-50)', color: 'var(--teal-700)', borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}
                  >
                    View Full XAI Report →
                  </button>
                </>
              )}
            </div>
          </div>

          {result && (
            <div className="panel" style={{ marginTop: '1rem' }}>
              <div className="panel-header">
                <div className="panel-title">Treatment Recommendation</div>
              </div>
              <div className="panel-body">
                <div style={{ fontSize: 12, lineHeight: 1.8, color: 'var(--slate-600)' }}>
                  {result.type === 'malignant' ? (
                    <ul style={{ paddingLeft: 16 }}>
                      <li>Stereotactic core needle biopsy</li>
                      <li>Oncology referral within 2 weeks</li>
                      <li>MRI breast for staging</li>
                      <li>Multidisciplinary team review</li>
                    </ul>
                  ) : (
                    <ul style={{ paddingLeft: 16 }}>
                      <li>6-month follow-up mammography</li>
                      <li>Continue routine screening</li>
                      <li>Patient reassurance advised</li>
                    </ul>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
