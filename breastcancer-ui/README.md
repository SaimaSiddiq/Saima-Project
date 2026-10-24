# MammoXplain – Setup Guide

Explainable Cross-Attention Multimodal Framework for Breast Cancer Diagnosis  
M.Phil Research Project | Frontend: React · Backend: Node.js + Flask

---

## Project Structure

```
breastcancer-ui/
├── src/
│   ├── App.js                  # Main shell with navigation
│   ├── App.css                 # Light-theme design system
│   ├── index.js                # React entry point
│   └── pages/
│       ├── DiagnosisPage.js    # Upload + clinical form + results
│       ├── ModelComparisonPage.js  # Benchmarks with charts
│       └── XAIReportPage.js    # SHAP + Grad-CAM + LIME
├── public/index.html
├── package.json                # React dependencies
├── server.js                   # Node.js API gateway
├── server-package.json         # Node server dependencies
└── flask_api.py                # Flask ML backend (attach your models here)
```

---

## Step 1 – Start the React Frontend

```bash
cd breastcancer-ui
npm install
npm start
# Opens at http://localhost:3000
```

---

## Step 2 – Start the Node.js Gateway

```bash
# From the project root install the gateway deps and start the server
cd breastcancer-ui
npm install --save express cors http-proxy-middleware multer nodemon
node server.js
# Runs at http://localhost:3000
```

### Gateway route fix

- If you hit a startup error from `path-to-regexp` (PathError), it's caused by the previous catch-all route used for client-side routing. That route was replaced with a regex that excludes API paths to avoid conflicts. See [breastcancer-ui/server.js](breastcancer-ui/server.js#L1-L400).

---

## Step 3 – Start the Flask Backend

```bash
pip install flask flask-cors numpy pillow
python flask_api.py
# Runs at http://localhost:5000
```

---

## Step 4 – Attach Your Trained Models

Open `flask_api.py` and follow the commented sections:

1. **Set model paths** in `MODEL_PATHS`
2. **Uncomment** the preprocessing and inference blocks in `/api/diagnose`
3. **Add SHAP** explainer in `/api/shap/<case_id>`
4. **Add Grad-CAM** generation in `/api/gradcam/<case_id>`

The React frontend will call `http://localhost:4000/api/diagnose` which proxies to Flask automatically.

---

## API Endpoints (Flask)

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Connection status |
| `/api/diagnose` | POST | Main diagnosis (image + clinical JSON) |
| `/api/compare` | POST | Multi-model comparison |
| `/api/gradcam/<case_id>` | GET | Grad-CAM heatmap PNG |
| `/api/shap/<case_id>` | GET | SHAP attribution data |

---

## Frontend Pages

| Page | Description |
|---|---|
| **Diagnosis** | Upload mammogram, enter clinical metadata, get AI prediction |
| **Model Comparison** | Table + bar chart + radar comparing all 5 models |
| **XAI Report** | SHAP bar chart, Grad-CAM heatmap, LIME analysis, export |

---

## Technologies

- **React 18** + Recharts (charts) + Lucide Icons
- **Node.js** + Express (API gateway / proxy)
- **Flask** (ML inference API — attach your PyTorch/TF models)
- **CBIS-DDSM** dataset (1,592 mass + 1,511 calc images)
- **XAI**: SHAP · Grad-CAM · LIME
