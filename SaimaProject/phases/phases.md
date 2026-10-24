Phase 1 — Data Preparation
├── Link images to CSV rows via patient_id + image_view (CC/MLO)
├── Train/test split already done (separate CSVs)
└── Dataset class: (image_path, clinical_vector, label)

Phase 2 — Image Branch
├── Load PNG → resize 224×224 → normalize
├── Backbone: EfficientNetB3 (better than DenseNet201 for med imaging, lighter)
│   → frozen pretrained weights, fine-tune last N layers
└── Output: 512-dim mammograph embedding

Phase 3 — Clinical Branch  
├── Input: one-hot encoded vector (already done in CSVs)
│   Mass: ~49 features | Calc: ~65 features
├── ANN: Linear(in→256) → ReLU → Dropout(0.3) → Linear(256→128) → ReLU
└── Output: 128-dim clinical embedding

Phase 4 — Cross-Attention Fusion
├── Image embedding queries Clinical embedding (image→clinical attention)
├── Clinical embedding queries Image embedding (clinical→image attention)
└── Concat fused vectors → Linear → Sigmoid → binary (MALIGNANT/BENIGN)

Phase 5 — Training
├── Loss: BCEWithLogitsLoss (handle class imbalance via pos_weight)
├── Optimizer: AdamW, lr=1e-4, weight_decay=1e-5
├── Scheduler: CosineAnnealingLR
├── Modality dropout: randomly zero out clinical vector (p=0.2) during training
└── Metrics: Accuracy, Precision, Recall, F1, AUC-ROC

Phase 6 — XAI
├── Grad-CAM on image backbone → heatmap overlay
└── SHAP on clinical branch → feature importance bar chart

Phase 7 — Flask Web App
├── Upload mammogram PNG
├── Fill clinical form (dropdowns → one-hot encode on server)
├── Run inference → return: diagnosis + confidence + Grad-CAM + SHAP plot
└── Single route /predict
