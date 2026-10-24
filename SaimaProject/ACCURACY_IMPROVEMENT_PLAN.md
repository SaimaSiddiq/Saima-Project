# 🎯 ACCURACY IMPROVEMENT ANALYSIS & ACTION PLAN
## From 61.7% → Target 85%+ (27.3% improvement needed)

**Document Date:** May 6, 2026  
**Status:** Critical - Supervisor Feedback: Model Underperforming  
**Priority:** HIGH - Required for clinical deployment

---

## 1. ROOT CAUSE ANALYSIS: WHY 61.7% IS PROBLEMATIC

### 1.1 Performance Gap Assessment

| Context | Accuracy | Gap | Status |
|---------|----------|-----|--------|
| Random Chance (3-class) | 33.3% | - | Baseline |
| Current Model | 61.7% | +28.4% | ⚠️ Only 28.4% above random |
| Clinically Acceptable | ≥85% | **-23.3%** | ❌ UNACCEPTABLE |
| Published SOTA | 90-95% | -28.3% | ❌ FAR BELOW |

**Verdict:** Model shows only marginal improvement over random guessing on medical diagnosis. **NOT SUITABLE FOR CLINICAL DEPLOYMENT.**

---

### 1.2 Critical Root Causes Identified

#### **ROOT CAUSE #1: SEVERE CLASS IMBALANCE** ⚠️
```
Training Set Distribution:
  • MALIGNANT:                637 samples (48.3%)
  • BENIGN:                   577 samples (43.8%)
  • BENIGN_WITHOUT_CALLBACK:  104 samples (7.9%) ← SEVERELY UNDERREPRESENTED
  
Impact:
  ✗ Model biased toward majority classes
  ✗ Insufficient samples for minority class learning
  ✗ High risk of misclassifying minority class cases
  ✗ Cross-entropy loss doesn't account for imbalance
```

#### **ROOT CAUSE #2: GROSSLY INSUFFICIENT TRAINING** ⚠️
```
Current Training Configuration:
  • Epochs Trained: 2 (CRITICALLY LOW)
  • Expected Epochs: 50-200+
  • Learning Plateau Reached: Likely after epoch 10-15
  
Evidence:
  ✗ Loss curves not converging
  ✗ Model barely seeing each sample 2x
  ✗ Weights not optimized
  ✗ Early stopping may trigger prematurely
```

#### **ROOT CAUSE #3: SMALL DATASET SIZE** ⚠️
```
Dataset Statistics:
  • Total Samples: 1,318 (VERY SMALL for deep learning)
  • Training Samples: ~922 (70%)
  • Per-class Training: ~90-400 samples/class
  • CNN Best Practices: 10,000+ samples/class
  
Impact:
  ✗ Deep networks prone to overfitting
  ✗ DenseNet201 (169 layers) severely under-constrained
  ✗ Gradient estimates unreliable
  ✗ Cannot learn rich feature hierarchies
```

#### **ROOT CAUSE #4: MISSING DATA AUGMENTATION** ⚠️
```
Current Implementation: ❌ NO AUGMENTATION DETECTED
  
Missing Augmentations (especially critical for medical imaging):
  ✗ Rotation (±10-20°)
  ✗ Elastic deformations
  ✗ Gaussian noise
  ✗ Contrast enhancement (medical-specific)
  ✗ Gaussian blur
  
Potential gain: +5-15% accuracy
```

#### **ROOT CAUSE #5: WEAK MULTIMODAL FUSION** ⚠️
```
Current Approach: Cross-attention fusion on small data
  
Problems:
  ✗ Cross-attention requires >5K samples to train effectively
  ✗ Attention weights unstable with 104 minority samples
  ✗ Clinical features (48D) not engineered/optimized
  ✗ No validation that fusion is working
  ✗ Attention mechanism might be learning to ignore one modality
```

#### **ROOT CAUSE #6: SUBOPTIMAL HYPERPARAMETERS** ⚠️
```
Current Configuration:
  • Learning Rate: 0.001 (generic)
  • Batch Size: 16 (small dataset needs different strategy)
  • Weight Decay: 0.0001 (not accounting for imbalance)
  • Optimizer: Adam (no learning rate scheduling)
  
NOT optimized for:
  ✗ Small datasets
  ✗ Class imbalance
  ✗ Medical imaging domain
```

---

## 2. CONCRETE IMPROVEMENT PLAN

### PHASE 1: QUICK WINS (1-2 hours) - Expected +5-8% accuracy

#### Action 1.1: Enable Class Weights ⭐ CRITICAL
```python
# Calculate class weights inversely proportional to frequency
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# In loss function:
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))

# For class with 7.9% samples: weight ≈ 6.3x
# For classes with ~45%: weight ≈ 1.1x
```
**Expected Gain:** +3-5%  
**Why:** Prevents model from ignoring rare class

#### Action 1.2: Increase Training Epochs
```python
# Current: 2 epochs (UNACCEPTABLE)
# Recommended: 100+ epochs with early stopping

trainer = pl.Trainer(
    max_epochs=100,  # Changed from 2
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Stop if no improvement for 15 epochs
            mode='min',
            restore_best_weights=True
        ),
        ModelCheckpoint(save_best_only=True)
    ]
)
```
**Expected Gain:** +5-10%  
**Why:** Allows proper convergence; current 2 epochs = 0.2% data per epoch

#### Action 1.3: Add Data Augmentation for Medical Images
```python
from albumentations import Compose, Rotate, GaussNoise, GaussianBlur, Normalize
from albumentations.pytorch import ToTensorV2

train_transform = Compose([
    Rotate(limit=15, p=0.7),              # Medical images stable under ±15°
    GaussNoise(p=0.3),                     # Simulate imaging noise
    GaussianBlur(blur_limit=3, p=0.3),    # Smooth variations
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=None)
```
**Expected Gain:** +3-5%  
**Why:** Augmentation effectively multiplies dataset size

#### Action 1.4: Implement Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Option A: Cosine Annealing (recommended)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Option B: Reduce on Plateau (if loss plateaus)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# In training loop:
trainer.fit(model, train_loader, val_loader)
scheduler.step(val_loss)
```
**Expected Gain:** +2-3%  
**Why:** Fine-tunes convergence; prevents early local minima

---

### PHASE 2: ADVANCED TECHNIQUES (3-4 hours) - Expected +8-15% additional

#### Action 2.1: Handle Class Imbalance with SMOTE ⭐ HIGHLY RECOMMENDED
```python
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# SMOTE for minority class oversampling
smote = SMOTE(
    sampling_strategy={
        'BENIGN_WITHOUT_CALLBACK': 400,  # Upsample from 104→400
    },
    random_state=42,
    k_neighbors=3  # Lower k for small minority class
)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Result: More balanced distribution
# MALIGNANT: 637 → BENIGN: 577 → BENIGN_WITHOUT_CALLBACK: 400
```
**Expected Gain:** +5-10%  
**Why:** Solves fundamental class imbalance problem

#### Action 2.2: Feature Engineering for Clinical Data
```python
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

# Step 1: Feature normalization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(clinical_features)

# Step 2: Add polynomial features (interactions)
poly = PolynomialFeatures(degree=2, include_bias=False)
features_poly = poly.fit_transform(features_scaled)

# Step 3: Dimensionality reduction with PCA
pca = PCA(n_components=64, random_state=42)  # 48→64 optimal features
features_pca = pca.fit_transform(features_poly)

# Result: 48 → ~1500 (interactions) → 64 (principal components)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
```
**Expected Gain:** +2-4%  
**Why:** Captures feature interactions; reduces noise

#### Action 2.3: Test Multiple Backbones in Parallel
```python
backbones = {
    'DenseNet201': models.densenet201(pretrained=True),    # Current: 61.7%
    'EfficientNetB5': models.efficientnet_b5(pretrained=True),
    'ResNeXt101': models.resnext101_32x8d(pretrained=True),
    'InceptionV3': models.inception_v3(pretrained=True),
}

# Train each with consistent hyperparameters
for name, backbone in backbones.items():
    model = MultimodalClassifier(backbone=backbone, ...)
    results[name] = train_and_evaluate(model)
```
**Expected Gain:** +3-8%  
**Why:** Different architectures learn different representations

#### Action 2.4: Implement K-Fold Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold = X[train_idx]
    X_val_fold = X[val_idx]
    
    model = MultimodalClassifier(...)
    train_and_evaluate(model, X_train_fold, X_val_fold)
    
    fold_results.append(accuracy)

# More robust evaluation
mean_acc = np.mean(fold_results)
std_acc = np.std(fold_results)
```
**Expected Gain:** +0% direct, but +reliability  
**Why:** Robust evaluation; reduces test set variance

---

### PHASE 3: FINE-TUNING & OPTIMIZATION (2-3 days) - Expected +5-10% additional

#### Action 3.1: Hyperparameter Grid Search
```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.002],
    'batch_size': [8, 16, 32],
    'weight_decay': [0.00001, 0.0001, 0.001],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
}

best_params = None
best_accuracy = 0

for params in ParameterGrid(param_grid):
    model = train_model(config=params)
    accuracy = evaluate_model(model)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
```
**Expected Gain:** +3-5%  
**Why:** Finds optimal configuration

#### Action 3.2: Ensemble Methods
```python
# Train multiple models with different:
# - Random seeds
# - Data augmentation strategies
# - Architectures

models = [
    train_model(backbone='DenseNet201', seed=42),
    train_model(backbone='EfficientNetB5', seed=43),
    train_model(backbone='ResNeXt101', seed=44),
]

# Ensemble prediction (voting)
def ensemble_predict(x):
    predictions = [model(x) for model in models]
    return torch.mean(torch.stack(predictions), dim=0)
```
**Expected Gain:** +2-5%  
**Why:** Reduces overfitting; averages model biases

#### Action 3.3: Attention Visualization & Debugging
```python
import shap
from pytorch_grad_cam import GradCAM

# Understand model decisions
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_data)

# Visualize which image regions matter
cam = GradCAM(model=model, target_layers=[model.image_encoder[-1]])
gradcam_results = cam(test_images)

# Check multimodal fusion
attention_weights = model.get_attention_weights()
print(f"Image modality weight: {attention_weights[0]:.3f}")
print(f"Clinical modality weight: {attention_weights[1]:.3f}")
```
**Why:** Ensures fusion is working; builds clinical trust

---

## 3. IMPLEMENTATION ROADMAP

### Sprint 1: Quick Wins (2 hours)
- [ ] Modify loss function with class weights
- [ ] Change max_epochs from 2 → 100
- [ ] Add data augmentation
- [ ] Add learning rate scheduler
- **Estimated Result:** +6-8% accuracy

### Sprint 2: SMOTE + Feature Engineering (3 hours)
- [ ] Implement SMOTE oversampling
- [ ] Add polynomial features and PCA
- [ ] Retrain with balanced dataset
- **Estimated Result:** +8-12% cumulative

### Sprint 3: Multi-Backbone Testing (4 hours)
- [ ] Test EfficientNetB5, ResNeXt101, InceptionV3
- [ ] Compare results
- [ ] Select best performer
- **Estimated Result:** +3-8% cumulative

### Sprint 4: K-Fold Validation (2 hours)
- [ ] Implement 5-fold cross-validation
- [ ] Generate robust metrics
- [ ] Document variance
- **Estimated Result:** More reliable evaluation

### Sprint 5: Hyperparameter Optimization (8+ hours)
- [ ] Grid search over key parameters
- [ ] Use best params from Sprint 1-4
- [ ] Train ensemble models
- **Estimated Result:** +5-10% cumulative

### Sprint 6: Interpretability & Validation (4 hours)
- [ ] SHAP analysis for clinical validation
- [ ] Attention visualization
- [ ] Confusion matrix analysis
- [ ] Generate clinical validation report

---

## 4. EXPECTED OUTCOMES

### Current State
```
DenseNet201: 61.7%
├─ Problem: Only 11.7% above random
└─ Clinical Status: UNACCEPTABLE
```

### After Sprint 1 (Quick Wins)
```
Expected: 67-70%
├─ Class weights: +3-5%
├─ More epochs: +5-10%
├─ Augmentation: +3-5%
├─ LR scheduling: +2-3%
└─ Overlap/conflicts may reduce sum
```

### After Sprint 2 (SMOTE + Engineering)
```
Expected: 75-80%
├─ Previous: 67-70%
└─ + SMOTE + features: +8-12%
```

### After Sprint 3 (Multi-Backbone)
```
Expected: 78-85%
├─ Best of 3+ architectures
├─ EfficientNetB5 often beats DenseNet
└─ + ensemble: +2-5%
```

### After Sprints 4-6 (Optimization & Validation)
```
Target: 85-92%
├─ Hyperparameter tuning: +3-5%
├─ Ensemble methods: +2-5%
├─ Domain-specific fixes: +1-3%
└─ Clinical validation complete
```

---

## 5. PRIORITY IMPLEMENTATION FILE

**Create this file immediately:** `src/improved_train_multimodal.py`

Key changes:
1. Class weights in loss function
2. 100 epochs instead of 2
3. Data augmentation in transforms
4. Learning rate scheduler
5. SMOTE integration
6. K-fold evaluation

---

## 6. SUPERVISOR JUSTIFICATION TEMPLATE

**If current 61.7% remains:**

❌ **CANNOT say:** "This is good for early stage research"
- Only 11.7% above random is unacceptable

✅ **SHOULD say:** 
- "We identified root causes: class imbalance (7.9% minority), only 2 epochs training, missing augmentation"
- "Implementing recommended improvements (SMOTE, 100 epochs, multi-backbone) targets 85%+"
- "We have concrete 6-week plan to reach clinical threshold"
- "Training now with improvements; results by [DATE]"

---

## 7. QUICK START: Apply Phase 1 Now

```bash
# 1. Update training configuration
# 2. Modify loss function (add class weights)
# 3. Add augmentation
# 4. Change epochs to 100
# 5. Train immediately

python src/improved_train_multimodal.py --epochs 100 --use-class-weights --use-augmentation
```

**Expected time to see results:** 2-4 hours

---

## 8. CRITICAL SUCCESS FACTORS

| Factor | Current | Target | Why |
|--------|---------|--------|-----|
| Epochs | 2 | 100+ | Proper convergence |
| Class Balance | 7.9% imbalance | SMOTE balanced | Fair learning |
| Augmentation | None | Active | Effective data multiplying |
| Backbones tested | 1 (DenseNet) | 3-5 | Find best performer |
| Cross-validation | None | 5-fold | Robust evaluation |
| LR scheduling | No | Yes | Avoid local minima |

---

**Document Status:** READY FOR IMPLEMENTATION  
**Next Step:** Create improved training script (Phase 1)  
**Target Completion:** 2 weeks for all phases  
**Success Metric:** 85%+ test accuracy on 5-fold CV
