# 🚀 QUICK START: Run Improved Training

**Status:** Ready to execute  
**Expected Runtime:** 2-6 hours  
**Expected Accuracy Gain:** +6-8%  

---

## What Changed (Phase 1 Improvements)

| Feature | Old | New | Gain |
|---------|-----|-----|------|
| **Epochs** | 2 | 100 | +5-10% |
| **Data Augmentation** | ❌ None | ✅ Active | +3-5% |
| **Class Weights** | ❌ None | ✅ Balanced | +3-5% |
| **LR Scheduler** | ❌ Fixed | ✅ Cosine Annealing | +2-3% |
| **Early Stopping** | 3 patience | 15 patience | More training |

---

## 📋 Pre-Requisites

✅ Virtual environment activated  
✅ All dependencies installed (verified earlier)  
✅ Data in `data/processed/csv/` (check below)  
✅ Images in `data/processed/images/` (if available)  

### Verify Data Exists:
```powershell
Test-Path "c:\Users\hp\Desktop\RoomaProject\data\processed\csv\mass_train_encoded.csv"
Test-Path "c:\Users\hp\Desktop\RoomaProject\data\processed\images"
```

---

## 🎯 Option 1: Run with Default Settings (Recommended)

```powershell
cd c:\Users\hp\Desktop\RoomaProject

# Make sure venv is activated
.\.venv\Scripts\Activate.ps1

# Run improved training
python src/improved_train_multimodal.py
```

**Expected Output:**
```
================================================================
IMPROVED MULTIMODAL TRAINING - PHASE 1
================================================================
Start Time: 2026-05-06 14:30:00

Configuration:
  max_epochs                    : 100
  batch_size                    : 16
  learning_rate                 : 0.001
  use_class_weights             : True
  use_data_augmentation         : True
  use_lr_scheduler              : True
  early_stopping_patience       : 15
  ...

CLASS WEIGHTS (for loss function)
====================================================================
Class 0 (BENIGN): weight = 1.2345
Class 1 (BENIGN_WITHOUT_CALLBACK): weight = 6.3456 ← Minority gets 6.3x
Class 2 (MALIGNANT): weight = 1.1234

Starting training...
Epoch 1/100: [████████░░░░░░░░░░░░] - loss: 0.892, val_acc: 0.643
Epoch 2/100: [████████░░░░░░░░░░░░] - loss: 0.785, val_acc: 0.678
...
Epoch 50/100: [████████░░░░░░░░░░░░] - loss: 0.520, val_acc: 0.721
...
```

---

## 🎯 Option 2: Customize Parameters

### More Epochs (for better convergence):
```powershell
python src/improved_train_multimodal.py --epochs 150
```

### Smaller Batch Size (for smaller GPU memory):
```powershell
python src/improved_train_multimodal.py --batch-size 8
```

### Disable Augmentation (if testing):
```powershell
python src/improved_train_multimodal.py --no-augmentation
```

### All Options:
```powershell
python src/improved_train_multimodal.py `
  --epochs 100 `
  --batch-size 16 `
  --learning-rate 0.001 `
  --data-path data/processed/csv/mass_train_encoded.csv `
  --image-dir data/processed/images
```

---

## 📊 What to Monitor

### Console Output to Watch:

1. **Class Weights Loading:**
   ```
   CLASS WEIGHTS (for loss function)
   Class 0 (BENIGN): weight = 1.2345
   Class 1 (BENIGN_WITHOUT_CALLBACK): weight = 6.3456 ← Should be high
   Class 2 (MALIGNANT): weight = 1.1234
   ```
   ✅ If minority class weight >> 1: Good! Class imbalance being addressed

2. **Epoch Progress:**
   ```
   Epoch 1/100: loss: 0.892, val_loss: 0.845, val_acc: 0.643
   Epoch 2/100: loss: 0.785, val_loss: 0.798, val_acc: 0.678
   ```
   ✅ Look for: Decreasing loss, increasing val_acc

3. **Learning Rate Decay:**
   ```
   LR: 0.001000 (Epoch 1)
   LR: 0.000997 (Epoch 2)
   ...
   LR: 0.000001 (Epoch 100)
   ```
   ✅ Should gradually decrease to improve fine-tuning

4. **Early Stopping (if triggered):**
   ```
   EarlyStopping: val_loss did not improve for 15 epochs. Stopping.
   ```
   ✅ Normal if validation improves then plateaus

---

## 📈 Expected Results

### Timing:
- **Epoch time:** ~2-5 minutes per epoch (CPU)
- **Total time:** 2-8 hours for 100 epochs
- **Total time:** 1-2 hours for 50 epochs

### Accuracy Progression (Approximate):
```
Epoch 5:    ~64% (starting to improve)
Epoch 20:   ~68% (noticeable improvement)
Epoch 50:   ~72% (good progress)
Epoch 100:  ~75% (Phase 1 target)
```

### Final Results Location:
```
checkpoints/mass_densenet201_improved/
├── epoch=05-val_loss=0.750-val_acc=0.640.ckpt
├── epoch=20-val_loss=0.620-val_acc=0.680.ckpt
├── epoch=50-val_loss=0.480-val_acc=0.720.ckpt
└── epoch=99-val_loss=0.420-val_acc=0.750.ckpt ← Best model

results/logs/mass_multimodal_improved/
└── version_0/
    └── events.out.tfevents...  ← TensorBoard logs
```

---

## 📊 View TensorBoard (Real-time Monitoring)

While training is running:

```powershell
# In another terminal
cd c:\Users\hp\Desktop\RoomaProject
.\.venv\Scripts\Activate.ps1
tensorboard --logdir results/logs/mass_multimodal_improved
```

Then open browser: `http://localhost:6006`

---

## ✅ Success Criteria

### Training Succeeds If:
- [ ] No errors in first 100 lines of output
- [ ] Class weights show: minority class weight > 1
- [ ] Val accuracy increases from Epoch 1 to Epoch 20
- [ ] Model checkpoint files created in `checkpoints/`
- [ ] Training completes or early stops after patience exceeded

### Expected Accuracy:
- ✅ **Success:** Final accuracy ≥ 65% (from current 61.7%)
- ✅ **Strong:** Final accuracy ≥ 70%
- 🎯 **Phase 1 Target:** Final accuracy ≥ 68%

---

## 🔧 If Training Fails

### Common Issues:

**Issue:** `FileNotFoundError: data/processed/csv/mass_train_encoded.csv`  
**Fix:** Check data exists: `ls data/processed/csv/`

**Issue:** `CUDA out of memory`  
**Fix:** Reduce batch size: `--batch-size 8`

**Issue:** `ImportError: cannot import name 'MultimodalClassifier'`  
**Fix:** Make sure you're in project root: `cd c:\Users\hp\Desktop\RoomaProject`

**Issue:** Training too slow  
**Fix:** Use GPU if available (automatically detected) or reduce epochs

---

## 📝 Next Steps After Training

### 1. Check Results:
```powershell
# View best checkpoint
ls checkpoints/mass_densenet201_improved -Recurse
```

### 2. Copy Best Model:
```powershell
# Copy best model to deployment location
Copy-Item "checkpoints/mass_densenet201_improved/epoch=99-*.ckpt" `
          "checkpoints/best_improved_model.ckpt"
```

### 3. Phase 2 (if accuracy < 75%):
- Implement SMOTE oversampling
- Add feature engineering
- See: `ACCURACY_IMPROVEMENT_PLAN.md` → Phase 2

### 4. Phase 3 (if accuracy 75-85%):
- Test multiple backbones
- See: `ACCURACY_IMPROVEMENT_PLAN.md` → Phase 3

---

## 💾 Save Training Config

After successful training, save the config for reproducibility:

```powershell
# This is automatically saved, but you can verify:
Get-Content "results/logs/mass_multimodal_improved/version_0/hparams.yaml"
```

---

## 🎓 Expected Improvements Breakdown

| Improvement | Expected Gain | How It Works |
|------------|--------------|-------------|
| **100 epochs vs 2** | +5-10% | Allows proper model convergence |
| **Data Augmentation** | +3-5% | Simulates imaging variations; effective data multiplying |
| **Class Weights** | +3-5% | Minority class (7.9%) gets 6.3x importance; fair learning |
| **LR Scheduling** | +2-3% | Gradual LR decay prevents local minima |
| **Early Stopping (patience=15)** | +1-2% | Allows model to fully converge |
| **Total Expected** | **+6-8%** | Current 61.7% → **Estimated 67-70%** |

---

## 📞 After Training Support

If you have questions about:
- **Results:** Check `results/comprehensive_report/`
- **Checkpoints:** See `checkpoints/mass_densenet201_improved/`
- **Logs:** View with TensorBoard
- **Next steps:** See `ACCURACY_IMPROVEMENT_PLAN.md`

---

## 🚀 START NOW

Copy this command and paste in PowerShell:

```powershell
cd c:\Users\hp\Desktop\RoomaProject; .\.venv\Scripts\Activate.ps1; python src/improved_train_multimodal.py
```

**Estimated Time to First Results:** 5-10 minutes  
**Estimated Time to Final Results:** 2-8 hours  

Good luck! 🎯
