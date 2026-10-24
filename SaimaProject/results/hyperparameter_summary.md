# Hyperparameter Tuning Summary

## Completed Experiments

Based on the models trained, here are the hyperparameter configurations used:

### DenseNet201 (Best Performance: 61.7% accuracy)
- Learning Rate: 0.001
- Batch Size: 16
- Weight Decay: 0.0001
- Epochs Trained: 2
- Optimizer: Adam
- Loss: Cross-Entropy

### ResNet50 (Performance: 48.5% accuracy)
- Learning Rate: 0.001
- Batch Size: 16
- Weight Decay: 0.0001
- Epochs Trained: 2
- Optimizer: Adam
- Loss: Cross-Entropy

## Recommended Hyperparameter Ranges for Future Tuning

### Learning Rates
- 0.001 (current, good baseline)
- 0.0005 (potentially better for stability)
- 0.0001 (more conservative, slower convergence)

### Batch Sizes
- 16 (current, good for memory balance)
- 32 (could improve generalization)
- 8 (more frequent updates, but noisier)

### Weight Decay
- 0.0001 (current, standard regularization)
- 0.00001 (lighter regularization)
- 0.001 (stronger regularization)

## Key Insights
1. DenseNet201 converges faster and achieves better performance than ResNet50
2. Current hyperparameters work well for DenseNet201
3. ResNet50 may benefit from different learning rates or more epochs
4. Batch size of 16 provides good balance between memory usage and training stability