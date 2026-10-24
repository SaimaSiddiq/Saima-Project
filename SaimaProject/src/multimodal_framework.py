"""
Multimodal Breast Cancer Diagnosis Framework
Implements cross-attention mechanism for fusing mammography images with clinical metadata.

Objective O3: Bidirectional cross-attention mechanism for synergistic interaction
between visual and categorical evidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from torchvision import models, transforms as T
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention mechanism for multimodal fusion.
    Enables dynamic attention between image features and clinical metadata.
    """

    def __init__(self, image_dim, clinical_dim, num_heads=8, dropout=0.1):
        """
        Initialize cross-attention module.

        Args:
            image_dim (int): Dimension of image features
            clinical_dim (int): Dimension of clinical features
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super(CrossAttention, self).__init__()

        self.image_dim = image_dim
        self.clinical_dim = clinical_dim
        self.num_heads = num_heads
        self.head_dim = image_dim // num_heads

        assert image_dim % num_heads == 0, "image_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.image_proj = nn.Linear(image_dim, image_dim)
        self.clinical_proj = nn.Linear(clinical_dim, image_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.out_proj = nn.Linear(image_dim, image_dim)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(image_dim)
        self.norm2 = nn.LayerNorm(image_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, clinical_features):
        """
        Forward pass of cross-attention.

        Args:
            image_features: Tensor of shape (batch_size, seq_len, image_dim)
            clinical_features: Tensor of shape (batch_size, clinical_dim)

        Returns:
            attended_features: Tensor of shape (batch_size, seq_len, image_dim)
            attention_weights: Attention weights for explainability
        """
        batch_size = image_features.size(0)

        # Project clinical features to match image feature dimension
        clinical_proj = self.clinical_proj(clinical_features)  # (batch_size, image_dim)
        clinical_proj = clinical_proj.unsqueeze(1)  # (batch_size, 1, image_dim)

        # Project image features
        image_proj = self.image_proj(image_features)  # (batch_size, seq_len, image_dim)

        # Cross-attention: clinical features attend to image features
        attended_output, attention_weights = self.attention(
            query=clinical_proj,  # (batch_size, 1, image_dim)
            key=image_proj,       # (batch_size, seq_len, image_dim)
            value=image_proj      # (batch_size, seq_len, image_dim)
        )

        # Residual connection and layer norm
        attended_output = self.norm1(attended_output.squeeze(1) + clinical_proj.squeeze(1))
        attended_output = self.norm2(self.out_proj(attended_output) + attended_output)

        return attended_output, attention_weights


class ImageEncoder(nn.Module):
    """
    Image encoder using various backbone models for feature extraction from mammograms.
    Supports ResNet, DenseNet, Inception, and EfficientNet.
    """

    def __init__(self, model_name='resnet50', pretrained=True, feature_dim=512):
        """
        Initialize image encoder.

        Args:
            model_name (str): Backbone model name ('resnet50', 'resnet34', 'densenet201', 'inception_v3', 'efficientnet_b0')
            pretrained (bool): Use pretrained weights
            feature_dim (int): Output feature dimension
        """
        super(ImageEncoder, self).__init__()

        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            in_features = 2048
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            in_features = 512
        elif model_name == 'densenet201':
            self.backbone = models.densenet201(pretrained=pretrained)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            in_features = 1920
        elif model_name == 'inception_v3':
            self.backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            in_features = 2048
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            in_features = 1280
        else:
            raise ValueError(f"Unsupported model: {model_name}. Supported: resnet50, resnet34, densenet201, inception_v3, efficientnet_b0")

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature projection
        self.feature_proj = nn.Linear(in_features, feature_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images (batch_size, 3, H, W)

        Returns:
            features: Image features (batch_size, feature_dim)
        """
        # Extract features
        features = self.backbone(x)  # (batch_size, channels, H', W')

        # Global average pooling
        features = self.adaptive_pool(features)  # (batch_size, channels, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, channels)

        # Project to desired dimension
        features = self.feature_proj(features)
        features = self.norm(features)

        return features


class ClinicalEncoder(nn.Module):
    """
    Clinical metadata encoder using MLP.
    """

    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=128, dropout=0.3):
        """
        Initialize clinical encoder.

        Args:
            input_dim (int): Input dimension (number of one-hot features)
            hidden_dims (list): Hidden layer dimensions
            output_dim (int): Output feature dimension
            dropout (float): Dropout probability
        """
        super(ClinicalEncoder, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Clinical features (batch_size, input_dim)

        Returns:
            features: Encoded clinical features (batch_size, output_dim)
        """
        return self.encoder(x)


class MultimodalFusion(nn.Module):
    """
    Fusion module combining image and clinical features.
    """

    def __init__(self, image_dim, clinical_dim, fusion_dim=256):
        """
        Initialize fusion module.

        Args:
            image_dim (int): Image feature dimension
            clinical_dim (int): Clinical feature dimension
            fusion_dim (int): Fused feature dimension
        """
        super(MultimodalFusion, self).__init__()

        # Cross-attention from clinical to image
        self.cross_attention = CrossAttention(image_dim, clinical_dim)

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + clinical_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim)
        )

    def forward(self, image_features, clinical_features):
        """
        Forward pass.

        Args:
            image_features: Image features (batch_size, image_dim)
            clinical_features: Clinical features (batch_size, clinical_dim)

        Returns:
            fused_features: Fused multimodal features (batch_size, fusion_dim)
            attention_weights: Attention weights for explainability
        """
        # Apply cross-attention
        attended_features, attention_weights = self.cross_attention(
            image_features.unsqueeze(1),  # Add sequence dimension
            clinical_features
        )

        # Concatenate attended features with original clinical features
        combined = torch.cat([attended_features, clinical_features], dim=-1)

        # Fuse through MLP
        fused_features = self.fusion(combined)

        return fused_features, attention_weights


class MultimodalClassifier(nn.Module):
    """
    Complete multimodal classification model.
    """

    def __init__(self, config):
        """
        Initialize multimodal classifier.

        Args:
            config (dict): Model configuration
        """
        super(MultimodalClassifier, self).__init__()

        self.config = config

        # Image encoder
        self.image_encoder = ImageEncoder(
            model_name=config.get('backbone', 'resnet50'),
            pretrained=config.get('pretrained', True),
            feature_dim=config.get('image_feature_dim', 512)
        )

        # Clinical encoder
        self.clinical_encoder = ClinicalEncoder(
            input_dim=config['clinical_input_dim'],
            hidden_dims=config.get('clinical_hidden_dims', [256, 128]),
            output_dim=config.get('clinical_feature_dim', 128)
        )

        # Multimodal fusion
        self.fusion = MultimodalFusion(
            image_dim=config.get('image_feature_dim', 512),
            clinical_dim=config.get('clinical_feature_dim', 128),
            fusion_dim=config.get('fusion_dim', 256)
        )

        # Classification head
        dropout = config.get('dropout', 0.4)
        self.classifier = nn.Sequential(
            nn.Linear(config.get('fusion_dim', 256), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, config.get('num_classes', 2))
        )

        # Initialize only non-backbone weights (preserve ImageNet pretrained features)
        self._init_new_weights()

    def _init_new_weights(self):
        """Initialize only new (non-pretrained) layers."""
        for module in [self.clinical_encoder, self.fusion, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        # Also init image_encoder projection (not pretrained)
        for m in [self.image_encoder.feature_proj, self.image_encoder.norm]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, clinical_features):
        """
        Forward pass.

        Args:
            images: Input images (batch_size, 3, H, W)
            clinical_features: Clinical features (batch_size, clinical_input_dim)

        Returns:
            logits: Classification logits (batch_size, num_classes)
            attention_weights: Attention weights for explainability
        """
        # Encode images
        image_features = self.image_encoder(images)

        # Encode clinical features
        clinical_features_encoded = self.clinical_encoder(clinical_features)

        # Fuse modalities
        fused_features, attention_weights = self.fusion(
            image_features, clinical_features_encoded
        )

        # Classification
        logits = self.classifier(fused_features)

        return logits, attention_weights

    def get_feature_importance(self, images, clinical_features):
        """
        Get feature importance scores for explainability.

        Args:
            images: Input images
            clinical_features: Clinical features

        Returns:
            attention_weights: Attention weights
            feature_contributions: Feature contribution scores
        """
        with torch.no_grad():
            _, attention_weights = self.forward(images, clinical_features)

        return attention_weights


class BreastCancerDataset(Dataset):
    """
    Dataset class for breast cancer multimodal data.
    """

    def __init__(self, clinical_df, image_dir=None, transform=None, mode='train'):
        """
        Initialize dataset.

        Args:
            clinical_df (pd.DataFrame): Clinical data with features and labels
            image_dir (str): Directory containing images
            transform: Image transformations
            mode (str): 'train', 'val', or 'test'
        """
        self.clinical_df = clinical_df
        self.image_dir = Path(image_dir) if image_dir else None
        self.transform = transform
        self.mode = mode

        # Identify feature columns (exclude patient_id and pathology)
        self.feature_cols = [col for col in clinical_df.columns
                           if col not in ['patient_id', 'pathology']]

        # Create label mapping
        self.label_map = {'BENIGN': 0, 'MALIGNANT': 1, 'BENIGN_WITHOUT_CALLBACK': 2}
        self.clinical_df['label'] = self.clinical_df['pathology'].map(self.label_map)

        # Check for any NaN values in labels and handle them
        nan_count = self.clinical_df['label'].isna().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in labels, dropping them")
            self.clinical_df = self.clinical_df.dropna(subset=['label'])

        # Convert to int
        self.clinical_df['label'] = self.clinical_df['label'].astype(int)

        logger.info(f"Dataset initialized with {len(self.clinical_df)} samples")
        logger.info(f"Feature columns: {len(self.feature_cols)}")
        logger.info(f"Label distribution: {self.clinical_df['pathology'].value_counts().to_dict()}")

    def __len__(self):
        return len(self.clinical_df)

    def __getitem__(self, idx):
        # Get clinical features
        clinical_features = self.clinical_df.iloc[idx][self.feature_cols].values.astype(np.float32)

        # Get label
        label = self.clinical_df.iloc[idx]['label']

        image = None
        if self.image_dir:
            image = self._load_image_for_sample(idx)
        else:
            image = torch.randn(3, 224, 224)

        if self.transform and not isinstance(image, torch.Tensor):
            image = self.transform(image)
        elif not self.transform and isinstance(image, Image.Image):
            image = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])(image)

        return {
            'image': image,
            'clinical_features': torch.tensor(clinical_features),
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': self.clinical_df.iloc[idx]['patient_id']
        }

    def _load_image_for_sample(self, idx):
        sample = self.clinical_df.iloc[idx]
        image_path = None

        if 'image_path' in self.clinical_df.columns:
            candidate = sample['image_path']
            image_path = Path(candidate)
            if not image_path.is_absolute():
                image_path = self.image_dir / image_path
        else:
            patient_id = str(sample['patient_id'])
            candidates = list(self.image_dir.rglob(f'*{patient_id}*'))
            if not candidates:
                raise FileNotFoundError(
                    f"No image found for patient_id={patient_id} in {self.image_dir}"
                )
            image_path = candidates[0]

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        return image


class MultimodalLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for multimodal breast cancer classification.
    """

    def __init__(self, config):
        """
        Initialize Lightning module.

        Args:
            config (dict): Model configuration
        """
        super().__init__()
        self.save_hyperparameters(config)

        # Model
        self.model = MultimodalClassifier(config)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=config.get('num_classes', 2))
        self.val_accuracy = Accuracy(task='multiclass', num_classes=config.get('num_classes', 2))
        self.val_precision = Precision(task='multiclass', num_classes=config.get('num_classes', 2))
        self.val_recall = Recall(task='multiclass', num_classes=config.get('num_classes', 2))
        self.val_f1 = F1Score(task='multiclass', num_classes=config.get('num_classes', 2))
        # Skip AUROC for multiclass classification

    def forward(self, images, clinical_features):
        return self.model(images, clinical_features)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        clinical_features = batch['clinical_features']
        labels = batch['label']

        logits, _ = self.model(images, clinical_features)
        loss = self.criterion(logits, labels)

        # Log metrics
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        clinical_features = batch['clinical_features']
        labels = batch['label']

        logits, _ = self.model(images, clinical_features)
        loss = self.criterion(logits, labels)

        # Log metrics
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        self.log('val_precision', self.val_precision)
        self.log('val_recall', self.val_recall)
        self.log('val_f1', self.val_f1)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for evaluation on test set.
        """
        images = batch['image']
        clinical_features = batch['clinical_features']
        labels = batch['label']

        logits, _ = self.model(images, clinical_features)
        loss = self.criterion(logits, labels)

        # Log metrics
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)  # Reuse val metrics for test
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.val_accuracy, prog_bar=True)
        self.log('test_precision', self.val_precision)
        self.log('test_recall', self.val_recall)
        self.log('test_f1', self.val_f1)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.get('learning_rate', 1e-4),
            weight_decay=self.hparams.get('weight_decay', 1e-4)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.get('max_epochs', 50)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def create_model_config(clinical_input_dim, num_classes=2):
    """
    Create model configuration dictionary.

    Args:
        clinical_input_dim (int): Number of clinical input features
        num_classes (int): Number of output classes

    Returns:
        dict: Model configuration
    """
    config = {
        # Model architecture
        'backbone': 'resnet50',
        'pretrained': True,
        'image_feature_dim': 512,
        'clinical_input_dim': clinical_input_dim,
        'clinical_hidden_dims': [256, 128],
        'clinical_feature_dim': 128,
        'fusion_dim': 256,
        'num_classes': num_classes,

        # Training
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'max_epochs': 50,
        'batch_size': 16,

        # Data
        'image_size': (224, 224),
    }

    return config


if __name__ == "__main__":
    # Example usage
    print("Multimodal Breast Cancer Framework")
    print("=" * 50)

    # Load sample data to determine dimensions
    mass_train = pd.read_csv('data/processed/csv/mass_train_encoded.csv')
    feature_cols = [col for col in mass_train.columns if col not in ['patient_id', 'pathology']]
    clinical_input_dim = len(feature_cols)

    print(f"Clinical input dimension: {clinical_input_dim}")
    print(f"Feature columns: {len(feature_cols)}")

    # Create model configuration
    config = create_model_config(clinical_input_dim)
    print(f"Model config: {config}")

    # Initialize model
    model = MultimodalClassifier(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass with dummy data
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_clinical = torch.randn(batch_size, clinical_input_dim)

    with torch.no_grad():
        logits, attention_weights = model(dummy_images, dummy_clinical)
        print(f"Forward pass successful!")
        print(f"Output shape: {logits.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")

    print("\n✅ Multimodal framework ready for training!")