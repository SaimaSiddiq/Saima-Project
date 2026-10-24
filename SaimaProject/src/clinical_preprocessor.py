"""
Clinical Data Preprocessor
Converts raw CBIS-DDSM clinical metadata into one-hot encoded vectors suitable for neural network integration.

Objective O2: To implement a specialized preprocessing pipeline that transforms raw textual 
clinical labels from CSV datasets into high-dimensional one-hot encoded vectors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import OneHotEncoder
import json
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalPreprocessor:
    """
    Preprocesses CBIS-DDSM clinical metadata and converts categorical features to one-hot vectors.
    """
    
    def __init__(self, output_dir='data/processed/csv'):
        """
        Initialize the preprocessor.
        
        Args:
            output_dir (str): Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature mappings for each dataset type
        self.mass_categorical_features = [
            'breast_density', 'left or right breast', 'image view', 
            'mass shape', 'mass margins', 'pathology'
        ]
        
        self.calc_categorical_features = [
            'breast density', 'left or right breast', 'image view',
            'calc type', 'calc distribution', 'pathology'
        ]
        
        self.numerical_features = ['assessment', 'subtlety']
        
        self.encoders = {}
        self.feature_names = {}
        
    def load_data(self, filepath):
        """Load CSV file."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def identify_dataset_type(self, df):
        """Identify if dataset is mass or calcification based on columns."""
        if 'mass shape' in df.columns:
            return 'mass'
        elif 'calc type' in df.columns:
            return 'calcification'
        else:
            raise ValueError("Cannot determine dataset type from columns")
    
    def handle_missing_values(self, df, dataset_type):
        """Handle missing values in clinical features."""
        logger.info("Handling missing values...")
        
        categorical_features = self.mass_categorical_features if dataset_type == 'mass' else self.calc_categorical_features
        
        for col in categorical_features:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    logger.warning(f"  {col}: {missing_count} missing values - filling with 'UNKNOWN'")
                    df[col] = df[col].fillna('UNKNOWN')
        
        return df
    
    def encode_features(self, df, dataset_type, fit=False):
        """
        One-hot encode categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            dataset_type (str): 'mass' or 'calcification'
            fit (bool): Whether to fit the encoder (True for training data)
        
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        logger.info(f"Encoding features (fit={fit})...")
        
        categorical_features = self.mass_categorical_features if dataset_type == 'mass' else self.calc_categorical_features
        
        df_encoded = df.copy()
        encoded_arrays = []
        all_feature_names = []
        
        # Initialize encoder key
        encoder_key = dataset_type
        
        if fit:
            self.encoders[encoder_key] = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                dtype='uint8'
            )
            # Fit on available categorical features
            available_cats = [col for col in categorical_features if col in df.columns]
            self.encoders[encoder_key].fit(df[available_cats])
        
        # Transform
        available_cats = [col for col in categorical_features if col in df.columns]
        encoded_array = self.encoders[encoder_key].transform(df[available_cats])
        
        # Get feature names
        feature_names = self.encoders[encoder_key].get_feature_names_out(available_cats)
        self.feature_names[encoder_key] = feature_names
        
        # Create encoded dataframe
        df_encoded_features = pd.DataFrame(
            encoded_array,
            columns=feature_names,
            index=df.index
        )
        
        logger.info(f"  Created {encoded_array.shape[1]} one-hot encoded features")
        
        return df_encoded_features
    
    def process_dataset(self, input_file, dataset_type=None, fit=False):
        """
        Process a complete dataset.
        
        Args:
            input_file (str): Path to input CSV
            dataset_type (str): 'mass' or 'calcification' (auto-detected if None)
            fit (bool): Whether to fit encoders
        
        Returns:
            tuple: (df_encoded, df_original) with encoded and original data
        """
        df = self.load_data(input_file)
        
        # Detect dataset type if not provided
        if dataset_type is None:
            dataset_type = self.identify_dataset_type(df)
            logger.info(f"Detected dataset type: {dataset_type}")
        
        # Handle missing values
        df = self.handle_missing_values(df, dataset_type)
        
        # One-hot encode categorical features
        df_encoded = self.encode_features(df, dataset_type, fit=fit)
        
        # Add numerical features if present
        for col in self.numerical_features:
            if col in df.columns:
                df_encoded[col] = df[col].values
        
        # Add patient_id for reference
        if 'patient_id' in df.columns:
            df_encoded['patient_id'] = df['patient_id'].values
        
        # Add pathology label (target variable)
        if 'pathology' not in df_encoded.columns and 'pathology' in df.columns:
            df_encoded['pathology'] = df['pathology'].values
        
        logger.info(f"Final encoded shape: {df_encoded.shape}")
        logger.info(f"Columns: {list(df_encoded.columns)}")
        
        return df_encoded, df
    
    def save_processed_data(self, df_encoded, output_name):
        """Save processed data to CSV and metadata."""
        output_file = self.output_dir / f"{output_name}.csv"
        df_encoded.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        
        # Save feature names for reference
        metadata = {
            'shape': df_encoded.shape,
            'columns': list(df_encoded.columns),
            'feature_names': {k: v.tolist() for k, v in self.feature_names.items()}
        }
        
        metadata_file = self.output_dir / f"{output_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")
        
        return output_file
    
    def save_encoders(self, encoder_dir='data/processed'):
        """Save fitted encoders for future use."""
        encoder_dir = Path(encoder_dir)
        encoder_dir.mkdir(parents=True, exist_ok=True)
        
        for key, encoder in self.encoders.items():
            encoder_file = encoder_dir / f"encoder_{key}.pkl"
            with open(encoder_file, 'wb') as f:
                pickle.dump(encoder, f)
            logger.info(f"Saved encoder to {encoder_file}")
    
    def process_mass_dataset(self):
        """Process mass abnormality datasets."""
        logger.info("\n" + "="*80)
        logger.info("PROCESSING MASS ABNORMALITY DATASETS")
        logger.info("="*80)
        
        # Process training data
        df_train_encoded, _ = self.process_dataset(
            'mass_case_description_train_set.csv',
            dataset_type='mass',
            fit=True
        )
        self.save_processed_data(df_train_encoded, 'mass_train_encoded')
        
        # Process test data
        df_test_encoded, _ = self.process_dataset(
            'mass_case_description_test_set.csv',
            dataset_type='mass',
            fit=False
        )
        self.save_processed_data(df_test_encoded, 'mass_test_encoded')
        
        logger.info(f"Mass dataset processing complete!")
        logger.info(f"  Train: {df_train_encoded.shape}")
        logger.info(f"  Test: {df_test_encoded.shape}")
    
    def process_calcification_dataset(self):
        """Process calcification abnormality datasets."""
        logger.info("\n" + "="*80)
        logger.info("PROCESSING CALCIFICATION DATASETS")
        logger.info("="*80)
        
        # Process training data
        df_train_encoded, _ = self.process_dataset(
            'calc_case_description_train_set.csv',
            dataset_type='calcification',
            fit=True
        )
        self.save_processed_data(df_train_encoded, 'calc_train_encoded')
        
        # Process test data
        df_test_encoded, _ = self.process_dataset(
            'calc_case_description_test_set.csv',
            dataset_type='calcification',
            fit=False
        )
        self.save_processed_data(df_test_encoded, 'calc_test_encoded')
        
        logger.info(f"Calcification dataset processing complete!")
        logger.info(f"  Train: {df_train_encoded.shape}")
        logger.info(f"  Test: {df_test_encoded.shape}")
    
    def process_all(self):
        """Process all datasets."""
        self.process_mass_dataset()
        self.process_calcification_dataset()
        self.save_encoders()
        
        logger.info("\n" + "="*80)
        logger.info("ALL DATASETS PROCESSED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")


if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ClinicalPreprocessor(output_dir='data/processed/csv')
    
    # Process all datasets
    preprocessor.process_all()
    
    logger.info("\nPreprocessing complete! Check data/processed/csv/ for outputs.")
