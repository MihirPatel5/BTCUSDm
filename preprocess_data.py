"""
Data Preprocessing Script
Preprocesses downloaded BTCUSD data for model training
"""

import sys
import os
from pathlib import Path
import logging
import yaml

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration"""
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None

def main():
    """Main data preprocessing function"""
    
    logger.info("=" * 60)
    logger.info("BTCUSD DATA PREPROCESSING STARTED")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        return False
    
    try:
        # Import data preprocessor
        from data_preprocessing.data_preprocessor import DataPreprocessor
        
        # Initialize preprocessor
        logger.info("Initializing DataPreprocessor...")
        preprocessor = DataPreprocessor(config)
        
        # Preprocess data
        logger.info("Starting data preprocessing...")
        processed_data = preprocessor.preprocess_data()
        
        if processed_data is not None:
            logger.info("=" * 60)
            logger.info("DATA PREPROCESSING SUCCESSFUL!")
            logger.info("=" * 60)
            
            # Get preprocessing stats
            stats = preprocessor.get_preprocessing_stats(processed_data)
            logger.info(f"Processed records: {stats.get('total_records', 'N/A')}")
            logger.info(f"Date range: {stats.get('date_range', {}).get('duration_days', 'N/A')} days")
            logger.info(f"Price statistics: {stats.get('price_statistics', {})}")
            
            logger.info("\nData saved to: data/processed/BTCUSD_5min_preprocessed.csv")
            logger.info("You can now proceed with feature engineering and model training!")
            
            return True
        else:
            logger.error("Failed to preprocess data")
            logger.error("Please check if raw data exists in data/raw/ directory")
            return False
            
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Data preprocessing completed successfully!")
        print("Next steps:")
        print("1. Run: python src/main.py (full pipeline)")
        print("2. Or run individual steps for feature engineering")
    else:
        print("\n❌ Data preprocessing failed. Please check the logs above.")
    
    input("\nPress Enter to exit...")
