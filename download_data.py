"""
BTCUSD Data Download Script
Downloads historical data from MetaTrader 5 and saves it for model training
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
        logging.FileHandler('data_download.log'),
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
    """Main data download function"""
    
    logger.info("=" * 60)
    logger.info("BTCUSD FOREX DATA DOWNLOAD STARTED")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        return False
    
    try:
        # Import data collector
        from data_collection.data_collector import DataCollector
        
        # Initialize data collector
        logger.info("Initializing DataCollector...")
        data_collector = DataCollector(config)
        
        # Get data collector info
        info = data_collector.get_data_info()
        logger.info(f"Data Collector Info: {info}")
        
        # Download historical data (3 years as configured)
        years = config.get('data', {}).get('historical_years', 3)
        logger.info(f"Starting historical data download for {years} years...")
        
        historical_data = data_collector.collect_historical_data(years=years)
        
        if historical_data is not None:
            logger.info("=" * 60)
            logger.info("DATA DOWNLOAD SUCCESSFUL!")
            logger.info("=" * 60)
            logger.info(f"Records downloaded: {len(historical_data)}")
            logger.info(f"Date range: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")
            logger.info(f"Price range: ${historical_data['close'].min():.2f} - ${historical_data['close'].max():.2f}")
            logger.info(f"Average volume: {historical_data['volume'].mean():.0f}")
            
            # Show sample data
            logger.info("\nSample data (first 5 rows):")
            logger.info(historical_data.head().to_string())
            
            logger.info("\nData saved to: data/raw/BTCUSD_5min_latest.csv")
            logger.info("You can now proceed with data preprocessing and model training!")
            
            return True
        else:
            logger.error("Failed to download historical data")
            logger.error("Please check:")
            logger.error("1. MetaTrader 5 is running and logged in")
            logger.error("2. BTCUSD symbol is available in your broker")
            logger.error("3. Internet connection is stable")
            return False
            
    except Exception as e:
        logger.error(f"Error in data download: {e}")
        logger.error("Make sure MetaTrader 5 is running and you're logged in")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Data download completed successfully!")
        print("Next steps:")
        print("1. Run: python preprocess_data.py")
        print("2. Run: python src/main.py (full pipeline)")
    else:
        print("\n❌ Data download failed. Please check the logs above.")
    
    input("\nPress Enter to exit...")
