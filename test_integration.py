"""
Integration Test Script
Tests all modules and their integrations for the BTCUSD Forex Prediction Model
"""

import sys
import os
from pathlib import Path
import logging
import yaml

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all modules can be imported successfully"""
    logger.info("Testing module imports...")
    
    try:
        # Test core modules
        from data_collection.data_collector import DataCollector
        logger.info("✅ DataCollector imported successfully")
        
        from data_preprocessing.data_preprocessor import DataPreprocessor
        logger.info("✅ DataPreprocessor imported successfully")
        
        from feature_engineering.technical_indicators import TechnicalIndicators
        logger.info("✅ TechnicalIndicators imported successfully")
        
        from feature_engineering.statistical_features import StatisticalFeatures
        logger.info("✅ StatisticalFeatures imported successfully")
        
        from feature_engineering.time_features import TimeFeatures
        logger.info("✅ TimeFeatures imported successfully")
        
        from feature_engineering.advanced_features import AdvancedFeatures
        logger.info("✅ AdvancedFeatures imported successfully")
        
        from feature_engineering.feature_selector import FeatureSelector
        logger.info("✅ FeatureSelector imported successfully")
        
        from target_definition.target_creator import TargetCreator
        logger.info("✅ TargetCreator imported successfully")
        
        from training.training_strategy import TrainingStrategy
        logger.info("✅ TrainingStrategy imported successfully")
        
        from training.advanced_training import AdvancedTrainingStrategy
        logger.info("✅ AdvancedTrainingStrategy imported successfully")
        
        from models.xgboost_model import XGBoostModel
        logger.info("✅ XGBoostModel imported successfully")
        
        from models.lstm_model import LSTMModel
        logger.info("✅ LSTMModel imported successfully")
        
        from models.ensemble_model import EnsembleModel
        logger.info("✅ EnsembleModel imported successfully")
        
        from tuning.hyperparameter_tuner import HyperparameterTuner
        logger.info("✅ HyperparameterTuner imported successfully")
        
        from deployment.deployer import Deployer
        logger.info("✅ Deployer imported successfully")
        
        from monitoring.monitor import Monitor
        logger.info("✅ Monitor imported successfully")
        
        from continuous_improvement.model_updater import ModelUpdater
        logger.info("✅ ModelUpdater imported successfully")
        
        from advanced_techniques.advanced_analyzer import AdvancedAnalyzer
        logger.info("✅ AdvancedAnalyzer imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test if configuration files are properly structured"""
    logger.info("Testing configuration...")
    
    try:
        # Test YAML config
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("✅ config.yaml loaded successfully")
            
            # Check key sections
            required_sections = ['data', 'features', 'models', 'training', 'evaluation', 'risk']
            for section in required_sections:
                if section in config:
                    logger.info(f"✅ Configuration section '{section}' found")
                else:
                    logger.warning(f"⚠️ Configuration section '{section}' missing")
            
            # Check advanced features config
            if 'advanced_features' in config.get('features', {}):
                logger.info("✅ Advanced features configuration found")
            else:
                logger.warning("⚠️ Advanced features configuration missing")
                
            # Check advanced training config
            if 'advanced_training' in config.get('training', {}):
                logger.info("✅ Advanced training configuration found")
            else:
                logger.warning("⚠️ Advanced training configuration missing")
                
        else:
            logger.error("❌ config.yaml not found")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_directory_structure():
    """Test if all required directories exist"""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        "src",
        "src/data_collection",
        "src/data_preprocessing", 
        "src/feature_engineering",
        "src/target_definition",
        "src/training",
        "src/models",
        "src/tuning",
        "src/deployment",
        "src/monitoring",
        "src/continuous_improvement",
        "src/advanced_techniques",
        "config",
        "data",
        "models"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            logger.info(f"✅ Directory '{dir_path}' exists")
        else:
            logger.warning(f"⚠️ Directory '{dir_path}' missing")
            all_exist = False
    
    return all_exist

def test_class_instantiation():
    """Test if classes can be instantiated with config"""
    logger.info("Testing class instantiation...")
    
    try:
        # Load config
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Test key classes
        from data_collection.data_collector import DataCollector
        data_collector = DataCollector(config)
        logger.info("✅ DataCollector instantiated successfully")
        
        from feature_engineering.advanced_features import AdvancedFeatures
        advanced_features = AdvancedFeatures(config)
        logger.info("✅ AdvancedFeatures instantiated successfully")
        
        from training.advanced_training import AdvancedTrainingStrategy
        advanced_training = AdvancedTrainingStrategy(config)
        logger.info("✅ AdvancedTrainingStrategy instantiated successfully")
        
        from advanced_techniques.advanced_analyzer import AdvancedAnalyzer
        advanced_analyzer = AdvancedAnalyzer(config)
        logger.info("✅ AdvancedAnalyzer instantiated successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Class instantiation failed: {e}")
        return False

def main():
    """Run all integration tests"""
    logger.info("=" * 60)
    logger.info("BTCUSD FOREX PREDICTION MODEL - INTEGRATION TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Directory Structure", test_directory_structure),
        ("Class Instantiation", test_class_instantiation)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION TEST RESULTS")
    logger.info("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED! System is ready for deployment.")
    else:
        logger.warning("⚠️ Some tests failed. Please review and fix issues before deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
