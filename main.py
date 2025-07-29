#!/usr/bin/env python3
"""
BTCUSD Forex Prediction Model - Main Execution Pipeline
12-Week Implementation Plan Executor
"""

import argparse
import logging
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.logger import setup_logger
from utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="BTCUSD Forex Prediction Model Pipeline")
    parser.add_argument("--phase", type=int, choices=range(1, 12), 
                       help="Execute specific phase (1-11)")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Configuration file path")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(level=log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info("=" * 60)
    logger.info("BTCUSD Forex Prediction Model Pipeline")
    logger.info("=" * 60)
    
    if args.phase:
        execute_phase(args.phase, config, logger)
    else:
        logger.info("Available phases:")
        print_phase_menu()

def execute_phase(phase_num, config, logger):
    """Execute specific phase of the 12-week plan"""
    
    phases = {
        1: phase_1_data_collection,
        2: phase_2_feature_engineering,
        3: phase_3_target_definition,
        4: phase_4_model_development,
        5: phase_5_training_strategy,
        6: phase_6_evaluation,
        7: phase_7_hyperparameter_tuning,
        8: phase_8_backtesting,
        9: phase_9_validation,
        10: phase_10_deployment,
        11: phase_11_continuous_improvement
    }
    
    if phase_num in phases:
        logger.info(f"Executing Phase {phase_num}")
        phases[phase_num](config, logger)
    else:
        logger.error(f"Invalid phase number: {phase_num}")

def phase_1_data_collection(config, logger):
    """Phase 1: Data Collection & Preprocessing (Weeks 1-2)"""
    logger.info("Phase 1: Data Collection & Preprocessing")
    
    try:
        from data_collection.mt5_collector import MT5Collector
        from data_collection.yfinance_collector import YFinanceCollector
        from data_collection.alpha_vantage_collector import AlphaVantageCollector
        from data_collection.binance_collector import BinanceCollector
        from preprocessing.data_cleaner import DataCleaner
        
        # Initialize collectors based on config
        collectors = []
        
        if config['data']['sources']['primary'] == 'mt5':
            collectors.append(MT5Collector(config))
        
        for source in config['data']['sources']['alternatives']:
            if source == 'yfinance':
                collectors.append(YFinanceCollector(config))
            elif source == 'alpha_vantage':
                collectors.append(AlphaVantageCollector(config))
            elif source == 'binance':
                collectors.append(BinanceCollector(config))
        
        # Collect data from all sources
        for collector in collectors:
            logger.info(f"Collecting data from {collector.__class__.__name__}")
            collector.collect_historical_data()
        
        # Clean and preprocess data
        cleaner = DataCleaner(config)
        cleaner.process_all_sources()
        
        logger.info("Phase 1 completed successfully!")
        
    except ImportError as e:
        logger.error(f"Missing dependencies for Phase 1: {e}")
        logger.info("Run: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Error in Phase 1: {e}")

def phase_2_feature_engineering(config, logger):
    """Phase 2: Feature Engineering (Weeks 2-3)"""
    logger.info("Phase 2: Feature Engineering")
    
    try:
        from feature_engineering.technical_indicators import TechnicalIndicators
        from feature_engineering.statistical_features import StatisticalFeatures
        from feature_engineering.time_features import TimeFeatures
        
        # Create feature engineers
        tech_indicators = TechnicalIndicators(config)
        stat_features = StatisticalFeatures(config)
        time_features = TimeFeatures(config)
        
        # Generate all features
        logger.info("Generating technical indicators...")
        tech_indicators.generate_all_indicators()
        
        logger.info("Generating statistical features...")
        stat_features.generate_all_features()
        
        logger.info("Generating time-based features...")
        time_features.generate_all_features()
        
        logger.info("Phase 2 completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Phase 2: {e}")

def print_phase_menu():
    """Print available phases"""
    phases = [
        "Phase 1: Data Collection & Preprocessing (Weeks 1-2)",
        "Phase 2: Feature Engineering (Weeks 2-3)",
        "Phase 3: Target Variable Definition (Week 3)",
        "Phase 4: Model Development (Weeks 4-6)",
        "Phase 5: Training Strategy (Week 6)",
        "Phase 6: Model Evaluation (Week 7)",
        "Phase 7: Hyperparameter Tuning (Week 8)",
        "Phase 8: Backtesting Framework (Weeks 9-10)",
        "Phase 9: Model Validation & Robustness (Week 11)",
        "Phase 10: Production Deployment (Week 12)",
        "Phase 11: Continuous Improvement (Ongoing)"
    ]
    
    for i, phase in enumerate(phases, 1):
        print(f"{i:2d}. {phase}")
    
    print("\nUsage: python main.py --phase <number>")
    print("Example: python main.py --phase 1")

# Placeholder functions for remaining phases
def phase_3_target_definition(config, logger):
    logger.info("Phase 3: Target Variable Definition - Implementation needed")

def phase_4_model_development(config, logger):
    logger.info("Phase 4: Model Development - Implementation needed")

def phase_5_training_strategy(config, logger):
    logger.info("Phase 5: Training Strategy - Implementation needed")

def phase_6_evaluation(config, logger):
    logger.info("Phase 6: Model Evaluation - Implementation needed")

def phase_7_hyperparameter_tuning(config, logger):
    logger.info("Phase 7: Hyperparameter Tuning - Implementation needed")

def phase_8_backtesting(config, logger):
    logger.info("Phase 8: Backtesting Framework - Implementation needed")

def phase_9_validation(config, logger):
    logger.info("Phase 9: Model Validation & Robustness - Implementation needed")

def phase_10_deployment(config, logger):
    logger.info("Phase 10: Production Deployment - Implementation needed")

def phase_11_continuous_improvement(config, logger):
    logger.info("Phase 11: Continuous Improvement - Implementation needed")

if __name__ == "__main__":
    main()
