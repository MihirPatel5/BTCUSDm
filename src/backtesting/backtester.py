"""
Backtesting Framework
Implements historical replay backtesting for BTCUSD prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns


class Backtester:
    """Historical replay backtester for trading strategies"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed")
        self.backtest_dir = Path("data/backtest")
        self.backtest_dir.mkdir(parents=True, exist_ok=True)
        
    def run_backtest(self, predictions_file: str, initial_capital: float = 10000.0) -> dict:
        """Run historical backtest using model predictions"""
        
        try:
            self.logger.info(f"Running backtest with initial capital: ${initial_capital:,.2f}")
            
            # Load predictions
            pred_path = Path(predictions_file)
            if not pred_path.exists():
                self.logger.error(f"Predictions file not found: {pred_path}")
                return {}
            
            predictions = pd.read_csv(pred_path)
            predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])
            
            # Load original price data
            price_file = self.data_dir / "BTCUSD_5min_processed.csv"
            if not price_file.exists():
                self.logger.error(f"Price data file not found: {price_file}")
                return {}
            
            price_data = pd.read_csv(price_file)
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            
            # Merge predictions with price data
            backtest_data = pd.merge(predictions, price_data, on='timestamp', how='left')
            
            # Calculate prediction horizon returns
            prediction_horizon = self.config.get('data', {}).get('prediction_horizon_minutes', 15)
            periods_ahead = prediction_horizon // 5  # 5-minute intervals
            
            # Get future prices
            future_prices = price_data.set_index('timestamp')['close'].shift(-periods_ahead)
            backtest_data = backtest_data.merge(future_prices, left_on='timestamp', right_index=True, how='left', suffixes=('', '_future'))
            
            # Calculate actual returns
            backtest_data['actual_return'] = (backtest_data['close_future'] - backtest_data['close']) / backtest_data['close']
            
            # Apply trading strategy based on predictions
            backtest_data = self._apply_trading_strategy(backtest_data, initial_capital)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(backtest_data, initial_capital)
            
            # Generate plots
            self._plot_equity_curve(backtest_data, initial_capital)
            self._plot_drawdown(backtest_data)
            self._plot_trades(backtest_data)
            
            # Save backtest results
            results = {
                'performance_metrics': performance_metrics,
                'trade_log': backtest_data.to_dict('records')
            }
            
            results_file = self.backtest_dir / "backtest_results.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save detailed trade log
            trade_log_file = self.backtest_dir / "trade_log.csv"
            backtest_data.to_csv(trade_log_file, index=False)
            
            self.logger.info(f"Backtest completed. Final portfolio value: ${performance_metrics['final_portfolio_value']:,.2f}")
            self.logger.info(f"Backtest results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            return {}
    
    def _apply_trading_strategy(self, data: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        """Apply trading strategy based on model predictions"""
        
        try:
            # Initialize portfolio
            data['portfolio_value'] = initial_capital
            data['position'] = 0  # 0 = no position, 1 = long, -1 = short
            data['cash'] = initial_capital
            data['shares'] = 0.0
            data['trade'] = 0  # 1 = buy, -1 = sell, 0 = hold
            
            # Get risk management parameters
            position_size = self.config.get('risk_management', {}).get('position_size', 0.1)  # 10% of portfolio
            slippage = self.config.get('backtesting', {}).get('slippage', 0.001)  # 0.1% slippage
            transaction_cost = self.config.get('backtesting', {}).get('transaction_cost', 0.001)  # 0.1% transaction cost
            
            cash = initial_capital
            shares = 0.0
            position = 0
            
            for i in range(len(data)):
                current_price = data.iloc[i]['close']
                prediction = data.iloc[i]['predicted']
                actual_return = data.iloc[i]['actual_return']
                
                # Risk management - limit position size
                max_position_value = cash * position_size
                
                # Trading logic
                trade = 0
                
                # Buy signal (prediction = 1, price going up)
                if prediction == 1 and position != 1:
                    # Close short position if exists
                    if position == -1:
                        cash = cash + shares * current_price * (1 - slippage - transaction_cost)
                        shares = 0
                        trade = 1
                    
                    # Open long position
                    shares_to_buy = max_position_value / current_price
                    cost = shares_to_buy * current_price * (1 + slippage + transaction_cost)
                    
                    if cost <= cash:  # Only trade if we have enough cash
                        cash = cash - cost
                        shares = shares_to_buy
                        position = 1
                        trade = 1
                
                # Sell signal (prediction = 0, price going down)
                elif prediction == 0 and position != -1:
                    # Close long position if exists
                    if position == 1:
                        cash = cash + shares * current_price * (1 - slippage - transaction_cost)
                        shares = 0
                        trade = -1
                    
                    # Open short position (simplified - assumes ability to short)
                    shares_to_short = max_position_value / current_price
                    proceeds = shares_to_short * current_price * (1 - slippage - transaction_cost)
                    
                    cash = cash + proceeds
                    shares = -shares_to_short
                    position = -1
                    trade = -1
                
                # Update portfolio
                data.at[data.index[i], 'cash'] = cash
                data.at[data.index[i], 'shares'] = shares
                data.at[data.index[i], 'position'] = position
                data.at[data.index[i], 'trade'] = trade
                
                # Calculate portfolio value
                if position == 1:  # Long position
                    portfolio_value = cash + shares * current_price
                elif position == -1:  # Short position
                    # Simplified short position valuation
                    portfolio_value = cash + abs(shares) * (2 * current_price - current_price)
                else:  # No position
                    portfolio_value = cash
                
                data.at[data.index[i], 'portfolio_value'] = portfolio_value
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error applying trading strategy: {e}")
            return data
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, initial_capital: float) -> dict:
        """Calculate comprehensive performance metrics"""
        
        try:
            # Calculate returns
            data['portfolio_return'] = data['portfolio_value'].pct_change().fillna(0)
            
            # Total return
            total_return = (data['portfolio_value'].iloc[-1] / initial_capital) - 1
            
            # Annualized return
            num_periods = len(data)
            periods_per_year = 365 * 24 * 12  # 5-minute periods per year
            annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1
            
            # Volatility (annualized)
            volatility = data['portfolio_return'].std() * np.sqrt(periods_per_year)
            
            # Sharpe ratio (assuming risk-free rate = 0)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            data['rolling_max'] = data['portfolio_value'].expanding().max()
            data['drawdown'] = (data['portfolio_value'] - data['rolling_max']) / data['rolling_max']
            max_drawdown = data['drawdown'].min()
            
            # Win rate
            winning_trades = (data['portfolio_return'] > 0).sum()
            total_trades = (data['trade'] != 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profits = data[data['portfolio_return'] > 0]['portfolio_return'].sum()
            gross_losses = abs(data[data['portfolio_return'] < 0]['portfolio_return'].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
            
            # Number of trades
            num_trades = total_trades
            
            # Buy and hold benchmark
            buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
            
            metrics = {
                'initial_capital': initial_capital,
                'final_portfolio_value': data['portfolio_value'].iloc[-1],
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'number_of_trades': num_trades,
                'buy_and_hold_return': buy_hold_return
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _plot_equity_curve(self, data: pd.DataFrame, initial_capital: float):
        """Plot equity curve"""
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot portfolio value
            plt.plot(data['timestamp'], data['portfolio_value'], label='Strategy', linewidth=1)
            
            # Plot buy and hold benchmark
            buy_hold_value = initial_capital * (data['close'] / data['close'].iloc[0])
            plt.plot(data['timestamp'], buy_hold_value, label='Buy & Hold', linewidth=1)
            
            plt.title('Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            plot_file = self.backtest_dir / "plots" / "equity_curve.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Equity curve plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")
    
    def _plot_drawdown(self, data: pd.DataFrame):
        """Plot drawdown"""
        
        try:
            plt.figure(figsize=(12, 6))
            plt.fill_between(data['timestamp'], data['drawdown'] * 100, 0, alpha=0.3, color='red')
            plt.plot(data['timestamp'], data['drawdown'] * 100, color='red', linewidth=1)
            plt.title('Drawdown (%)')
            plt.xlabel('Time')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.tight_layout()
            
            plot_file = self.backtest_dir / "plots" / "drawdown.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Drawdown plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting drawdown: {e}")
    
    def _plot_trades(self, data: pd.DataFrame):
        """Plot trades on price chart"""
        
        try:
            # Sample data for better visualization
            sample_data = data.sample(n=min(1000, len(data)), random_state=42).sort_values('timestamp')
            
            plt.figure(figsize=(12, 8))
            
            # Plot price
            plt.subplot(2, 1, 1)
            plt.plot(sample_data['timestamp'], sample_data['close'], label='BTCUSD Price', linewidth=1)
            
            # Plot buy signals
            buy_signals = sample_data[sample_data['trade'] == 1]
            plt.scatter(buy_signals['timestamp'], buy_signals['close'], color='green', marker='^', s=50, label='Buy')
            
            # Plot sell signals
            sell_signals = sample_data[sample_data['trade'] == -1]
            plt.scatter(sell_signals['timestamp'], sell_signals['close'], color='red', marker='v', s=50, label='Sell')
            
            plt.title('Trading Signals on Price Chart')
            plt.xlabel('Time')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True)
            
            # Plot positions
            plt.subplot(2, 1, 2)
            plt.plot(sample_data['timestamp'], sample_data['position'], linewidth=1)
            plt.title('Position Over Time')
            plt.xlabel('Time')
            plt.ylabel('Position (1=Long, -1=Short, 0=Flat)')
            plt.grid(True)
            
            plt.tight_layout()
            
            plot_file = self.backtest_dir / "plots" / "trades.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Trades plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting trades: {e}")
    
    def run_walk_forward_backtest(self, model, splits: list, initial_capital: float = 10000.0) -> dict:
        """Run walk-forward backtest on time series splits"""
        
        try:
            self.logger.info("Running walk-forward backtest")
            
            # Load original price data
            price_file = self.data_dir / "BTCUSD_5min_processed.csv"
            if not price_file.exists():
                self.logger.error(f"Price data file not found: {price_file}")
                return {}
            
            price_data = pd.read_csv(price_file)
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            
            all_results = []
            portfolio_value = initial_capital
            
            for i, split_data in enumerate(splits):
                self.logger.info(f"Backtesting split {i+1}/{len(splits)}")
                
                # Get out-of-sample data
                X_val = split_data['X_val']
                timestamps_val = split_data['timestamps_val']
                
                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_val)
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                    else:
                        y_pred_proba = y_pred
                else:
                    continue
                
                # Create predictions dataframe
                pred_df = pd.DataFrame({
                    'timestamp': timestamps_val,
                    'predicted': y_pred,
                    'probability': y_pred_proba
                })
                
                # Merge with price data
                backtest_data = pd.merge(pred_df, price_data, on='timestamp', how='left')
                
                # Apply trading strategy
                backtest_data = self._apply_trading_strategy(backtest_data, portfolio_value)
                
                # Update portfolio value for next iteration
                portfolio_value = backtest_data['portfolio_value'].iloc[-1]
                
                # Calculate split metrics
                split_metrics = self._calculate_performance_metrics(backtest_data, portfolio_value)
                split_metrics['split'] = i + 1
                
                all_results.append({
                    'split': i + 1,
                    'metrics': split_metrics,
                    'data': backtest_data
                })
                
                self.logger.info(f"Split {i+1} completed. Portfolio value: ${portfolio_value:,.2f}")
            
            # Calculate overall metrics
            if all_results:
                final_value = all_results[-1]['metrics']['final_portfolio_value']
                total_return = (final_value / initial_capital) - 1
                
                results = {
                    'initial_capital': initial_capital,
                    'final_portfolio_value': final_value,
                    'total_return': total_return,
                    'split_results': all_results
                }
                
                # Save results
                results_file = self.backtest_dir / "walk_forward_backtest_results.json"
                import json
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                self.logger.info(f"Walk-forward backtest completed. Final value: ${final_value:,.2f}")
                
                return results
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward backtest: {e}")
            return {}
