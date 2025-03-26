import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os

from stable_baselines3 import DDPG,PPO
import yfinance as yf


class RealTimeTradingBot:
    def __init__(self, allocation_model_path, signal_model_path, initial_balance=100000):
        self.stocks = ['AAPL', 'JPM', 'MSFT', 'UNH', 'V']
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.investment_fund = 0  # Profits from sales for reinvestment
        self.holdings = {stock: 0 for stock in self.stocks}

        # Load trained models
        custom_objects = {"lr_schedule": lambda _: 0.0001,
                          "clip_range": lambda _: 0.2}
        self.allocation_model = DDPG.load(allocation_model_path,custom_objects=custom_objects)
        self.signal_model = PPO.load(signal_model_path,custom_objects=custom_objects)

    def fetch_stock_prices(self):
        """Fetch the previous day's closing prices for the selected stocks."""
        end_date = datetime.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)  # Fetch 1 week to ensure availability

        stock_data = yf.download(self.stocks, start=start_date, end=end_date, progress=False)
        latest_prices = stock_data["Close"].iloc[-1]  # Get the last available closing prices
        
        return latest_prices.to_dict()  # Convert to { 'AAPL': price, ... }

    def get_observation(self, stock_prices):
        """Formats stock prices and balance for the models."""
        prices = np.array([stock_prices[stock] for stock in self.stocks])
        return np.append(prices, [self.balance, self.investment_fund])

    def allocate_funds(self, stock_prices):
        """Allocates funds across selected stocks using the trained model."""
        obs = self.get_observation(stock_prices)[:6]
        action, _ = self.allocation_model.predict(obs, deterministic=True)

        action = np.clip(action, 0, 1)  # Ensure values are between 0 and 1
        action /= np.sum(action)  # Normalize to sum to 1

        allocation_amount = self.balance * action  # Divide balance among stocks
        allocation_results = {}

        print("\n📊 Morning Allocation Analysis:")

        for i, stock in enumerate(self.stocks):
            shares_bought = allocation_amount[i] / stock_prices[stock]  # Calculate shares bought
            self.holdings[stock] += shares_bought
            self.balance -= allocation_amount[i]
            allocation_results[stock] = shares_bought

            print(f"  ✅ Bought {shares_bought:.4f} shares of {stock} at ${stock_prices[stock]:.2f} each "
                  f"→ Spent ${allocation_amount[i]:.2f}")

        print(f"\n💰 Remaining Cash Balance: ${self.balance:.2f}")

    def trade(self, stock_prices):
        """Checks buy/sell signals at market close and executes trades."""
        obs = self.get_observation(stock_prices)[:6]
        trade_signals, _ = self.signal_model.predict(obs, deterministic=True)
        
        trade_results = {}

        print("\n📉 Evening Trading Signal Analysis:")

        for i, stock in enumerate(self.stocks):
            morning_shares = self.holdings.get(stock, 0.0)
            signal_value = trade_signals[i]

            if signal_value < 0:  # Sell signal
                shares_to_sell = abs(signal_value) * morning_shares
                sell_value = shares_to_sell * stock_prices[stock]
                self.balance += sell_value
                self.holdings[stock] -= shares_to_sell
                self.investment_fund += sell_value  # Store profit for reinvestment
                trade_results[stock] = -shares_to_sell

                print(f"  🔻 Sold {shares_to_sell:.4f} shares of {stock} at ${stock_prices[stock]:.2f} each "
                      f"→ Earned ${sell_value:.2f}")

            elif signal_value > 0:  # Buy signal (if enough funds)
                shares_to_buy = signal_value * (self.balance / stock_prices[stock])
                buy_value = shares_to_buy * stock_prices[stock]
                if self.balance >= buy_value:
                    self.balance -= buy_value
                    self.holdings[stock] += shares_to_buy
                    trade_results[stock] = shares_to_buy

                    print(f"  🔺 Bought {shares_to_buy:.4f} shares of {stock} at ${stock_prices[stock]:.2f} each "
                          f"→ Spent ${buy_value:.2f}")
                else:
                    print(f"  ❌ Not enough balance to buy {stock}")

            else:
                print(f"  ⏸ No action for {stock} (holding position)")

        print(f"\n💰 Updated Cash Balance: ${self.balance:.2f}")
        print(f"🏦 Investment Fund (Profits for Reinvestment): ${self.investment_fund:.2f}")

    def run(self):
        """Runs the trading bot every trading day."""
        
        test_mode = True
        
        while True:
            now = datetime.now()

            # Fetch stock prices at 9:00 AM (Market Open)
            if test_mode or (now.hour == 9 and now.minute == 0):
                print("📈 Fetching morning prices...")
                stock_prices = self.fetch_stock_prices()
                self.allocate_funds(stock_prices)

            # Fetch stock prices again at 4:00 PM (Market Close)
            '''elif'''
            if test_mode or (now.hour == 16 and now.minute == 0):
                print("📉 Fetching closing prices...")
                stock_prices = self.fetch_stock_prices()
                self.trade(stock_prices)

                # Reinvest profits
                if self.investment_fund > 0:
                    self.balance += self.investment_fund
                    self.investment_fund = 0
                    self.allocate_funds(stock_prices)
                    
            if test_mode:
                break

            time.sleep(60)  # Check every minute

# Run the bot
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DDPG_MODEL_PATH = os.path.join(BASE_DIR, "models", "ddpg_allocation_model.zip")
PPO_MODEL_PATH = os.path.join(BASE_DIR,"models","ppo_trading_model")
trading_bot = RealTimeTradingBot(DDPG_MODEL_PATH,PPO_MODEL_PATH)
trading_bot.run()