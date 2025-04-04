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
        """Allocates funds using the trained model and buys integer shares."""
        obs = self.get_observation(stock_prices)[:6]
        action, _ = self.allocation_model.predict(obs, deterministic=True)

        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            print("⚠️ Model returned zero allocation, skipping allocation.")
            return
        action /= np.sum(action)

        buffer_cash = 0.05 * self.balance  # Keep 5% as buffer
        available_cash = self.balance - buffer_cash
        allocation_amount = available_cash * action

        print("\n📊 Morning Allocation Analysis:")
        for i, stock in enumerate(self.stocks):
            max_shares = int(allocation_amount[i] // stock_prices[stock])
            cost = max_shares * stock_prices[stock]
            if max_shares > 0 and self.balance >= cost:
                self.holdings[stock] += max_shares
                self.balance -= cost
                print(f"  ✅ Bought {max_shares} shares of {stock} at ${stock_prices[stock]:.2f} → Spent ${cost:.2f}")
            else:
                print(f"  ❌ Skipped {stock}, insufficient funds or zero shares.")

        print(f"\n💰 Remaining Cash Balance: ${self.balance:.2f}")

    def trade(self, stock_prices):
        """Executes trades based on PPO signals, only whole shares."""
        obs = self.get_observation(stock_prices)[:6]
        trade_signals, _ = self.signal_model.predict(obs, deterministic=True)

        print("\n📉 Evening Trading Signal Analysis:")
        for i, stock in enumerate(self.stocks):
            signal = trade_signals[i]
            current_price = stock_prices[stock]

            if signal < 0:  # Sell
                shares_to_sell = int(abs(signal) * self.holdings[stock])
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * current_price
                    self.holdings[stock] -= shares_to_sell
                    self.balance += proceeds
                    self.investment_fund += proceeds
                    print(f"  🔻 Sold {shares_to_sell} shares of {stock} at ${current_price:.2f} → Gained ${proceeds:.2f}")
                else:
                    print(f"  ⚠️ No shares to sell for {stock}")

            elif signal > 0:  # Buy
                max_affordable = int((signal * self.balance) // current_price)
                cost = max_affordable * current_price
                if max_affordable > 0 and self.balance >= cost:
                    self.holdings[stock] += max_affordable
                    self.balance -= cost
                    print(f"  🔺 Bought {max_affordable} shares of {stock} at ${current_price:.2f} → Spent ${cost:.2f}")
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