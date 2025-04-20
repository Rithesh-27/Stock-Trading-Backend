# trading_bot.py

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from stable_baselines3 import DDPG, PPO


class RealTimeTradingBot:
    def __init__(self, sector_models, sector_stocks, initial_balance=50000, sectors=[]):
        self.sector_models = sector_models
        self.sector_stocks = sector_stocks
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.investment_fund = 0
        self.holdings = {}
        self.selected_sector = None
        self.stocks = []
        self.allocation_model = None
        self.signal_model = None
        self.selected_sectors = sectors[:2]  # only top 2

        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def fetch_stock_prices(self):
        end_date = datetime.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)

        stock_data = yf.download(self.stocks, start=start_date, end=end_date, progress=False, auto_adjust=True)
        latest_prices = stock_data["Close"].iloc[-1]
        return latest_prices.to_dict()

    def get_observation(self, stock_prices):
        prices = np.array([stock_prices[stock] for stock in self.stocks])
        return np.append(prices, [self.balance, self.investment_fund])

    def setup_models(self):
        self.stocks = [stock for s in self.selected_sectors for stock in self.sector_stocks[s]]
        self.holdings = {stock: 0 for stock in self.stocks}

        # Load and merge models
        self.allocation_models = {}
        self.signal_models = {}

        custom_objects = {"lr_schedule": lambda _: 0.0001, "clip_range": lambda _: 0.2}
        for sector in self.selected_sectors:
            alloc_path, signal_path = self.sector_models[sector]
            self.allocation_models[sector] = DDPG.load(alloc_path, custom_objects=custom_objects)
            self.signal_models[sector] = PPO.load(signal_path, custom_objects=custom_objects)

    def allocate_funds(self, stock_prices):
        allocation_result = []
        buffer_cash = 0.05 * self.balance
        available_cash = self.balance - buffer_cash

        for sector in self.selected_sectors:
            model = self.allocation_models[sector]
            stocks = self.sector_stocks[sector]
            obs = np.append(
                [stock_prices[s] for s in stocks],
                [self.balance, self.investment_fund]
            )
            action, _ = model.predict(obs[:6], deterministic=True)
            action = np.clip(action, 0, 1)
            if np.sum(action) == 0:
                continue
            action /= np.sum(action)

            allocation_amount = available_cash * action
            for i, stock in enumerate(stocks):
                max_shares = int(allocation_amount[i] // stock_prices[stock])
                cost = max_shares * stock_prices[stock]
                if max_shares > 0 and self.balance >= cost:
                    self.holdings[stock] += max_shares
                    self.balance -= cost
                    allocation_result.append({
                        "stock": stock,
                        "shares": max_shares,
                        "price": stock_prices[stock],
                        "cost": cost
                    })
        return allocation_result

    def trade(self, stock_prices):
        trade_result = []
        for sector in self.selected_sectors:
            model = self.signal_models[sector]
            stocks = self.sector_stocks[sector]
            obs = np.append(
                [stock_prices[s] for s in stocks],
                [self.balance, self.investment_fund]
            )
            signals, _ = model.predict(obs[:6], deterministic=True)
            signals = np.where(np.abs(signals) < 0.00005, 0, signals * 20)
            signals = np.clip(signals, -1, 1)

            for i, stock in enumerate(stocks):
                signal = signals[i]
                current_price = stock_prices[stock]

                if signal < 0:
                    shares_to_sell = int(abs(signal) * self.holdings[stock])
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_price
                        self.holdings[stock] -= shares_to_sell
                        self.balance += proceeds
                        self.investment_fund += proceeds
                        trade_result.append({
                            "stock": stock,
                            "action": "sell",
                            "shares": shares_to_sell,
                            "price": current_price,
                            "amount": proceeds
                        })
                elif signal > 0:
                    max_affordable = int((signal * self.balance) // current_price)
                    cost = max_affordable * current_price
                    if max_affordable > 0 and self.balance >= cost:
                        self.holdings[stock] += max_affordable
                        self.balance -= cost
                        trade_result.append({
                            "stock": stock,
                            "action": "buy",
                            "shares": max_affordable,
                            "price": current_price,
                            "amount": cost
                        })
        return trade_result

    def run_once(self):
        self.setup_models()
        stock_prices = self.fetch_stock_prices()
        allocations = self.allocate_funds(stock_prices)
        trades = self.trade(stock_prices)
        return {
            "selected_sectors": self.selected_sectors,
            "allocations": allocations,
            "trades": trades,
            "final_cash": self.balance,
            "holdings": self.holdings
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)