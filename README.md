# Stock Trading Bot Backend

This is the backend for a stock trading bot built using reinforcement learning.  
It uses DDPG for allocation and PPO for trading signals.  

- Real-time stock data from yfinance  
- Stock-wise news sentiment using `ProsusAI/finbert`  
- News fetched using Marketaux API  
- Indian stocks only  
- Sector-specific models  

## How to run

1. Set your Marketaux API key in an `.env` file:

2. Run the bot:
```bash
python bot_backend.py
