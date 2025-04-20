from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import asyncio
import schedule
import time
import threading
from .TradingBotBackend import RealTimeTradingBot  # Make sure trading_bot.py is in the same folder or in PYTHONPATH

app = FastAPI()

class InvestmentRequest(BaseModel):
    initial_amount: float
    sector: str

# Global bot instance
bot = None

@app.post("/start_investment")
async def start_investment(request: InvestmentRequest):
    global bot
    sector = request.sector
    valid_sectors = ["technology", "energy", "manufacturing", "banking"]
    if sector not in valid_sectors:
        raise HTTPException(status_code=400, detail=f"Sector must be one of {valid_sectors}")

    # Initialize bot with only one sector
    bot = RealTimeTradingBot(
        sector_models={
            "technology": ("models/ddpg_allocation_model_tech.zip", "models/ppo_trading_model_tech.zip"),
            "energy": ("models/ddpg_allocation_model_energy.zip", "models/ppo_trading_model_energy.zip"),
            "manufacturing": ("models/ddpg_allocation_model_manufacturing.zip", "models/ppo_trading_model_manufacturing.zip"),
            "banking": ("models/ddpg_allocation_model_banking.zip", "models/ppo_trading_model_banking.zip")
        },
        sector_stocks={
            "technology": ["HCLTECH.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS", "TECHM.NS"],
            "energy": ["BPCL.NS", "COALINDIA.NS", "GAIL.NS", "ONGC.NS", "POWERGRID.NS"],
            "manufacturing": ["BAJAJ-AUTO.NS", "LT.NS", "HEROMOTOCO.NS", "MARUTI.NS", "TATAMOTORS.NS"],
            "banking": ["HDFCBANK.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "SBIN.NS", "AXISBANK.NS"]
        },
        initial_balance=request.initial_amount,
        sectors=[sector]  # only one sector in a list
    )
    bot.setup_models()
    return {"message": "Investment started", "selected_sector": sector}

@app.get("/allocations")
async def get_allocations():
    if not bot:
        raise HTTPException(status_code=400, detail="Investment not started yet.")
    stock_prices = bot.fetch_stock_prices()
    allocations = bot.allocate_funds(stock_prices)
    portfolio_value = calculate_portfolio_value(bot, stock_prices)
    return {
        "allocations": allocations,
        "portfolio_value": portfolio_value,
        "cash_balance": bot.balance,
        "investment_fund": bot.investment_fund
    }

@app.get("/trades")
async def get_trades():
    if not bot:
        raise HTTPException(status_code=400, detail="Investment not started yet.")
    stock_prices = bot.fetch_stock_prices()
    trades = bot.trade(stock_prices)
    portfolio_value = calculate_portfolio_value(bot, stock_prices)
    return {
        "trades": trades,
        "portfolio_value": portfolio_value,
        "cash_balance": bot.balance,
        "investment_fund": bot.investment_fund
    }
    
@app.post("/manual/allocate")
async def manual_allocate():
    if not bot:
        raise HTTPException(status_code=400, detail="Bot not initialized")
    stock_prices = bot.fetch_stock_prices()
    allocations = bot.allocate_funds(stock_prices)
    portfolio_value = calculate_portfolio_value(bot, stock_prices)
    return {
        "allocations": allocations,
        "portfolio_value": portfolio_value,
        "cash_balance": bot.balance,
        "investment_fund": bot.investment_fund
    }

@app.post("/manual/trade")
async def manual_trade():
    if not bot:
        raise HTTPException(status_code=400, detail="Bot not initialized")
    stock_prices = bot.fetch_stock_prices()
    trades = bot.trade(stock_prices)
    portfolio_value = calculate_portfolio_value(bot, stock_prices)
    return {
        "trades": trades,
        "portfolio_value": portfolio_value,
        "cash_balance": bot.balance,
        "investment_fund": bot.investment_fund
    }

# Async functions to run scheduled tasks
async def run_allocate_funds():
    global bot
    if bot:
        stock_prices = bot.fetch_stock_prices()
        allocations = bot.allocate_funds(stock_prices)
        print("Allocations at 9 AM:", allocations)

async def run_trade():
    global bot
    if bot:
        stock_prices = bot.fetch_stock_prices()
        trades = bot.trade(stock_prices)
        print("Trades at 4 PM:", trades)
  
# Caluculates portfolio value   
def calculate_portfolio_value(bot, stock_prices):
    total_value = 0.0
    for stock, shares in bot.holdings.items():
        price = stock_prices.get(stock, 0)
        total_value += shares * price
    return total_value

# Scheduler thread function
def scheduler_thread():
    schedule.every().day.at("09:00").do(lambda: asyncio.run(run_allocate_funds()))
    schedule.every().day.at("16:00").do(lambda: asyncio.run(run_trade()))

    while True:
        schedule.run_pending()
        time.sleep(1)

# Start scheduler in background thread
threading.Thread(target=scheduler_thread, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
