import threading
import time as truetime
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
import websocket, json
from pandas_datareader import data as pdr
from datetime import datetime, time, timezone, timedelta, date
from typing import List, Dict, Union, Optional
from TradeBot.portfolio import Portfolio
from TradeBot.stock_frame import StockFrame
from TradeBot.trades import Trade

class TradeBot():

    def __init__(self) -> None:
        self.trades = {}
        self.historical_prices = {}
        self.current_prices = {}

    @property
    def pre_market_open(self) -> bool:
        pre_market_start_time = datetime.utcnow().replace(hour=13,minute=00,second=00).timestamp()
        market_start_time = datetime.utcnow().replace(hour=13,minute=30,second=00).timestamp()
        right_now = datetime.utcnow().timestamp()

        if market_start_time >= right_now >= pre_market_start_time:
            return True
        else:
            return False

    @property
    def post_market_open(self):
        post_market_end_time = datetime.utcnow().replace(hour=22,minute=00,second=00).timestamp()
        market_end_time = datetime.utcnow().replace(hour=20,minute=00,second=00).timestamp()
        right_now = datetime.utcnow().timestamp()

        if post_market_end_time >= right_now >= market_end_time:
          return True
        else:
            return False

    @property
    def regular_market_open(self):
        market_start_time = datetime.utcnow().replace(hour=13,minute=30,second=00).timestamp()
        market_end_time = datetime.utcnow().replace(hour=20,minute=00,second=00).timestamp()
        right_now = datetime.utcnow().timestamp()

        if market_end_time >= right_now >= market_start_time:
            return True
        else:
            return False

    def create_portfolio(self):
        self.portfolio = Portfolio()
        return self.portfolio

    def ws_on_open(self, ws):
        print("Connection Opened")
        symbols = self.portfolio.positions.keys()
        auth_data = {"action": "authenticate","data": {"key_id": 'API_KEY', "secret_key": 'API_SECRET_KEY'}}
        ws.send(json.dumps(auth_data))

        for symbol in symbols:
            listen_message = {"action": "listen", "data": {"streams": ["Q.{symbol}".format(symbol=symbol)]}}
            ws.send(json.dumps(listen_message))

    def ws_on_message(self, ws, message):
        x = json.loads(message)
        if x["stream"] == "authorization" or x["stream"] == "listening":
            pass
        else:
            symbol = x["data"]["T"]
            price = x["data"]["p"]
            self.current_prices[symbol] = price

    def ws_on_close(self, ws):
        print("Connection Closed")
        self.grab_current_quotes()

    def open_ws(self):
        socket = "wss://data.alpaca.markets/stream"
        ws = websocket.WebSocketApp(socket, on_open = lambda ws: self.ws_on_open(ws), on_message = lambda ws,msg: self.ws_on_message(ws, msg), on_close = lambda ws: self.ws_on_close(ws))
        ws.run_forever()

    def grab_current_quotes(self) -> dict:
        thr = threading.Thread(target=self.open_ws, args=(), kwargs={})
        thr.start()
        truetime.sleep(90)
        return self.current_prices

    def grab_historical_prices(self, start: datetime, end: datetime, bar: str = '1d', symbols: Optional[List[str]] = None) -> dict:
        self._bar = bar

        new_prices = []

        if not symbols:
            symbols = self.portfolio.positions.keys()

        for symbol in symbols:
            historical_price_response = yf.download(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), group_by="ticker", interval=bar)
            historical_price_response_dict = historical_price_response.to_dict(orient='index')

            for date in historical_price_response_dict:
                new_price_mini_dict = {}
                new_price_mini_dict['symbol'] = symbol
                new_price_mini_dict['open'] = historical_price_response_dict[date]['Open']
                new_price_mini_dict['close'] = historical_price_response_dict[date]['Close']
                new_price_mini_dict['low'] = historical_price_response_dict[date]['Low']
                new_price_mini_dict['high'] = historical_price_response_dict[date]['High']
                new_price_mini_dict['volume'] = historical_price_response_dict[date]['Volume']
                new_price_mini_dict['datetime'] = date.to_pydatetime()
                new_prices.append(new_price_mini_dict)

        self.historical_prices['aggregated'] = new_prices
        return self.historical_prices

    def create_stock_frame(self, data: List[dict]) -> StockFrame:
        self.stock_frame = StockFrame(data=data)
        return self.stock_frame

    def create_trade(self, trade_id: str, enter_or_exit: str, long_or_short: str, order_type: str = 'mkt', price: float = 0.0, stop_limit_price=0.0) -> Trade:
        trade = Trade()

        trade.new_trade(
            trade_id=trade_id,
            order_type=order_type,
            side=long_or_short,
            enter_or_exit=enter_or_exit,
            price=price,
            stop_limit_price=stop_limit_price
        )

        self.trades[trade_id] = trade
        return trade
