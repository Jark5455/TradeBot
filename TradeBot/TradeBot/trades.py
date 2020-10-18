import alpaca_trade_api as tradeapi
import json

from datetime import datetime

from typing import List
from typing import Union
from typing import Optional

API_KEY = 'PKEV8QACEX3PR55ST3FG'
API_SECRET_KEY = 'k4QH3bHIcg2WuyPhXKilTqp8eePMxbRNWxy9aXO6'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, API_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
class Trade():
    def __init__(self):
        self.order = {}
        self.trade_id = ""

        self.side = ""
        self.side_opposite = ""
        self.enter_or_exit = ""
        self.enter_or_exit_opposite = ""

        self._order_response = {}
        self._triggered_added = False
        self._multi_leg = False

    def new_trade(self, trade_id: str, order_type: str, side: str, enter_or_exit: str, price: float = 0.00, stop_limit_price: float = 0.00) -> dict:

        self.trade_id = trade_id
        self.price = price
        self.enter_or_exit = enter_or_exit
        self.side = side

        self.order_types = {
            'mkt':'market',
            'lmt':'limit',
            'stop':'stop',
            'stop_lmt':'stop_limit',
        }

        self.order_instructions = {
            'enter':{
                'long':'buy',
                'short':'sell_short'
            },
            'exit':{
                'long':'sell',
                'short':'buy_to_cover'                
            }
        }

        self.order = {
            "side": self.order_instructions[enter_or_exit][side],
            "symbol": None,
            "type": self.order_types[order_type],
            "qty": 0,
            "time_in_force": "gtc",
            "childOrderStrategies": []
        }

        if self.order['type'] == 'stop':
            self.order['stop_price'] = price

        elif self.order['type'] == 'limit':
            self.order['limit_price'] = price

        elif self.order['type'] == 'stop_limit':
            self.order['stop_loss']['stop_price'] = price
            self.order['stop_loss']['limit_price'] = stop_limit_price

        if self.is_stop_order or self.is_stop_limit_order:
            self.stop_price = price
        else:
            self.stop_price = 0.0

        if self.is_stop_limit_order:
            self.stop_limit_price = stop_limit_price
        else:
            self.stop_limit_price = 0.0

        if self.is_limit_order:
            self.limit_price = price
        else:
            self.limit_price = 0.0

        if self.enter_or_exit == 'enter':
            self.enter_or_exit_opposite = 'exit'
        if self.enter_or_exit == 'exit':
            self.enter_or_exit_opposite = 'enter'

        if self.side == 'long':
            self.side_opposite = 'short'
        if self.side == 'short':
            self.side_opposite = 'long'

        return self.order

    def instrument(self, symbol: str, quantity: int) -> dict:

        leg = self.order

        leg['symbol'] = symbol
        leg['qty'] = quantity

        self.order_size = quantity
        self.symbol = symbol

        return leg

    def modify_side(self, side: Optional[str] , leg_id: int = 0) -> None:
        if side and side not in ['buy', 'sell', 'sell_short', 'buy_to_cover']:
            raise ValueError(
                "The side you have specified is not valid. Please choose a valid side: ['buy', 'sell', 'sell_short', 'buy_to_cover']"
            )
        
        if side:
            self.order['side'] = side
        else:
            self.order['side'] = self.order_instructions[self.enter_or_exit][self.side_opposite]

    def add_box_range(self, profit_size: float = 0.00, percentage: bool = False, stop_limit: bool = False):
        if not self._triggered_added:
            self._convert_to_trigger()

        self.add_take_profit(profit_size=profit_size, percentage=percentage)
        if not stop_limit:
            self.add_stop_loss(stop_size=profit_size, percentage=percentage)

    def add_stop_loss(self, stop_size: float, percentage: bool = False) -> bool:
        if not self._triggered_added:
            self._convert_to_trigger()
        
        if self.order['type'] == 'market':
            price = self.price
        elif self.order['type'] == 'limit':
            price = self.price
        
        if percentage:
            adjustment = 1.0 - stop_size
            new_price = self._calculate_new_price(price=price, adjustment=adjustment, percentage=True)
        else:
            adjustment = -stop_size
            new_price = self._calculate_new_price(price=price, adjustment=adjustment, percentage=False)

        stop_loss_order = {
            "side": self.order_instructions[self.enter_or_exit_opposite][self.side],
            "symbol": self.symbol,
            "type": "stop",
            "qty": self.order_size,
            "time_in_force": "gtc",
            "stop_loss": {
                "stop_price": new_price,
            }
        }

        self.stop_loss_order = stop_loss_order
        self.order['childOrderStrategies'].append(self.stop_loss_order)
        return True

    def add_stop_limit(self, stop_size: float, limit_size: float, stop_percentage: bool = False, limit_percentage: bool = False):
        if not self._triggered_added:
            self._convert_to_trigger()
        
        if self.order_type == 'mkt':
            price = self.price
            
        elif self.order_type == 'lmt':
            price = self.price
        
        if stop_percentage:
            adjustment = 1.0 - stop_size
            stop_price = self._calculate_new_price(
                price=price,
                adjustment=adjustment,
                percentage=True
            )
        else:
            adjustment = -stop_size
            stop_price = self._calculate_new_price(
                price=price,
                adjustment=adjustment,
                percentage=False
            )

        if limit_percentage:
            adjustment = 1.0 - limit_size
            limit_price = self._calculate_new_price(
                price=price,
                adjustment=adjustment,
                percentage=True
            )
        else:
            adjustment = -limit_size
            limit_price = self._calculate_new_price(
                price=price,
                adjustment=adjustment,
                percentage=False
            )

        stop_limit_order = {
            "side": self.order_instructions[self.enter_or_exit_opposite][self.side],
            "symbol": self.symbol,
            "type": "stop",
            "qty": self.order_size,
            "time_in_force": "gtc",
            "stop_loss": {
                "stop_price": stop_price,
                "limit_price": limit_price
            }
        }

        self.stop_limit_order = stop_limit_order
        self.order['childOrderStrategies'].append(self.stop_limit_order)

        return True

    def _calculate_new_price(self, price: float, adjustment: float, percentage: bool) -> float:
        if percentage:
            new_price = price * adjustment
        else:
            new_price = price + adjustment
        if new_price < 1:
            new_price = round(new_price,4)
        else:
            new_price = round(new_price, 2)
        return new_price

    def add_take_profit(self, profit_size: float, percentage: bool = False) -> bool:
        if not self._triggered_added:
            self._convert_to_trigger()

        if self.order_type == 'mkt':
            price = self.price
        elif self.order_type == 'lmt':
            price = self.price

        if percentage:
            adjustment = 1.0 + profit_size
            new_price = self._calculate_new_price(
                price=price,
                adjustment=adjustment,
                percentage=True
            )
        else:
            adjustment = profit_size
            new_price = self._calculate_new_price(
                price=price,
                adjustment=adjustment,
                percentage=False
            )

        take_profit_order = {
            "side": self.order_instructions[self.enter_or_exit_opposite][self.side],
            "symbol": self.symbol,
            "type": "limit",
            "limit_price": new_price,
            "qty": self.order_size,
            "time_in_force": "gtc",
            "stop_loss": {
                "stop_price": stop_price,
                "limit_price": limit_price
            }
        }
        self.take_profit_order = take_profit_order
        self.order['childOrderStrategies'].append(self.take_profit_order)

        return True

    def _convert_to_trigger(self):
        if self.order and self._triggered_added == False:
            self.order['order_class'] = 'oto'
            self.order['childOrderStrategies'] = []
            self._triggered_added = True

    def _generate_order_id(self) -> str:
        if self.order:     
            order_id = "{symbol}_{side}_{enter_or_exit}_{timestamp}"
            order_id = order_id.format(
                symbol=self.symbol,
                side=self.side,
                enter_or_exit=self.enter_or_exit,
                timestamp=datetime.now().timestamp()
            )
            return order_id
        else:
            return ""

    def modify_price(self, new_price: float, price_type: str) -> None:
        if price_type == 'price':
            self.order['price'] = new_price
        elif price_type == 'stop-price' and self.is_stop_order:
            self.order['stopPrice'] = new_price
            self.stop_price = new_price
        elif price_type == 'limit-price' and self.is_limit_order:
            self.order['price'] = new_price
            self.price = new_price
        elif price_type == 'stop-limit-limit-price' and self.is_stop_limit_order:
            self.order['price'] = new_price
            self.stop_limit_price = new_price
        elif price_type == 'stop-limit-stop-price' and self.is_stop_limit_order:
            self.order['stopPrice'] = new_price
            self.stop_price = new_price


    def execute_trade(self):
        if len(self.order['childOrderStrategies']) > 0:
            strategy = self.order['childOrderStrategies'][0]
            if strategy['type'] == 'stop':
                api.submit_order(
                symbol=strategy['symbol'],
                qty=strategy['qty'],
                side=self.order['side'],
                type=strategy['type'],
                time_in_force=strategy['time_in_force'],
                stop_price=strategy['stop_loss']['stop_price'],
                stop_loss=strategy['stop_loss']
                )

            elif strategy['type'] == 'limit':
                api.submit_order(
                symbol=strategy['symbol'],
                qty=strategy['qty'],
                side=self.order['side'],
                type=strategy['type'],
                time_in_force=strategy['time_in_force'],
                limit_price=strategy['stop_loss']['limit_price'],
                stop_loss=strategy['stop_loss']
                )

            else:
                api.submit_order(
                symbol=strategy['symbol'],
                qty=strategy['qty'],
                side=self.order['side'],
                type=strategy['type'],
                time_in_force=strategy['time_in_force'],
                stop_loss=strategy['stop_loss']
                )
        else:
            order = api.submit_order(
            symbol=self.order['symbol'],
            qty=self.order_size,
            side=self.order['side'],    
            type='market',    
            time_in_force='gtc'    
            )
            print(order)

    @property
    def is_stop_order(self) -> bool: 
        if self.order['type'] != 'stop':
            return False
        else:
            return True

    @property
    def is_stop_limit_order(self) -> bool:
        if self.order['type'] != 'stop_limit':
            return False
        else:
            return True

    @property
    def is_limit_order(self) -> bool:
        if self.order['type'] != 'limit':
            return False
        else:
            return True