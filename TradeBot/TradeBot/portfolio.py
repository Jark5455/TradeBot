import numpy as np
from pandas import DataFrame
from typing import Tuple, List, Optional

class Portfolio():
    def __init__(self) -> None:

        self.positions = {}
        self.positions_count = 0

        self.profit_loss = 0.00
        self.market_value = 0.00
        self.risk_tolerance = 0.00

        self._historical_prices = []

    def add_position(self, symbol: str, asset_type: str, purchase_date: Optional[str] = None, selling_date: Optional[str] = None, quantity: int = 0, purchase_price: float = 0.0) -> dict:
        
        self.positions[symbol] = {}
        self.positions[symbol]['symbol'] = symbol
        self.positions[symbol]['quantity'] = quantity
        self.positions[symbol]['purchase_price'] = purchase_price
        self.positions[symbol]['purchase_date'] = purchase_date
        self.positions[symbol]['selling_date'] = selling_date
        self.positions[symbol]['asset_type'] = asset_type

    def add_positions(self, positions: List[dict]) -> dict:
        if isinstance(positions, list):

            for position in positions:

                self.add_position(
                    symbol=position['symbol'],
                    asset_type=position['asset_type'],
                    quantity=position.get('quantity', 0),
                    purchase_price=position.get('purchase_price', 0.0),
                    purchase_date=position.get('purchase_date', None),
                    selling_date=position.get('selling_date', None)
                )

            return self.positions

        else:
            raise TypeError('Positions must be a list of dictionaries.')

    def remove_position(self, symbol: str) -> Tuple[bool, str]:
        if symbol in self.positions:
            del self.positions[symbol]
            return (True, "{symbol} was successfully removed.".format(symbol=symbol))
        else:
            return (False, "{symbol} did not exist in the porfolio.".format(symbol=symbol))

    def in_portfolio(self, symbol: str) -> bool:
        if symbol in self.positions:
            return True
        else:
            return False

    def is_profitable(self, symbol: str, current_price: float) -> bool:
        if self.in_portfolio(symbol=symbol):
            purchase_price = self.positions[symbol]['purchase_price']
        else:
            raise KeyError("The Symbol you tried to request does not exist.")

        if (purchase_price <= current_price):
            return True
        elif (purchase_price > current_price):
            return False

    def total_allocation(self):
        pass

    def risk_exposure(self):
        pass

    def total_market_value(self):
        pass
