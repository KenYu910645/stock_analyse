'''
Naive consecutive-move strategy.
'''


class NaiveStrategy:
    '''
    Buy one lot after three consecutive falling closes, and sell one lot after
    three consecutive rising closes.
    '''

    def __init__(self, fund, lot_size=1000):
        self.initial_fund = float(fund)
        self.current_fund = float(fund)
        self.position = 0
        self.lot_size = int(lot_size)
        self.latest_action = 'hold'
        self.trades = []
        self._previous_price = None
        self._down_days = 0
        self._up_days = 0

    @property
    def fund(self):
        return self.current_fund

    @property
    def shares(self):
        return self.position

    def total_equity(self, stock_price):
        return self.current_fund + self.position * float(stock_price)

    def run(self, stock_price):
        '''
        Process one close price and return the action for this row.
        '''
        price = float(stock_price)
        self.latest_action = 'hold'

        if self._previous_price is None:
            self._previous_price = price
            return self.latest_action

        if price < self._previous_price:
            self._down_days += 1
            self._up_days = 0
        elif price > self._previous_price:
            self._up_days += 1
            self._down_days = 0
        else:
            self._down_days = 0
            self._up_days = 0

        if self._down_days >= 3:
            self._buy(price)
        elif self._up_days >= 3:
            self._sell(price)

        self._previous_price = price
        return self.latest_action

    def _buy(self, price):
        cost = price * self.lot_size
        if self.current_fund < cost:
            return

        self.current_fund -= cost
        self.position += self.lot_size
        self.latest_action = 'buy'
        self.trades.append({
            'action': 'buy',
            'price': price,
            'shares': self.lot_size,
            'value': cost,
        })

    def _sell(self, price):
        if self.position < self.lot_size:
            return

        proceeds = price * self.lot_size
        self.current_fund += proceeds
        self.position -= self.lot_size
        self.latest_action = 'sell'
        self.trades.append({
            'action': 'sell',
            'price': price,
            'shares': self.lot_size,
            'value': proceeds,
        })
