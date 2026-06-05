'''
Buy and hold strategy.
'''
from strategies.trade_cost import calculate_trade_cost


class BuyAndHoldStrategy:
    '''
    Buy all affordable full lots on the first day and sell everything on the
    final day.
    '''

    def __init__(self, fund, lot_size=1000):
        self.initial_fund = float(fund)
        self.current_fund = float(fund)
        self.position = 0
        self.lot_size = int(lot_size)
        self.latest_action = 'hold'
        self.trades = []
        self._day_index = 0
        self._final_day_index = None

    @property
    def fund(self):
        return self.current_fund

    @property
    def shares(self):
        return self.position

    def total_equity(self, stock_price):
        return self.current_fund + self.position * float(stock_price)

    def prepare(self, price_series):
        prices = list(price_series)
        self._day_index = 0
        self._final_day_index = len(prices) - 1 if prices else None

    def run(self, stock_price):
        '''
        Process one close price and return the action for this row.
        '''
        price = float(stock_price)
        self.latest_action = 'hold'

        if self._day_index == 0:
            self._buy(price)

        if self._day_index == self._final_day_index:
            self._sell(price)

        self._day_index += 1
        return self.latest_action

    def _buy(self, price):
        lots = int(self.current_fund // (price * self.lot_size))

        while lots > 0:
            shares = lots * self.lot_size
            value = price * shares
            costs = calculate_trade_cost('buy', value)
            cash_required = value + costs['transaction_cost']

            if self.current_fund >= cash_required:
                self.current_fund -= cash_required
                self.position += shares
                self.latest_action = 'buy'
                self.trades.append({
                    'action': 'buy',
                    'price': price,
                    'shares': shares,
                    'value': value,
                    **costs,
                })
                return

            lots -= 1

    def _sell(self, price):
        if self.position == 0:
            return

        shares = self.position
        value = price * shares
        costs = calculate_trade_cost('sell', value)
        self.current_fund += value - costs['transaction_cost']
        self.position = 0
        self.latest_action = 'sell'
        self.trades.append({
            'action': 'sell',
            'price': price,
            'shares': shares,
            'value': value,
            **costs,
        })
