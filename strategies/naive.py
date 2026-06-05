'''
Naive consecutive-move strategy.
'''
from strategies.trade_cost import calculate_trade_cost


class NaiveStrategy:
    '''
    Buy all affordable full lots after three consecutive falling closes, and
    sell the full position after three consecutive rising closes.
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
        lots = int(self.current_fund // (price * self.lot_size))
        shares = 0
        value = 0
        costs = {'fee': 0, 'tax': 0, 'transaction_cost': 0}

        while lots > 0:
            candidate_shares = lots * self.lot_size
            candidate_value = price * candidate_shares
            candidate_costs = calculate_trade_cost('buy', candidate_value)
            cash_required = candidate_value + candidate_costs['transaction_cost']

            if self.current_fund >= cash_required:
                shares = candidate_shares
                value = candidate_value
                costs = candidate_costs
                break

            lots -= 1

        if shares == 0:
            return

        self.current_fund -= value + costs['transaction_cost']
        self.position += shares
        self.latest_action = 'buy'
        self.trades.append({
            'action': 'buy',
            'price': price,
            'shares': shares,
            'value': value,
            **costs,
        })

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
