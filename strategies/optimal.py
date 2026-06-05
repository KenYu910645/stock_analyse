'''
Future-knowing theoretical optimal strategy.
'''
from strategies.trade_cost import calculate_trade_cost


class OptimalStrategy:
    '''
    Compute and replay an all-in/all-out future-knowing optimum.
    '''

    def __init__(self, fund, lot_size=1000):
        self.initial_fund = float(fund)
        self.current_fund = float(fund)
        self.position = 0
        self.lot_size = int(lot_size)
        self.latest_action = 'hold'
        self.trades = []
        self._planned_actions = []
        self._day_index = 0

    @property
    def fund(self):
        return self.current_fund

    @property
    def shares(self):
        return self.position

    def total_equity(self, stock_price):
        return self.current_fund + self.position * float(stock_price)

    def prepare(self, price_series):
        '''
        Build the optimal all-in/all-out action plan for the full price series.
        '''
        prices = [float(price) for price in price_series]
        self._planned_actions = ['hold'] * len(prices)
        self._day_index = 0

        if not prices:
            return

        cash_path = {
            'cash': self.initial_fund,
            'shares': 0,
            'actions': [],
        }
        hold_path = None

        for day_index, price in enumerate(prices):
            is_final_day = day_index == len(prices) - 1
            next_cash_path = cash_path
            next_hold_path = hold_path

            if cash_path is not None and not is_final_day:
                buy_path = self._buy_path(cash_path, price, day_index)
                if self._is_better_hold_path(buy_path, next_hold_path, price):
                    next_hold_path = buy_path

            if hold_path is not None:
                sell_path = self._sell_path(hold_path, price, day_index)
                if self._is_better_cash_path(sell_path, next_cash_path):
                    next_cash_path = sell_path

            cash_path = next_cash_path
            hold_path = next_hold_path

        final_path = cash_path
        if hold_path is not None:
            forced_sell_path = self._sell_path(hold_path, prices[-1], len(prices) - 1)
            if self._is_better_cash_path(forced_sell_path, final_path):
                final_path = forced_sell_path

        self._planned_actions = ['hold'] * len(prices)
        for day_index, action in final_path['actions']:
            self._planned_actions[day_index] = action

    def run(self, stock_price):
        '''
        Replay the precomputed optimal action for this row.
        '''
        price = float(stock_price)
        action = (
            self._planned_actions[self._day_index]
            if self._day_index < len(self._planned_actions)
            else 'hold'
        )
        self.latest_action = 'hold'

        if action == 'buy':
            self._buy(price)
        elif action == 'sell':
            self._sell(price)

        self._day_index += 1
        return self.latest_action

    def _buy_path(self, path, price, day_index):
        buy = self._calculate_all_in_buy(path['cash'], price)
        if buy is None:
            return None

        return {
            'cash': path['cash'] - buy['value'] - buy['transaction_cost'],
            'shares': buy['shares'],
            'actions': path['actions'] + [(day_index, 'buy')],
        }

    def _sell_path(self, path, price, day_index):
        if path['shares'] <= 0:
            return None

        value = price * path['shares']
        costs = calculate_trade_cost('sell', value)

        return {
            'cash': path['cash'] + value - costs['transaction_cost'],
            'shares': 0,
            'actions': path['actions'] + [(day_index, 'sell')],
        }

    def _is_better_cash_path(self, candidate, current):
        if candidate is None:
            return False
        if current is None:
            return True
        return candidate['cash'] > current['cash']

    def _is_better_hold_path(self, candidate, current, price):
        if candidate is None:
            return False
        if current is None:
            return True

        candidate_equity = candidate['cash'] + candidate['shares'] * price
        current_equity = current['cash'] + current['shares'] * price
        if candidate_equity != current_equity:
            return candidate_equity > current_equity

        return candidate['shares'] > current['shares']

    def _calculate_all_in_buy(self, cash, price):
        lots = int(cash // (price * self.lot_size))

        while lots > 0:
            shares = lots * self.lot_size
            value = price * shares
            costs = calculate_trade_cost('buy', value)
            cash_required = value + costs['transaction_cost']

            if cash >= cash_required:
                return {
                    'shares': shares,
                    'value': value,
                    **costs,
                }

            lots -= 1

        return None

    def _buy(self, price):
        buy = self._calculate_all_in_buy(self.current_fund, price)
        if buy is None:
            return

        self.current_fund -= buy['value'] + buy['transaction_cost']
        self.position += buy['shares']
        self.latest_action = 'buy'
        self.trades.append({
            'action': 'buy',
            'price': price,
            **buy,
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
