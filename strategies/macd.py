'''
MACD crossover strategy.
'''
from strategies.trade_cost import calculate_trade_cost


class MACDStrategy:
    '''
    Buy one lot on DIF crossing above DEM, and sell one lot on DIF crossing
    below DEM.
    '''

    def __init__(
        self,
        fund,
        lot_size=1000,
        fast_period=12,
        slow_period=26,
        signal_period=9,
    ):
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError('MACD periods must be greater than 0.')
        if fast_period >= slow_period:
            raise ValueError('fast_period must be less than slow_period.')

        self.initial_fund = float(fund)
        self.current_fund = float(fund)
        self.position = 0
        self.lot_size = int(lot_size)
        self.fast_period = int(fast_period)
        self.slow_period = int(slow_period)
        self.signal_period = int(signal_period)
        self.latest_action = 'hold'
        self.trades = []

        self.fast_ema = None
        self.slow_ema = None
        self.dif = None
        self.dem = None
        self.osc = None
        self.previous_dif = None
        self.previous_dem = None
        self.price_count = 0
        self._fast_alpha = 2 / (self.fast_period + 1)
        self._slow_alpha = 2 / (self.slow_period + 1)
        self._signal_alpha = 2 / (self.signal_period + 1)

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
        self.price_count += 1

        self.fast_ema = self._update_ema(self.fast_ema, price, self._fast_alpha)
        self.slow_ema = self._update_ema(self.slow_ema, price, self._slow_alpha)
        self.dif = self.fast_ema - self.slow_ema
        self.dem = self._update_ema(self.dem, self.dif, self._signal_alpha)
        self.osc = self.dif - self.dem

        if self.price_count < self.slow_period + self.signal_period:
            self._remember_previous_macd()
            return self.latest_action

        if self.previous_dif is None or self.previous_dem is None:
            self._remember_previous_macd()
            return self.latest_action

        crossed_up = self.previous_dif <= self.previous_dem and self.dif > self.dem
        crossed_down = self.previous_dif >= self.previous_dem and self.dif < self.dem

        if crossed_up:
            self._buy(price)
        elif crossed_down:
            self._sell(price)

        self._remember_previous_macd()
        return self.latest_action

    def _update_ema(self, previous_ema, value, alpha):
        if previous_ema is None:
            return value
        return alpha * value + (1 - alpha) * previous_ema

    def _remember_previous_macd(self):
        self.previous_dif = self.dif
        self.previous_dem = self.dem

    def _buy(self, price):
        value = price * self.lot_size
        costs = calculate_trade_cost('buy', value)
        cash_required = value + costs['transaction_cost']
        if self.current_fund < cash_required:
            return

        self.current_fund -= cash_required
        self.position += self.lot_size
        self.latest_action = 'buy'
        self.trades.append({
            'action': 'buy',
            'price': price,
            'shares': self.lot_size,
            'value': value,
            **costs,
        })

    def _sell(self, price):
        if self.position < self.lot_size:
            return

        value = price * self.lot_size
        costs = calculate_trade_cost('sell', value)
        self.current_fund += value - costs['transaction_cost']
        self.position -= self.lot_size
        self.latest_action = 'sell'
        self.trades.append({
            'action': 'sell',
            'price': price,
            'shares': self.lot_size,
            'value': value,
            **costs,
        })
