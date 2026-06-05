'''
Taiwan stock transaction cost helpers.
'''
import math


TRANSACTION_FEE_RATE = 0.001425
SELL_TAX_RATE = 0.003


def calculate_trade_cost(action, trade_value):
    '''
    Return fee, tax, and total transaction cost for a buy or sell.
    '''
    value = float(trade_value)
    fee = math.floor(value * TRANSACTION_FEE_RATE)
    tax = value * SELL_TAX_RATE if action == 'sell' else 0

    return {
        'fee': fee,
        'tax': tax,
        'transaction_cost': fee + tax,
    }
