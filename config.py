class Config:
    '''
    Config
    '''
    def __init__(self):
        self.stock_list = []
        self.is_plot = False

cfg = Config()

cfg.stock_list =  [
    '2392',
    '3481',
    '6770',
    '2412',
    '2308',
    '2049',
    '2634',
    '2345',
    '2395',
    '2317',
    '2454',
]

cfg.is_plot = False # False  # output the candlestick chat of stock prices
