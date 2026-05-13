class Config:
    '''
    Config
    '''
    def __init__(self):
        self.stock_list =  [
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
        self.download_all_stocks = True
        self.throttle_min_seconds = 1
        self.throttle_max_seconds = 3
        self.max_retries = 3
        self.retry_backoff_seconds = 10
        self.is_plot = False
        self.learning_rate = 0.001
        self.batch_size = 32 # 32
        self.num_epochs = 50
        self.num_hidden_feature = 256
        self.input_days = 20 # 20 # 10 # 5
        self.num_feature = 5  # 'Open', 'High', 'Low', 'Close', 'Capacity'
        self.num_hidden_layer = 5

        self.exp_name = (
            f"in_day_{self.input_days}_num_feature_{self.num_feature}_"
            f"num_hidden_feature_{self.num_hidden_feature}_"
            f"num_hidden_layer_{self.num_hidden_layer}_"
            f"batch_size_{self.batch_size}_"
            f"lr_{self.learning_rate}"
        )

cfg = Config()
