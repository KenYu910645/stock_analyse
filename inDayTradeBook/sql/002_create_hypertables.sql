SELECT create_hypertable(
    'raw_market_events',
    'received_at',
    if_not_exists => TRUE
);

SELECT create_hypertable(
    'trades',
    'ts',
    if_not_exists => TRUE
);

SELECT create_hypertable(
    'orderbook_5',
    'ts',
    if_not_exists => TRUE
);
