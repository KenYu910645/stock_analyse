CREATE INDEX IF NOT EXISTS idx_raw_market_events_symbol_time
ON raw_market_events (symbol, received_at DESC);

CREATE INDEX IF NOT EXISTS idx_raw_market_events_channel_time
ON raw_market_events (channel, received_at DESC);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_time
ON trades (symbol, ts DESC);

CREATE INDEX IF NOT EXISTS idx_orderbook_5_symbol_time
ON orderbook_5 (symbol, ts DESC);
