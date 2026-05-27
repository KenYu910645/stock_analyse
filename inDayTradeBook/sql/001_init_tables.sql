CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS raw_market_events (
    id BIGSERIAL NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    exchange_ts TIMESTAMPTZ,
    symbol TEXT NOT NULL,
    channel TEXT NOT NULL,
    event_type TEXT,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    ts TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price NUMERIC,
    size INTEGER,
    volume INTEGER,
    bid NUMERIC,
    ask NUMERIC,
    raw_event_id BIGINT,
    payload JSONB
);

CREATE TABLE IF NOT EXISTS orderbook_5 (
    ts TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    bid1_price NUMERIC,
    bid1_size INTEGER,
    bid2_price NUMERIC,
    bid2_size INTEGER,
    bid3_price NUMERIC,
    bid3_size INTEGER,
    bid4_price NUMERIC,
    bid4_size INTEGER,
    bid5_price NUMERIC,
    bid5_size INTEGER,
    ask1_price NUMERIC,
    ask1_size INTEGER,
    ask2_price NUMERIC,
    ask2_size INTEGER,
    ask3_price NUMERIC,
    ask3_size INTEGER,
    ask4_price NUMERIC,
    ask4_size INTEGER,
    ask5_price NUMERIC,
    ask5_size INTEGER,
    payload JSONB
);
