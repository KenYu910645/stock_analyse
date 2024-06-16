'''
minion.py

Get Rid of confidence because it influence my judgement on performance

The longer it trains, the worse the validation loss gets -> Maybe 5 days is too little

Try longer days:
    The longer it gets, The easier it overfit
    despite the trainig loss can goes down very quickly


'''
import argparse
import os
import sys
from datetime import datetime
import pandas as pd
import twstock
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import matplotlib
import mplfinance as mpf

# Local import
from model import Minion
from config import cfg

#######################
### Global variable ###
#######################
# Starting date for the data fetch
START_YEAR = 2020
START_MONTH = 5
# Hyperparameter
BATCH_SIZE = 32
NUM_EPOCHS = 50
INPUT_DAYS = 5 # 10 # 5
NUM_FEATURE = 5  # 'Open', 'High', 'Low', 'Close', 'Capacity'

def draw_kbar(df, out_fn):
    '''
    draw_kbar
    '''
    # Verify that the DataFrame index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expect data.index as DatetimeIndex")

    # Rename 'Turnover' to 'Volume' for mplfinance
    df.rename(columns={'Turnover': 'Volume'}, inplace=True)

    # Customizing the appearance: Set up colors and style
    mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

    # Prepare plotting parameters
    kwargs = dict(type='candle', mav=(5, 20, 60), volume=True,
                figratio=(10, 8), figscale=0.75,
                title=out_fn, style=s,
                savefig=f'plot/{out_fn}')

    # Plot the K-bar (candlestick) chart
    mpf.plot(df, **kwargs)

def fetch_stock_data(stock_tar):
    """
    Fetch daily trading data for a given stock from a specified start date to the present.

    Args:
        stock_tar (str): The stock code to query.

    Returns:
        pd.DataFrame: A DataFrame containing the stock's trading data.
    """

    # Create a Stock object
    stock = twstock.Stock(stock_tar)
    # Fetch daily trading data from the start date to the present
    target_price = stock.fetch_from(START_YEAR, START_MONTH)

    # Set headers for the collected data
    name_attribute = [
        'Date',  # 日期
        'Capacity',  # 總成交股數
        'Turnover',  # 總成交金額(Volume)
        'Open',
        'High',
        'Low',
        'Close',
        'Change',  # 漲跌幅
        'Transaction'  # 成交量
    ]

    df = pd.DataFrame(columns=name_attribute,
                      data=target_price)

    return df


def preprocess(df):
    '''
    preprocess
    '''
    X = []
    y = []

    # Normalize Capacity(總成交股數)
    capacity_norm = (df['Capacity'] - df['Capacity'].min()) / \
                    (df['Capacity'].max() - df['Capacity'].min())

    # Calculate price offsets and discard the first day
    for i in range(1, len(df) - INPUT_DAYS):

        # Calculate offsets for the input (5 days)
        X_price_diff = []
        for j in range(INPUT_DAYS):
            price_diff = df.iloc[i + j][['Open', 'High', 'Low', 'Close']].values - \
                         df.iloc[i + j - 1]['Close']

            # Add capacity to X
            X_price_diff.append(np.append(price_diff,
                                          capacity_norm.iloc[i + j]))

        # Calculate offsets for the output (next day)
        y_price_diff = df.iloc[i + INPUT_DAYS][['Open', 'High', 'Low', 'Close']].values - \
                       df.iloc[i + INPUT_DAYS - 1]['Close']

        X.append(X_price_diff)
        y.append(y_price_diff)

    X = np.array(X)
    y = np.array(y)

    # # Add a column for confidence
    # confidence = np.zeros((y.shape[0], 1))
    # y = np.hstack((y, confidence))

    return X.astype(np.float32), y.astype(np.float32)


if __name__ == "__main__":
    # Get use input arguments
    # parser = argparse.ArgumentParser(
    #          description="Fetch and analyse stock trading data.")
    # parser.add_argument('stock_tar', type=str,
    #                     help="The stock code to query, e.g.0050")
    # args = parser.parse_args()

    all_X = []
    all_y = []
    for stock_tar in cfg.stock_list:

        # Get output file name
        start_time = f"{START_YEAR}{str(START_MONTH).zfill(2)}"
        end_time = datetime.now().strftime("%Y%m")
        fn_out = f'./data/{stock_tar}_{start_time}_to_{end_time}.csv'

        # Get stock price by webcrawler
        if os.path.exists(fn_out):
            print(f"Stock data {fn_out} already exists."
                  "Skipping data fetch.")

            # Get data/*.csv
            df_stock = pd.read_csv(fn_out, parse_dates=['Date'])

        else:
            # Web Crawling data from TWSE
            try:
                df_stock = fetch_stock_data(stock_tar)
            except KeyError:
                print(f"stock index: {stock_tar} doesn't exist.")
                sys.exit(1)

            # Check if 'Date' column exists
            if 'Date' not in df_stock.columns:
                # Copy 'Date' from backup_df to stock_target
                backup_fn = f'./data/0050_{start_time}_to_{end_time}.csv'
                if os.path.exists(backup_fn):
                    # Read backup_df
                    backup_df = pd.read_csv(backup_fn, parse_dates=['Date'])
                    backup_df.set_index('Date', inplace=True)
                    # Copy 'Date'
                    df_stock['Date'] = backup_df['Date']
                else:
                    print(f"Backup file {backup_fn} not found.")
                    sys.exit(1)

            # Save DataFrame as CSV
            df_stock.to_csv(fn_out, index=False)
            print(f"Data fetched and saved to {fn_out}.")
            print(f"Fail to fetch stock target: {stock_tar}")

        #######################
        ### Draw K bar plot ###
        #######################
        if cfg.is_plot:
            # Normalize date format
            df_stock['Date'] = pd.to_datetime(df_stock['Date'])
            # Set 'Date' as index
            df_stock.set_index('Date', inplace=True)
            # Draw K-bar chart
            draw_kbar(df_stock,
                      f"{stock_tar}_{start_time}_to_{end_time}")

        #####################
        ### Preprocessing ###
        #####################
        X, y = preprocess(df_stock)
        all_X.append(X)
        all_y.append(y)

    ######################
    ### Training model ###
    ######################
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # Split data into training(80%) and validation(20%) sets
    X_train, X_val, y_train, y_val = train_test_split(all_X, all_y,
                                                    test_size=0.2,
                                                    random_state=42)

    # Create DataLoader for training and validation sets
    train_dataset = TensorDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),
                                  torch.from_numpy(y_val))
    print(f"Number of training data: {len(train_dataset)}")
    print(f"Number of validation data: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

    # Define NN model
    model = Minion(INPUT_DAYS*NUM_FEATURE)

    # Define loss function
    criterion = nn.MSELoss() # mean squared error

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(NUM_EPOCHS):

        # Train model
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # Valid model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS},"
            f"Training Loss: {train_loss:.4f},"
            f"Validation Loss: {val_loss:.4f}")

    ########################
    ### Model Evaluation ###
    ########################
    # model.eval()
    # with torch.no_grad():
    #     for X_batch, y_batch in val_loader:
    #         outputs = model(X_batch)
    #         print('Predicted:', outputs)
    #         print('Actual:', y_batch)
    #         break  # Display the first batch only for brevity
