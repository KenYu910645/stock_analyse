'''
train.py

Train the stock analysis neural network from cached or downloaded stock data.

Get Rid of confidence because it influence my judgement on performance

The longer it trains, the worse the validation loss gets -> Maybe 5 days is too little

Try longer days:
    The longer it gets, The easier it overfit
    despite the trainig loss can goes down very quickly
'''
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from fcn import FCN
from stockDownload import load_or_download_stock


def preprocess(df):
    '''
    preprocess
    '''
    X = []
    y = []

    # Normalize Capacity
    capacity_norm = (df['Capacity'] - df['Capacity'].min()) / \
                    (df['Capacity'].max() - df['Capacity'].min())

    # Calculate price offsets and discard the first day
    for i in range(1, len(df) - cfg.input_days):

        # Calculate offsets for the input
        X_price_diff = []
        for j in range(cfg.input_days):
            price_diff = df.iloc[i + j][['Open', 'High', 'Low', 'Close']].values - \
                         df.iloc[i + j - 1]['Close']

            # Add capacity to X
            X_price_diff.append(np.append(price_diff,
                                          capacity_norm.iloc[i + j]))

        # Calculate offsets for the output
        y_price_diff = df.iloc[i + cfg.input_days][['Open', 'High', 'Low', 'Close']].values - \
                       df.iloc[i + cfg.input_days - 1]['Close']

        X.append(X_price_diff)
        y.append(y_price_diff)

    X = np.array(X)
    y = np.array(y)

    # # Add a column for confidence
    # confidence = np.zeros((y.shape[0], 1))
    # y = np.hstack((y, confidence))

    return X.astype(np.float32), y.astype(np.float32)


if __name__ == '__main__':
    all_X = []
    all_y = []
    for stock_tar in cfg.stock_list:
        try:
            df_stock = load_or_download_stock(stock_tar, is_plot=cfg.is_plot)
        except KeyError:
            print(f"stock index: {stock_tar} doesn't exist.")
            sys.exit(1)

        X, y = preprocess(df_stock)
        all_X.append(X)
        all_y.append(y)

    writer = SummaryWriter(log_dir=f'runs/{cfg.exp_name}')

    ######################
    ### Training model ###
    ######################
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # Split data into training(80%) and validation(20%) sets
    X_train, X_val, y_train, y_val = train_test_split(all_X, all_y,
                                                      test_size=0.2,
                                                      random_state=1995)

    # Create DataLoader for training and validation sets
    train_dataset = TensorDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val),
                                torch.from_numpy(y_val))
    print(f'Number of training data: {len(train_dataset)}')
    print(f'Number of validation data: {len(val_dataset)}')
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False)

    # Define NN model
    model = FCN(cfg.input_days * cfg.num_feature)

    # Define loss function
    criterion = nn.MSELoss()  # mean squared error

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Training loop
    for epoch in range(cfg.num_epochs):

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

        # Log the losses
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        print(f'Epoch {epoch+1}/{cfg.num_epochs}, '
              f'Training Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}')

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

    writer.close()
