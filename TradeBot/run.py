import time
import math
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

import alpaca_trade_api as tradeapi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from threading import Timer
import pandas_datareader as web
import pandas as pd
import numpy as np
import pprint, pathlib, operator
import seaborn as sns
import IPython
import IPython.display

from datetime import datetime, timedelta
from TradeBot.TradeBot import TradeBot
from TradeBot.trades import Trade

#if datetime.today().weekday() == 1 or datetime.today().weekday() == 7:
#    raise TypeError("Stock Market Closed Today")

trading_robot = TradeBot()
trading_robot_portfolio = trading_robot.create_portfolio()

multi_position = [
    {
        'asset_type': 'equity',
        'quantity': 0,
        'purchase_price': 0.00,
        'symbol': 'TSLA',
        'purchase_date': '2020-01-31'
    },
    {
        'asset_type': 'equity',
        'quantity': 0,
        'purchase_price': 0.00,
        'symbol': 'SQ',
        'purchase_date': '2020-01-31'
    }
]

new_positions = trading_robot.portfolio.add_positions(positions=multi_position)

trading_robot.portfolio.add_position(symbol='MSFT', quantity=0, purchase_price=0.00, asset_type='equity', purchase_date='2020-04-01')

trading_robot.portfolio.add_position(symbol='WMT', quantity=0, purchase_price=0.00, asset_type='equity', purchase_date='2020-04-01')

trading_robot.portfolio.add_position(symbol='GOOGL', quantity=0, purchase_price=0.00, asset_type='equity', purchase_date='2020-04-01')

if trading_robot.pre_market_open:
    print("Pre market is open")
else:
    print("Pre market isn't open")

if trading_robot.regular_market_open:
    print("Regular market is open")
else:
    print("Regular market isn't open")

if trading_robot.post_market_open:
    print("Post market is open")
else:
    print("Post market isn't open")

if trading_robot.regular_market_open or trading_robot.post_market_open or trading_robot.pre_market_open:
    current_quotes = trading_robot.grab_current_quotes()
    pprint.pprint(current_quotes)

end_date = datetime.today()
start_date = end_date - timedelta(days=14600)

historical_prices = trading_robot.grab_historical_prices(start=start_date, end=end_date, bar='1d')
stock_frame = trading_robot.create_stock_frame(historical_prices['aggregated'])
#pprint.pprint(stock_frame.frame.head(n=20))

'''new_trade = trading_robot.create_trade(
    trade_id='long_msft',
    enter_or_exit='enter',
    long_or_short='long',
    order_type='lmt',
    price=150.00
)
new_trade.instrument(
    symbol='MSFT',
    quantity=1
)

new_trade.add_stop_loss(
    stop_size=.10,
    percentage=False
)

new_trade.execute_trade()'''

#pprint.pprint(new_trade.order)

plt.figure(figsize=(16,8))
plt.title('Close Price History')

for symbol in trading_robot.portfolio.positions.keys():
    plt.plot(stock_frame.frame.loc[symbol]['close'], label=symbol)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
#plt.show()

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

def plot(self, model=None, plot_col=1, max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(3, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [d]')

WindowGenerator.plot = plot

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(5)

def warmup(self, inputs):
  # inputs.shape => (batch, time, features)
  # x.shape => (batch, lstm_units)
  x, *state = self.lstm_rnn(inputs)

  # predictions.shape => (batch, features)
  prediction = self.dense(x)
  return prediction, state

FeedBack.warmup = warmup

def call(self, inputs, training=None):
  # Use a TensorArray to capture dynamically unrolled outputs.
  predictions = []
  # Initialize the lstm state
  prediction, state = self.warmup(inputs)

  # Insert the first prediction
  predictions.append(prediction)

  # Run the rest of the prediction steps
  for n in range(1, self.out_steps):
    # Use the last prediction as input.
    x = prediction
    # Execute one lstm step.
    x, state = self.lstm_cell(x, states=state,
                              training=training)
    # Convert the lstm output to a prediction.
    prediction = self.dense(x)
    # Add the prediction to the output
    predictions.append(prediction)

  # predictions.shape => (time, batch, features)
  predictions = tf.stack(predictions)
  # predictions.shape => (batch, time, features)
  predictions = tf.transpose(predictions, [1, 0, 2])
  return predictions

FeedBack.call = call

class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs


MAX_EPOCHS = 70

def compile_and_fit(model, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val, callbacks=early_stopping)
  return history

OUT_STEPS = 25
feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

def run():
    for Symbol in trading_robot.portfolio.positions.keys():
        if trading_robot.portfolio.positions[Symbol]['selling_date'] != None:
            if trading_robot.portfolio.positions[Symbol]['selling_date'] == (datetime.now()).strftime("%Y-%m-%d"):
                new_trade = trading_robot.create_trade(
                    trade_id='fooo',
                    enter_or_exit='exit',
                    long_or_short='long',
                    order_type='mkt',
                    price=trading_robot.current_prices[Symbol]
                )
                new_trade.instrument(
                    symbol=Symbol,
                    quantity=trading_robot.portfolio.positions[Symbol]['quantity']
                )

                new_trade.execute_trade()

                trading_robot.portfolio.add_position(symbol=Symbol, quantity=0, purchase_price=0.00, asset_type='equity', purchase_date='2020-04-01')

        df = stock_frame.frame.loc[Symbol].copy()

        n = len(df)
        train_df = df[0:int(n*0.7)]
        val_df = df[int(n*0.7):int(n*0.9)]
        test_df = df[int(n*0.9):]

        num_features = df.shape[1]

        min_max_scaler = MinMaxScaler(feature_range= (0, 1))

        x = train_df.values
        x_scaled = min_max_scaler.fit_transform(x)
        train_df = pd.DataFrame(x_scaled)

        x = val_df.values
        x_scaled = min_max_scaler.fit_transform(x)
        val_df = pd.DataFrame(x_scaled)

        x = test_df.values
        x_scaled = min_max_scaler.fit_transform(x)
        test_df = pd.DataFrame(x_scaled)

        multi_val_performance = {}
        multi_performance = {}

        multi_window = WindowGenerator(input_width=25, label_width=OUT_STEPS, shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)

        """prediction, state = feedback_model.warmup(multi_window.example[0])

        OUT_STEPS = 25
        feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
        history = compile_and_fit(feedback_model, multi_window)

        multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
        multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
        multi_window.plot(feedback_model)

        plt.title(symbol)
        plt.show()

        IPython.display.clear_output()"""

        """last_baseline = MultiStepLastBaseline()
        last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                            metrics=[tf.metrics.MeanAbsoluteError()])


        multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
        multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)
        multi_window.plot(last_baseline)

        print(last_baseline.summary())

        plt.title(symbol)
        plt.show()"""
    
        repeat_baseline = RepeatBaseline()
        repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])


        multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
        multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
        multi_window.plot(repeat_baseline)

        print(repeat_baseline.summary())

        plt.title(Symbol)
        plt.show()

        """multi_linear_model = tf.keras.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        history = compile_and_fit(multi_linear_model, multi_window)

        IPython.display.clear_output()
        multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
        multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
        multi_window.plot(multi_linear_model)

        print(multi_linear_model.summary())

        plt.title(symbol)
        plt.show()

        CONV_WIDTH = 3
        multi_conv_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        history = compile_and_fit(multi_conv_model, multi_window)

        IPython.display.clear_output()

        multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
        multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
        multi_window.plot(multi_conv_model)

        print(multi_conv_model.summary())

        plt.title(symbol)
        plt.show()

        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units]
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                kernel_initializer=tf.initializers.zeros),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        history = compile_and_fit(multi_lstm_model, multi_window)

        IPython.display.clear_output()

        multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
        multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
        multi_window.plot(multi_lstm_model)"""

        new_df = stock_frame.frame.copy()
    
        symbols = []

        for key in trading_robot.portfolio.positions.keys():
            symbols.append(key)

        symbols.remove(Symbol)

        print(symbols)

        new_df = new_df.drop(symbols)
        print(new_df.head(n=20))
        print(new_df.tail(n=20))
        new_df = min_max_scaler.fit_transform(new_df)
        predictions = repeat_baseline.predict(new_df[-25:])
        predictions = min_max_scaler.inverse_transform(predictions)
        predictions = [row[1] for row in predictions]
        print(predictions)

        max_index_col = np.argmax(predictions, axis=0)

        if np.amax(predictions) > trading_robot.current_prices[Symbol] and trading_robot.portfolio.positions[Symbol]['quantity'] == 0:
            trading_robot.portfolio.add_position(symbol=Symbol, quantity=math.ceil(200/trading_robot.current_prices[Symbol]), purchase_price=(math.ceil(200/trading_robot.current_prices[Symbol])*trading_robot.current_prices[Symbol]), asset_type='equity', purchase_date=(datetime.now()).strftime("%Y-%m-%d"), selling_date=(datetime.now() + timedelta(days=(max_index_col + 1).astype(float))).strftime("%Y-%m-%d"))
            new_trade = trading_robot.create_trade(
                trade_id='foo',
                enter_or_exit='enter',
                long_or_short='long',
                order_type='mkt',
                price=trading_robot.current_prices[Symbol]
            )
            new_trade.instrument(
                symbol=Symbol,
                quantity=math.ceil(200/trading_robot.current_prices[Symbol])
            )

            new_trade.execute_trade()

        """x = np.arange(len(multi_performance))
        width = 0.3


        metric_name = 'mean_absolute_error'
        metric_index = repeat_baseline.metrics_names.index('mean_absolute_error')
        val_mae = [v[metric_index] for v in multi_val_performance.values()]
        test_mae = [v[metric_index] for v in multi_performance.values()]

        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=multi_performance.keys(),
                rotation=45)
        plt.ylabel(f'MAE (average over all times and outputs)')
        _ = plt.legend()

        plt.show()"""

while True:
    run()

    time.sleep(86400)
