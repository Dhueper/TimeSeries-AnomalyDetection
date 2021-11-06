"""
Example of using Keras to implement a 1D convolutional neural network (CNN) for timeseries prediction.
"""

import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt


def make_timeseries_regressor(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):
    """:Return: a Keras Model for predicting the next value in a timeseries given a fixed-size lookback window of previous values.
    The model can handle multiple input timeseries (`nb_input_series`) and multiple prediction targets (`nb_outputs`).
    :param int window_size: The number of previous timeseries values to use as input features.  Also called lag or lookback.
    :param int nb_input_series: The number of input timeseries; 1 for a single timeseries.
      The `X` input to ``fit()`` should be an array of shape ``(n_instances, window_size, nb_input_series)``; each instance is
      a 2D array of shape ``(window_size, nb_input_series)``.  For example, for `window_size` = 3 and `nb_input_series` = 1 (a
      single timeseries), one instance could be ``[[0], [1], [2]]``. See ``make_timeseries_instances()``.
    :param int nb_outputs: The output dimension, often equal to the number of inputs.
      For each input instance (array with shape ``(window_size, nb_input_series)``), the output is a vector of size `nb_outputs`,
      usually the value(s) predicted to come after the last value in that input instance, i.e., the next value
      in the sequence. The `y` input to ``fit()`` should be an array of shape ``(n_instances, nb_outputs)``.
    :param int filter_length: the size (along the `window_size` dimension) of the sliding window that gets convolved with
      each position along each instance. The difference between 1D and 2D convolution is that a 1D filter's "height" is fixed
      to the number of input timeseries (its "width" being `filter_length`), and it can only slide along the window
      dimension.  This is useful as generally the input timeseries have no spatial/ordinal relationship, so it's not
      meaningful to look for patterns that are invariant with respect to subsets of the timeseries.
    :param int nb_filter: The number of different filters to learn (roughly, input patterns to recognize).
    """
    model = keras.models.Sequential((
        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.
        keras.layers.Conv1D(filters=nb_filter, kernel_size=filter_length*2, activation='linear', input_shape=(window_size, nb_input_series)),
        keras.layers.MaxPooling1D(),     # Downsample the output of convolution by 2X.
        keras.layers.Conv1D(filters=nb_filter, kernel_size=filter_length, activation='linear'),
        keras.layers.AveragePooling1D(pool_size = 2),
        keras.layers.Flatten(),
        keras.layers.Dense(window_size, activation=None),     # For binary classification, change the activation to 'sigmoid'
    ))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # To perform (binary) classification instead:
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model


def make_timeseries_instances(timeseries, smoothed_timeseries, window_size):
    """Make input features and prediction targets from a `timeseries` for use in machine learning.
    :return: A tuple of `(X, y, q)`.  `X` are the inputs to a predictor, a 3D ndarray with shape
      ``(timeseries.shape[0] - window_size, window_size, timeseries.shape[1] or 1)``.  For each row of `X`, the
      corresponding row of `y` is the next value in the timeseries.  The `q` or query is the last instance, what you would use
      to predict a hypothetical next (unprovided) value in the `timeseries`.
    :param ndarray timeseries: Either a simple vector, or a matrix of shape ``(timestep, series_num)``, i.e., time is axis 0 (the
      row) and the series is axis 1 (the column).
    :param int window_size: The number of samples to use as input prediction features (also called the lag or lookback).
    """
    timeseries = np.asarray(timeseries)
    smoothed_timeseries = np.asarray(smoothed_timeseries)
    assert 0 < window_size < timeseries.shape[0]
    X = np.atleast_3d(np.array([timeseries[start*window_size:(start+1)*window_size] for start in range(0, timeseries.shape[0] // window_size)]))
    y = np.atleast_3d(np.array([smoothed_timeseries[start*window_size:(start+1)*window_size] for start in range(0, smoothed_timeseries.shape[0] // window_size)]))
    # X = np.atleast_3d(np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    # y = np.atleast_3d(np.array([smoothed_timeseries[start:start + window_size] for start in range(0, smoothed_timeseries.shape[0] - window_size)]))
    return X, y


def evaluate_timeseries(timeseries, smoothed_timeseries, ts_test, sts_test, window_size):
    """Create a 1D CNN regressor to predict the next value in a `timeseries` using the preceding `window_size` elements
    as input features and evaluate its performance.
    :param ndarray timeseries: Timeseries data with time increasing down the rows (the leading dimension/axis).
    :param int window_size: The number of previous timeseries values to use to predict the next.
    """
    filter_length = 3  #Stencil 
    nb_filter = 100      #Number of features to be learned 

    timeseries = adjust_shape(timeseries)      # Convert 1D vectors to 2D column vectors
    smoothed_timeseries = adjust_shape(smoothed_timeseries)   
    ts_test = adjust_shape(ts_test)
    sts_test = adjust_shape(sts_test)   

    nb_samples, nb_series = timeseries.shape
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
    model = make_timeseries_regressor(window_size=window_size, filter_length=filter_length, nb_input_series=nb_series, nb_outputs=nb_series, nb_filter=nb_filter)
    print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))

    X, y = make_timeseries_instances(timeseries, smoothed_timeseries, window_size)
    print('\n\nInput features:', X, '\n\nOutput labels:', y)

    X_test, y_test = make_timeseries_instances(ts_test, sts_test, window_size)

    model.fit(X, y, epochs=1000, batch_size=2, validation_data=(X_test, y_test))

    return model

def adjust_shape(X):
  """ Reshape 1D array to 2D array
      Input: X(numpy.array), 1D
      Returns: 2D numpy.array
  """
  X = np.atleast_2d(X)
  if X.shape[0] == 1:
    X = X.T 
  return X 

def create_ts_dataset(N):
  """ Creates N random functions to train the CNN filter
      Input: N(integer), number of functions to be created
      Returns: timeseries (numpy.array), smoothed_timeseries (numpy.array), ts_test (numpy.array), sts_test (numpy.array)
  """
  t = np.linspace(0,1,1000)
  t_test = np.linspace(0,1,101)

  timeseries =[] 
  smoothed_timeseries =[] 
  ts_test =[] 
  sts_test =[] 

  for i in range(0,N):
    X = np.zeros(len(t))
    Y = np.zeros(len(t))
    X_test = np.zeros(len(t_test))
    Y_test = np.zeros(len(t_test))
    q = np.random.randint(0,9,size=5)
    for r in q:
      w = np.random.rand(1)
      t0 = np.random.rand(1)
      signo = np.random.randint(2, size=1)
      if signo == 0:
        signo = -1
      X = X + signo*w*(t-t0)**r
      X_test = X_test + signo*w*(t_test-t0)**r
    Y[:] = X[:]  
    Y_test[:] = X_test[:]  
    s = np.random.randint(0,200,5)
    for r in s:
      w = np.random.rand(1)
      phi = np.random.rand(1) * np.pi
      X = X + w*np.sin(2*np.pi*r * t - phi)
      X_test = X_test + w*np.sin(2*np.pi*r * t_test - phi)

    # timeseries = np.concatenate(timeseries,X)
    # smoothed_timeseries = np.concatenate(smoothed_timeseries,Y)
    # ts_test = np.concatenate(ts_test, X_test)
    # sts_test = np.concatenate(sts_test, Y_test)
    timeseries = timeseries + list(X)
    smoothed_timeseries = smoothed_timeseries + list(Y)
    ts_test = ts_test + list(X_test)
    sts_test = sts_test + list(Y_test)

  timeseries = np.array(timeseries)
  smoothed_timeseries = np.array(smoothed_timeseries)
  ts_test = np.array(ts_test)
  sts_test = np.array(sts_test)

  return timeseries, smoothed_timeseries, ts_test, sts_test

def main(X,Y,X_test,Y_test):
    """Prepare input data, build model, evaluate."""
    np.set_printoptions(threshold=25)
    ts_length = 1000
    window_size = 50

    # print('\nSimple single timeseries vector prediction')
    # t = np.linspace(0,1,ts_length)                  
    # X = t + np.sin(2*np.pi*50 * t)     # The timeseries f(t) = t + sin(100*pi*t)
    # Y = t                              #Smoothed timeseries 
    # #Test data
    # t_test = np.linspace(0,1,101)
    # X_test =  t_test + np.sin(2*np.pi*50 * t_test)
    # Y_test = t_test

    # plt.figure()
    # plt.plot(t,X,'b')
    # plt.plot(t,Y,'r')
    # plt.title('Timeseries')
    # plt.xlabel('t')
    # plt.ylabel('X(t)') 

    model = evaluate_timeseries(X, Y, X_test, Y_test, window_size)

    model.summary()

    #Test the model 
    t = np.linspace(0,1,ts_length)
    X2 = t - t**2. + np.sin(2*np.pi*50 * t) 
    Y2 = t - t**2.

    #Reshape arrays 
    X2 = adjust_shape(X2)
    Y2 = adjust_shape(Y2)

    X2, real = make_timeseries_instances(X2, Y2, window_size)

    pred = model.predict(X2)

    pred1 = pred.reshape(-1)
    real = real.reshape(-1)

    pred2 = np.zeros(len(pred1))
    pred2[:] = pred1[:]    
  
    for k in range(0,100):
      for i in range(1,len(pred1)-1):
        pred2[i] = (pred2[i-1] + 2*pred2[i] + pred2[i+1]) / 4 

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(real-pred1)
    plt.subplot(2,1,2)
    plt.plot(real-pred2)
    plt.title('Error')

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(real)
    plt.plot(pred1)
    plt.subplot(2,1,2)
    plt.plot(real)
    plt.plot(pred2)
    plt.xlabel('t')
    plt.ylabel('Y(t)')
    plt.title('Smoothed timeseries')
    plt.legend(['Real', 'Estimated'])

    # print('\nMultiple-input, multiple-output prediction')
    # timeseries = np.array([np.arange(ts_length), -np.arange(ts_length)]).T      # The timeseries f(t) = [t, -t]
    # evaluate_timeseries(timeseries, window_size)


if __name__ == '__main__':

    N = 5
    X, Y, X_test, Y_test = create_ts_dataset(N)

    main(X,Y,X_test,Y_test)

    t = np.linspace(0,1,1000*N)
    t_test = np.linspace(0,1,101*N)
    plt.figure()
    plt.plot(t,X)
    plt.plot(t,Y)
    plt.plot(t_test,X_test)
    plt.plot(t_test,Y_test)
    plt.show()
