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
        keras.layers.Conv1D(filters=nb_filter, kernel_size=filter_length, activation='linear', input_shape=(window_size, nb_input_series)),
        keras.layers.MaxPooling1D(),     # Downsample the output of convolution by 2X.
        # keras.layers.Conv1D(filters=nb_filter, kernel_size=filter_length, activation='relu'),
        # keras.layers.MaxPooling1D(),
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


def evaluate_timeseries(timeseries, smoothed_timeseries, window_size):
    """Create a 1D CNN regressor to predict the next value in a `timeseries` using the preceding `window_size` elements
    as input features and evaluate its performance.
    :param ndarray timeseries: Timeseries data with time increasing down the rows (the leading dimension/axis).
    :param int window_size: The number of previous timeseries values to use to predict the next.
    """
    filter_length = 3
    nb_filter = 1
    timeseries = np.atleast_2d(timeseries)
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T       # Convert 1D vectors to 2D column vectors
    smoothed_timeseries = np.atleast_2d(smoothed_timeseries)
    if smoothed_timeseries.shape[0] == 1:
        smoothed_timeseries = smoothed_timeseries.T       # Convert 1D vectors to 2D column vectors

    nb_samples, nb_series = timeseries.shape
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
    model = make_timeseries_regressor(window_size=window_size, filter_length=filter_length, nb_input_series=nb_series, nb_outputs=nb_series, nb_filter=nb_filter)
    print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
    model.summary()

    X, y = make_timeseries_instances(timeseries, smoothed_timeseries, window_size)

    print('\n\nInput features:', X, '\n\nOutput labels:', y)
    test_size = int(0.01 * nb_samples)           # In real life you'd want to use 0.2 - 0.5
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    model.fit(X_train, y_train, epochs=500, batch_size=2, validation_data=(X_test, y_test))

    return model

    # pred = model.predict(X_test)

    # real = y_test.reshape(-1)
    # pred = pred.reshape(-1)

    # plt.figure()
    # plt.plot(real-pred)
    # plt.title('Error')

    # plt.figure()
    # plt.plot(real)
    # plt.plot(pred)
    # plt.xlabel('t')
    # plt.ylabel('Y(t)')
    # plt.title('Smoothed timeseries')
    # plt.legend(['Real', 'Estimated'])
    # plt.show()



def main():
    """Prepare input data, build model, evaluate."""
    np.set_printoptions(threshold=25)
    ts_length = 1000
    window_size = 50

    print('\nSimple single timeseries vector prediction')
    t = np.linspace(0,1,ts_length)                  
    X = t + np.sin(2*np.pi*50 * t)     # The timeseries f(t) = t + sin(100*pi*t)
    Y = t                              #Smoothed timeseries 

    plt.figure()
    plt.plot(t,X,'b')
    plt.plot(t,Y,'r')
    plt.title('Timeseries')
    plt.xlabel('t')
    plt.ylabel('X(t)') 

    model = evaluate_timeseries(X, Y, window_size)

    #Test the model 

    X_test = t**2. + np.sin(2*np.pi*50 * t) 
    Y_test = t**2.

    X_test = np.atleast_2d(X_test)
    if X_test.shape[0] == 1:
        X_test = X_test.T       # Convert 1D vectors to 2D column vectors
    Y_test = np.atleast_2d(Y_test)
    if Y_test.shape[0] == 1:
        Y_test = Y_test.T 

    X_test, real = make_timeseries_instances(X_test, Y_test, window_size)

    pred = model.predict(X_test)

    pred1 = pred.reshape(-1)
    real = real.reshape(-1)

    plt.figure()
    plt.plot(real-pred1)
    plt.title('Error')

    plt.figure()
    plt.plot(real)
    plt.plot(pred1)
    plt.xlabel('t')
    plt.ylabel('Y(t)')
    plt.title('Smoothed timeseries')
    plt.legend(['Real', 'Estimated'])
    plt.show()

    # for i in range(0,10):
    #   pred = pred.reshape(pred.shape[0],pred.shape[1],1)
    #   pred = model.predict(pred)

    # pred2 = pred.reshape(-1)

    # plt.figure()
    # plt.plot(real-pred2)
    # plt.title('Error')

    # plt.figure()
    # plt.plot(real)
    # plt.plot(pred2)
    # plt.xlabel('t')
    # plt.ylabel('Y(t)')
    # plt.title('Smoothed timeseries')
    # plt.legend(['Real', 'Estimated'])
    # plt.show()

    # print('\nMultiple-input, multiple-output prediction')
    # timeseries = np.array([np.arange(ts_length), -np.arange(ts_length)]).T      # The timeseries f(t) = [t, -t]
    # evaluate_timeseries(timeseries, window_size)


if __name__ == '__main__':
    main()