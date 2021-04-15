
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data_handler import DataHandler
import numpy as np
import os
import matplotlib.pyplot as plt

models_dir = 'models'

class ModelHandler:
    
    def __init__(self, model_id):

        self.model_id = model_id
        self.model_dir = os.path.join(models_dir, model_id)
        if os.path.exists(self.model_dir):
            self.model = tf.keras.models.load_model(self.model_dir)


    def make_model(self, window_size, input_size = 6 * 10, dropout=.2):

        inputs = tf.keras.Input(shape=(6,window_size))
        obj = tf.keras.layers.Flatten()(inputs)
        obj = tf.keras.layers.Dense(input_size * 10, activation = 'relu')(obj)
        obj = tf.keras.layers.Dropout(dropout)(obj)
        obj = tf.keras.layers.Dense(input_size * 2, activation = 'relu')(obj)
        obj = tf.keras.layers.Dropout(dropout)(obj)
        obj = tf.keras.layers.Dense(3, activation = 'softmax')(obj)

        self.model = tf.keras.Model(inputs, obj)
        self.model.summary()
        self.model.save(self.model_dir)


    def train(self, no_epochs, steps_per_epoch, data_handler, test_data_handler, batch_size, starting_learning_rate, thresholds=[-0.02, 0.02], random_split=0.):

        input_shape = self.model.layers[0].input_shape[0]
        window_size = input_shape[2]

        data_generator = data_handler.data_generator(
            batch_size = batch_size,
            window_size = window_size,
            random_split = random_split, 
            thresholds = thresholds
        )

        test_data_generator = test_data_handler.data_generator(
            batch_size = batch_size,
            window_size = window_size,
            random_split = random_split, 
            thresholds = thresholds
        )

        opt = tf.keras.optimizers.Adam(learning_rate=starting_learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=opt)
        self.model.fit(
            data_generator,
            epochs=no_epochs,
            callbacks=[tf.keras.callbacks.TensorBoard()],
            steps_per_epoch=steps_per_epoch
        )

        self.model.evaluate(
            test_data_generator,
            steps = steps_per_epoch,
            callbacks = [tf.keras.callbacks.TensorBoard()]
        )

        self.model.save(self.model_dir)


    def evaluate(self, test_data_handler, no_samples):

        input_shape = self.model.layers[0].input_shape[0]
        window_size = input_shape[2]

        raw_data = test_data_handler.get_samples(window_size, no_samples)

        raw_open_p = raw_data[0]
        raw_close_p = raw_data[1]
        raw_low_p = raw_data[2]
        raw_high_p = raw_data[3]
        raw_vol = raw_data[4]
        raw_trades = raw_data[5]

        predictions = []
        points_price = []

        for ind in range(window_size, no_samples):
            x = test_data_handler.preprocess_data(
                open_p = raw_open_p[ind-window_size : ind+1], 
                close_p = raw_close_p[ind-window_size : ind+1], 
                low_p = raw_low_p[ind-window_size : ind+1], 
                high_p = raw_high_p[ind-window_size : ind+1], 
                vol = raw_vol[ind-window_size : ind+1], 
                trades = raw_trades[ind-window_size : ind+1]
            )
            y = self.model.predict(np.expand_dims(x, 0))[0]
            predictions.append(y)
            points_price.append(raw_close_p[ind])
        
        print(predictions)


        max_pred = max(predictions)
        min_pred = min(predictions)

        colors = []
        for i in range(len(predictions)):
            color = np.zeros(4)   
            if abs(predictions[i]) > max(max_pred, abs(min_pred)) / 5:
                if predictions[i] > 0:
                    color[1] = 1
                    color[3] = (max_pred - predictions[i]) / max_pred
                else:
                    color[0] = 1
                    color[3] = (min_pred - predictions[i]) / min_pred
            colors.append(color)

        plt.figure()
        plt.plot(points_price, label = 'Closing price')
        plt.scatter(range(len(points_price)),points_price,c=colors)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    model_handler   = ModelHandler('Decider')
    model_handler.make_model(window_size = 20)