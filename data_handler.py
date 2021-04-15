import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import random

datasets_dir = 'coin_datasets'
LABELS = [
    'open_time',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'quote_asset_volume',
    'number_of_trades',
    'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume'
]

'DateTime format: YYYY-MM-DD HH:MM:SS'

batch_example_dir = 'example_input_batch'

def change_percent(origin, value):
    if origin == 0:
        print('Origin value is 0, could not compute change fpr {}'.format(value))
        return value
    else:
        return (value - origin) / origin


def plot(series):
        
    plt.figure()
    plt.plot(series)
    plt.show()


def comparative_plot(series_1,series_2):

    plt.figure()
    plt.plot(series_1, label = 'Original')
    plt.plot(series_2, label = 'Resampled')
    plt.legend()
    plt.show()


def get_subset(data, start_date_str, end_date_str, sample_period = 5):
        
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S')
    data['open_time'] = pd.to_datetime(data['open_time'], format='%Y-%m-%d %H:%M:%S')

    mask = (data['open_time'] >= start_date) & (data['open_time'] <= end_date)
    selected_data = data.loc[mask]
    
    print_data_info(selected_data)

    open_time = selected_data['open_time'].tolist()
    open_price = selected_data['open'].to_numpy()
    high_price = selected_data['high'].to_numpy()
    low_price = selected_data['low'].to_numpy()
    close_price = selected_data['close'].to_numpy()
    volume = selected_data['volume'].to_numpy()
    no_trades = selected_data['number_of_trades'].to_numpy()

    no_resampled_instances = int(len(selected_data) / sample_period)
        
    resampled_open_time = []
    resampled_open_price = np.zeros(no_resampled_instances)
    resampled_high_price = np.zeros(no_resampled_instances)
    resampled_low_price = np.zeros(no_resampled_instances)
    resampled_close_price = np.zeros(no_resampled_instances)
    resampled_volume = np.zeros(no_resampled_instances)
    resampled_no_trades = np.zeros(no_resampled_instances)

    for ind in range(no_resampled_instances):
        old_ind = ind * sample_period
        resampled_open_time.append(open_time[old_ind])
        resampled_open_price[ind] = open_price[old_ind]
        resampled_close_price[ind] = close_price[old_ind + sample_period - 1]
        resampled_high_price[ind] = np.amax(high_price[old_ind: old_ind + sample_period])
        resampled_low_price[ind] = np.amin(low_price[old_ind: old_ind + sample_period])
        resampled_volume[ind] = np.sum(volume[old_ind: old_ind + sample_period])
        resampled_no_trades[ind] = np.sum(no_trades[old_ind: old_ind + sample_period])

    #plot(resampled_close_price)
    
    def check_resample():
        'Checking correctness'

        samples = no_resampled_instances*sample_period

        check_open_price = np.zeros(len(open_price))
        check_high_price = np.zeros(len(open_price))
        check_close_price = np.zeros(len(open_price))
        check_low_price = np.zeros(len(open_price))

        if np.sum(volume[:samples]) != np.sum(resampled_volume):
            print('Difference between original ({}) and resampled ({})'.format(np.sum(volume),np.sum(resampled_volume)))
        else:
            print('Correctly resampled volume.')
        if np.sum(no_trades[:samples]) != np.sum(resampled_no_trades):
            print('Difference between original ({}) and resampled ({})'.format(np.sum(no_trades),np.sum(resampled_no_trades)))
        else:
            print('Correctly resampled no_trades.')
            
        for old_ind in range(no_resampled_instances*sample_period):
            ind = int(old_ind / sample_period)
            check_open_price[old_ind] = resampled_open_price[ind]
            check_close_price[old_ind] = resampled_close_price[ind]
            check_low_price[old_ind] = resampled_low_price[ind]
            check_high_price[old_ind] = resampled_high_price[ind]

        samples = no_resampled_instances*sample_period
        comparative_plot(open_price[:samples], check_open_price[:samples])
        comparative_plot(close_price[:samples], check_close_price[:samples])
        comparative_plot(close_price[:samples], check_close_price[:samples])
        comparative_plot(high_price[:samples], check_high_price[:samples])
        comparative_plot(low_price[:samples], check_low_price[:samples])

    data = [
        resampled_open_time, 
        resampled_open_price, 
        resampled_close_price, 
        resampled_low_price, 
        resampled_high_price, 
        resampled_volume, 
        resampled_no_trades
    ]
    return data
        

def print_data_info(data):
    '''
    Info about the current data.
    '''

    no_items = len(data)
    start_date = data['open_time'].iloc[0]
    end_date = data['open_time'].iloc[-1]
    print('Data has {} entries, from {} to {}.'.format(
        no_items,
        start_date, end_date
    ))


def plot_batch(x_batch,y_batch):
    '''
    Function that checks the selected indexes (relevant if not random sampled) to correctly be before critical moments.
    '''



    plt.figure()

    max_change = np.amax(self.change)
    min_change = np.amin(self.change)

    colors = np.zeros((len(self.resampled_close_price),4))
    for i in selected_inds:
        if self.change[i] > 0:
            colors[i][1] = 1
            colors[i][3] = (max_change - self.change[i]) / max_change 
        else:
            colors[i][0] = 1
            colors[i][3] = (min_change - self.change[i]) / min_change 
    
    for i in random_inds:
        if self.change[i] > 0:
            colors[i][1] = 1
            colors[i][3] = (max_change - self.change[i]) / max_change 
        else:
            colors[i][0] = 1
            colors[i][3] = (min_change - self.change[i]) / min_change 

    plt.plot(self.resampled_close_price, label = 'Closing price')
    plt.scatter(range(len(self.resampled_close_price)),self.resampled_close_price,c=colors)
    plt.legend()
    plt.show()




class DataHandler:

    def __init__(self, coin_1, coin_2):
        '''
        Function to read all available dta.
        '''

        self.coin_1 = coin_1
        self.coin_2 = coin_2

        'Conversion directly parquet-panads_dataframe does not handle the columns correctly. Conversion through csv as a solution.'

        parquet_file = os.path.join(datasets_dir, '{}-{}.parquet'.format(self.coin_1, self.coin_2))
        csv_file = os.path.join(datasets_dir, '{}-{}.csv'.format(self.coin_1, self.coin_2))
        
        if not os.path.exists(parquet_file):
            print('{}-{} dataset not found.'.format(self.coin_1, self.coin_2))
            return
        try:
            if not os.path.exists(csv_file):
                pd.read_parquet(parquet_file).to_csv(csv_file)
            self.data = pd.read_csv(csv_file)
        except:
            print('Error reading {}-{} dataset.'.format(self.coin_1, self.coin_2))
            return
        
        print('Succesfully read {}-{} dataset.'.format(self.coin_1, self.coin_2))
        print_data_info(self.data)
         

    def define_subset(self, start_date_str, end_date_str, sample_period = 5):
        
        subset = get_subset(self.data, start_date_str, end_date_str, sample_period = 5)
        
        self.resampled_open_time = subset[0]
        self.resampled_open_price = subset[1]
        self.resampled_close_price = subset[2]
        self.resampled_low_price = subset[3]
        self.resampled_high_price = subset[4]
        self.resampled_volume = subset[5]
        self.resampled_no_trades = subset[6]
        
    

    def compute_change(self, lookup_samples = 10, average_lookup_samples = False):
        '''
        Iterate through all the data and covert the data points to the change percentage compared to the following _lookup_samples_ values
        '''

        lookup_samples = lookup_samples

        n = len(self.resampled_close_price) - lookup_samples

        change = np.zeros(n)
        for i in range(n):
            if average_lookup_samples:
                change[i] = change_percent(self.resampled_close_price[i], np.mean(self.resampled_close_price[i : i + lookup_samples]))
            else:
                change[i] = change_percent(self.resampled_close_price[i], self.resampled_close_price[i + lookup_samples])

        self.change = change
        self.indexes = list(np.argsort(self.change))
            
        def check_change():
            
            plt.figure()
        
            max_change = np.amax(self.change)
            min_change = np.amin(self.change)

            colors = np.zeros((len(self.resampled_close_price),4))
            for i in range(len(self.change)):
                if self.change[i] > 0:
                    colors[i][1] = 1
                    colors[i][3] = (max_change - self.change[i]) / max_change 
                else:
                    colors[i][0] = 1
                    colors[i][3] = (min_change - self.change[i]) / min_change 
                  
            plt.plot(self.resampled_close_price, label = 'Closing price')
            plt.scatter(range(len(self.resampled_close_price)),self.resampled_close_price,c=colors)
            plt.legend()
            plt.show()
    

    def preprocess_data(self, open_p, close_p, low_p, high_p, vol, trades):
        
        x = np.zeros((6,len(open_p[:-1])))

        x[0] = change_percent(open_p[-1], open_p[:-1])
        x[1] = change_percent(close_p[-1], close_p[:-1])
        x[2] = change_percent(low_p[-1], low_p[:-1])
        x[3] = change_percent(high_p[-1], high_p[:-1])
        x[4] = change_percent(vol[-1], vol[:-1])
        x[5] = change_percent(trades[-1], trades[:-1])

        return x

    def get_data(self, window_size, ind = None, thresholds = [-0.02, 0.02]):
        '''
        Returns (x,y) pairs - inputs and outputs of the neural net.
        x has shape(6,window_size) and consists of the [ind-window_size : ind] arrays:
            open_price
            close_price
            low_price
            high_price
            volume
            no_trades
        All ararays are normalized (x-ref)/ref to the value corresponding to the index.

        y is a one-hotencoding for:
            buy
            sell
            hold
        '''

        if not ind is None:
            if ind < window_size or ind > len(self.change):
                #print('Index {} out of bounds: {}-{}'.format(ind,window_size,len(self.change)))
                ind = None
        if ind is None:
            ind = random.randint(window_size+1, len(self.change)-1)
        
        '''
        Data does not include the refference point [ind]. In deployment this means that the prediction is one sample ahead of the lookout window.
        '''

        x = self.preprocess_data(
            open_p = self.resampled_open_price[ind-window_size : ind+1], 
            close_p = self.resampled_close_price[ind-window_size : ind+1], 
            low_p = self.resampled_low_price[ind-window_size : ind+1], 
            high_p = self.resampled_high_price[ind-window_size : ind+1], 
            vol = self.resampled_volume[ind-window_size : ind+1], 
            trades = self.resampled_no_trades[ind-window_size : ind+1]
        )

        y = np.zeros(3)
        if self.change[ind] > thresholds[1]:
            'Price raises above the threshold: buy'
            y[0] = 1.
        if self.change[ind] < thresholds[0]:
            'Price dropb below the threshold: sell'
            y[1] = 1.
        if self.change[ind] > thresholds[0] and self.change[ind] < thresholds[1]:
            'Price stays constant: hold'
            y[2] = 1.

        return x, y, ind


    def get_samples(self, window_size, no_samples):

        ind = random.randint(0, len(self.change)-1-no_samples)

        raw_open_p = self.resampled_open_price[ind : ind+no_samples] 
        raw_close_p = self.resampled_close_price[ind : ind+no_samples]
        raw_low_p = self.resampled_low_price[ind : ind+no_samples] 
        raw_high_p = self.resampled_high_price[ind : ind+no_samples] 
        raw_vol = self.resampled_volume[ind : ind+no_samples] 
        raw_trades = self.resampled_no_trades[ind : ind+no_samples]

        raw_data = [
            raw_open_p, 
            raw_close_p, 
            raw_low_p, 
            raw_high_p, 
            raw_vol, 
            raw_trades 
        ]

        return raw_data


    def create_batch(self, batch_size = 32, random_split = 0., window_size = 10, thresholds = [-0.02, 0.02]):
        '''
        Creates a batch of (x,y) pairs by selecting windows of data either randomly or by the order of change.
        '''

        batch_x = np.zeros((batch_size, 6, window_size))
        batch_y = np.zeros((batch_size, 3))

        selected_inds = []
        random_inds = []
        inds = []

        random_samples = int(batch_size * random_split)

        for i in range(random_samples):   
            x, y, ind = self.get_data(window_size=window_size,thresholds=thresholds)

            batch_x[i] = x
            batch_y[i] = y
            inds.append(ind)

        for i in range(random_samples, batch_size):
            
            if len(self.indexes) < len(self.change) / 2:
                self.indexes = list(np.argsort(self.change))

            extremes = int(len(self.indexes) / 10)
            ind = self.indexes.pop(random.randint(-extremes,extremes))
            x, y, ind_2 = self.get_data(window_size=window_size, ind=ind, thresholds=thresholds) 

            batch_x[i] = x
            batch_y[i] = y
            inds.append(ind_2)


        def check_batch():
            '''
            Function that checks the selected indexes (relevant if not random sampled) to correctly be before critical moments.
            '''

            plt.figure()
        
            max_change = np.amax(self.change)
            min_change = np.amin(self.change)

            colors = np.zeros((len(self.resampled_close_price),4))

            for ind in inds:
                y = batch_y[inds.index(ind)]  
                if y[0] == 1: 
                    'buy'
                    colors[ind][1] = 1
                    colors[ind][3] = 1
                if y[1] == 1: 
                    'sell'
                    colors[ind][0] = 1
                    colors[ind][3] = 1
                if y[2] == 1: 
                    'hold'
                    colors[ind][2] = 1
                    colors[ind][3] = 1

            plt.plot(self.resampled_close_price, label = 'Closing price')
            plt.scatter(range(len(self.resampled_close_price)),self.resampled_close_price,c=colors)
            plt.legend()

            no_examples = 3
            plt.savefig(os.path.join(batch_example_dir, 'batch_{}.png'.format(random.randint(0,no_examples))))
        
        check_batch()

        return batch_x, batch_y


    def data_generator(self, batch_size = 32, random_split = 0., window_size = 10, thresholds = [-0.005, 0.005]):
        while True:
            x, y = self.create_batch(batch_size, random_split, window_size, thresholds)
            yield (x, y)


if __name__ == '__main__':
    'data info'
    coin1 = 'BNB'
    coin2 = 'USDT'
    start_date = '2020-11-1 12:00:00'
    end_date = '2020-11-10 12:00:00'

    test_start_date = '2020-11-2 12:00:00'
    test_end_date = '2021-12-29 12:00:00'

    'lookup_samples refers to the umber of samples in the prediction horizon'
    lookup_samples = 20

    'window_size refers to the number of samples considered for the generation of the prediction'
    window_size = 10
    data_shape = (6,window_size)

    'size of batches for the model training'
    batch_size = 128


    'Initialize the data_handler'
    data_handler = DataHandler(coin1, coin2)
    data_handler.define_subset(start_date, end_date)
    data_handler.compute_change(lookup_samples = lookup_samples, average_lookup_samples = False)
    data_generator = data_handler.data_generator(batch_size = 512)
    next(data_generator)

    '''
    test_data_handler = DataHandler(coin1, coin2)
    test_data_handler.define_subset(test_start_date, test_end_date)
    test_data_handler.compute_change(lookup_samples = lookup_samples, average_lookup_samples = True)
    test_data_generator = test_data_handler.data_generator(batch_size = batch_size)
    '''
    