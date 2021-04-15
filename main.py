from model_handler import *
from data_handler import *

'data info'
coin1 = 'BNB'
coin2 = 'USDT'
start_date = '2019-12-1 12:00:00'
end_date = '2020-11-1 12:00:00'

test_start_date = '2020-11-2 12:00:00'
test_end_date = '2021-12-29 12:00:00'

'lookup_samples refers to the umber of samples in the prediction horizon'
lookup_samples = 5



'Initialize the data_handler'
data_handler = DataHandler(coin1, coin2)
data_handler.define_subset(start_date, end_date)
data_handler.compute_change(lookup_samples = lookup_samples, average_lookup_samples = True)


test_data_handler = DataHandler(coin1, coin2)
test_data_handler.define_subset(test_start_date, test_end_date)
test_data_handler.compute_change(lookup_samples = lookup_samples, average_lookup_samples = True)


model_handler = ModelHandler('Model2')
model_handler.make_model(window_size = 20)

model_handler.train(
    no_epochs = 15, 
    steps_per_epoch = 150, 
    batch_size = 128,
    data_handler = data_handler, 
    test_data_handler = test_data_handler,
    starting_learning_rate = 0.0001,
    thresholds=[-0.01, 0.01], 
    random_split=0.
    )

model_handler.evaluate(data_handler, 500)