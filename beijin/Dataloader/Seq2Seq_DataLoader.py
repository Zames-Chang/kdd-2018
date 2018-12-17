from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from torch.autograd import Variable
from sklearn import preprocessing
from Dataloader.data_config import feature_config, bejing_data_bias, londo_data_bias 
from Dataloader.data_config import bejing_target_columns, london_target_columns

import pandas as pd 
import numpy as np 
import torch


class DataTransformer(object):
    def __init__(self, path , use_cuda, for_bj=True):
        self.use_cuda = use_cuda
        self.path = path
        # self.path = "Data/beijing/beijing_2017_1_2018_3_aq.csv"
        self.columns = ['PM2.5','PM10','NO2','CO','O3','SO2']

        self.for_bj = for_bj
        if self.for_bj:
            self.target_columns = bejing_target_columns
            self.data_bias = bejing_data_bias
        else:
            self.target_columns = london_target_columns
            self.data_bias = londo_data_bias

        self.raw_data = pd.read_csv(self.path)
        self.prepare_data()

    def prepare_data(self):
        self.data = self.raw_data.dropna(axis='index', how='any')
        # self.data = self.raw_data.fillna(0)
        self.feature_expand()
        self.data.set_index(['stationId'], inplace=True)
        self.station_id = self.data.index.unique().tolist()
        self.every_station_data = []
        for station in self.station_id:
            self.every_station_data.append(self.data.loc[station,:].iloc[10:10282,:].values)
    
    def feature_expand(self):
        day_time = pd.to_datetime(self.data['utc_time']).dt

        should_drop = ['year', 'time_step', 'utc_time']

        # Expand some time feature
        self.data['hour'] = day_time.hour
        self.data['day'] = day_time.day
        self.data['month'] = day_time.month
        self.data['year'] = day_time.year - 2017 # will drop
        self.data['dayofweek'] = day_time.dayofweek
        self.data['dayofyear'] = day_time.dayofyear
        self.data['station'] = self.data['stationId'].values
        self.data['time_step'] = self.data['hour'] + (self.data['year'] * 365 + self.data['dayofyear'] - 1) * 24 # will drop
        self.data['time_lag'] = (self.data['time_step'] - self.data['time_step'].shift(1)).fillna(0)
        #self.data['is_weekend'] = (self.data['dayofweek']==5) | (self.data['dayofweek']==6)
        #self.data['work_time'] = ( self.data.hour > 6) & ( self.data.hour < 20)

        # Drop utc-time column. We already get some feature above
        self.data.drop(should_drop, axis=1, inplace=True)
        
        le = preprocessing.LabelEncoder()
        self.one_hot_candidate_columns = [
            'station',
        ]
        for col in self.one_hot_candidate_columns:
            le.fit(self.data[col])
            self.data[col] = le.transform(self.data[col])
        print("[Entity Features] ", self.data.columns.values)
        # self.year_dict = {'2017':0, '2018':1}
        # self.data['year'] = self.data['year'].apply(lambda v: self.year_dict[v])

    def create_mini_batch(self, data, batch_size= 10, window_size=2, use_cuda=False, time_lag=10, shuffle_input=True):
        encoder_batch_list = []
        decoder_label_list = []
        decoder_batch_list = []
        
        # NOTE THAT SHOULD DO SPECIAL PROCESS FOR FINAL BATCH!!!
        for k in range(0, len(data)):     # 0, data_length, stride
            if k + window_size * 24 + 48 + time_lag <= len(data):
                encoder_batch_list.append(data[k: k + window_size * 24])
                encoder_supervised_label = data[k+1: k + window_size * 24 + 1, self.target_columns]

                dec_input = data[k + window_size * 24 + time_lag: k + window_size * 24 + 48 + time_lag, self.data_bias:]
                #print("B", dec_input[:, feature_config['time_lag'] - bejing_data_bias])
                dec_input[:, feature_config['time_lag'] - self.data_bias][0] += time_lag # add an offset
                #print("A", dec_input[:, feature_config['time_lag'] - bejing_data_bias])
                decoder_supervised_label = data[k + window_size *24 + time_lag: k + window_size *24 + 48 + time_lag, self.target_columns]

                dual_supervised_label = np.concatenate((encoder_supervised_label, decoder_supervised_label))

                decoder_label_list.append(dual_supervised_label)
                decoder_batch_list.append(dec_input)

        assert len(encoder_batch_list) == len(decoder_batch_list)
        mini_encoder_batches = [
                np.array(encoder_batch_list[k: k + batch_size])
                for k in range(0, len(encoder_batch_list), batch_size)
        ]

        mini_decoder_batches = [
                np.array(decoder_batch_list[k: k + batch_size])
                for k in range(0, len(encoder_batch_list), batch_size)
        ]

        mini_labels = [
                np.array(decoder_label_list[k: k + batch_size])
                for k in range(0, len(decoder_batch_list), batch_size)
        ]

        if shuffle_input:
            mini_encoder_batches, mini_decoder_batches, mini_labels = shuffle(mini_encoder_batches, mini_decoder_batches, mini_labels, random_state=42)
        return mini_encoder_batches, mini_decoder_batches, mini_labels
    
    def prepare_all_station_data(self, all_station_data, training_portion, batch_size, valid_batch_size, window_size, time_lag, shuffle_input):
        all_station_train = []
        all_station_valid = []

        all_station_train_input = []
        all_station_train_label = []
        all_station_train_dec_input = []

        all_station_valid_input = []
        all_station_valid_label = []
        all_station_valid_dec_input = []


        for data in all_station_data:
            all_station_train.append(data[: int(len(data) * training_portion)])
            all_station_valid.append(data[ int(len(data) * training_portion):])
        
        for t in all_station_train:
            ci, di, cl = self.create_mini_batch(data=t, batch_size=batch_size, window_size=window_size, time_lag=time_lag, shuffle_input=shuffle_input)
            all_station_train_input += ci
            all_station_train_dec_input += di
            all_station_train_label += cl

        for t in all_station_valid:
            ci, di, cl = self.create_mini_batch(data=t, batch_size=valid_batch_size, window_size=window_size, time_lag=time_lag, shuffle_input=shuffle_input)
            all_station_valid_input += ci
            all_station_valid_dec_input += di
            all_station_valid_label += cl

        if shuffle_input:
            all_station_train_input, all_station_train_dec_input, all_station_train_label = shuffle(all_station_train_input, all_station_train_dec_input, all_station_train_label, random_state=42)
        
        return all_station_train_input, all_station_train_dec_input, all_station_train_label, np.array(all_station_valid_input), np.array(all_station_valid_dec_input), np.array(all_station_valid_label)

    def variables_generator(self, enc_batches, dec_batches, labels):
        for enc_batch, dec_batch, label in zip (enc_batches, dec_batches, labels):
            enc_input_var = Variable(torch.FloatTensor(enc_batch)).transpose(0, 1)  # time * batch
            dec_input_var = Variable(torch.FloatTensor(dec_batch)).transpose(0, 1)  # time * batch
            target_var = Variable(torch.FloatTensor(label)).transpose(0, 1)  # time * batch

            if self.use_cuda:
                    enc_input_var = enc_input_var.cuda()
                    dec_input_var = dec_input_var.cuda()
                    target_var = target_var.cuda()
            
            yield enc_input_var, dec_input_var, target_var

if __name__ == '__main__':
    data_tran = DataTransformer(path = "../Data/beijing/beijing_2017_1_2018_3_aq.csv",
                                use_cuda = True)

    for station_data in data_tran.every_station_data:
        for i , (batch, label) in enumerate(data_tran.mini_batch_generator(station_data)):
            print(batch.shape)
            print(label.shape)