import torch
import torch.nn as nn
from Dataloader.data_config import feature_config

class FeatureEmbedding(nn.Module):

    def __init__(self, embedding_size, bias=0, activation=True):
        super(FeatureEmbedding, self).__init__()

        self.embedding_size = embedding_size

        self.hour_embedding = nn.Embedding(25, 8)
        self.day_embedding = nn.Embedding(32, 8)
        self.month_embedding = nn.Embedding(13, 6)
        #self.year_embedding = nn.Embedding(2, embedding_size)
        self.dayofweek_embedding = nn.Embedding(8, 3)
        self.dayofyear_embedding = nn.Embedding(367, 20)
        self.station_embedding = nn.Embedding(50, 20)
        #self.is_weekend_embedding = nn.Embedding(2, embedding_size)
       # self.work_time_embedding = nn.Embedding(2, embedding_size)
        self.embedding_size = 8 + 8 + 6 + 3 + 20 + 20 + 1
        self.bias = bias
        self.activation = activation
        self.tanh = nn.Tanh()

    def forward(self, input_seqs):
        hour_embedded = self.hour_embedding(input_seqs[:, :, feature_config['hour'] - self.bias].long())
        day_embedded = self.day_embedding(input_seqs[:, :, feature_config['day'] - self.bias].long())
        month_embedded = self.month_embedding(input_seqs[:, :, feature_config['month'] - self.bias].long())
        #year_embedded = self.year_embedding(input_seqs[:, :, feature_config['year']].long())
        dayofweek_embedded = self.dayofweek_embedding(input_seqs[:, :, feature_config['dayofweek'] - self.bias].long())
        dayofyear_embedded = self.dayofyear_embedding(input_seqs[:, :, feature_config['dayofyear'] - self.bias].long())
        station_embedded = self.station_embedding(input_seqs[:, :, feature_config['station'] - self.bias].long())
        time_lag = input_seqs[:, :, feature_config['time_lag'] - self.bias].float().unsqueeze(2)

        #is_weekend_embedded = self.is_weekend_embedding(input_seqs[:, :, feature_config['is_weekend']].long())
        #work_time_embedded = self.work_time_embedding(input_seqs[:, :, feature_config['work_time']].long())

        feature_embedding = torch.cat((hour_embedded,
                                        day_embedded,
                                        month_embedded,
                                        #year_embedded,
                                        dayofweek_embedded,
                                        dayofyear_embedded,
                                        station_embedded,
                                        ), dim=2)
        if self.activation:
            feature_embedding = self.tanh(feature_embedding)
        feature_embedding = torch.cat((feature_embedding, time_lag), dim=2)
        return feature_embedding