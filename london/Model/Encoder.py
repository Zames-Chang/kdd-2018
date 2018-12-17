import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VanillaEncoder(nn.Module):

    def __init__(self, hidden_size, output_size):
        """Define layers for a vanilla rnn encoder"""
        super(VanillaEncoder, self).__init__()
        embedding_size = 100

        
        self.gru = nn.GRU(input_size= 803, 
                          hidden_size= hidden_size)

        self.hour_embedding = nn.Embedding(24, embedding_size)
        self.day_embedding = nn.Embedding(31, embedding_size)
        self.month_embedding = nn.Embedding(12, embedding_size)
        self.year_embedding = nn.Embedding(2, embedding_size)
        self.dayofweek_embedding = nn.Embedding(7, embedding_size)
        self.dayofyear_embedding = nn.Embedding(366, embedding_size)
        self.is_weekend_embedding = nn.Embedding(2, embedding_size)
        self.work_time_embedding = nn.Embedding(2, embedding_size)

        self.hidden_transform = nn.Linear(hidden_size, output_size)
        
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, hidden=None):

        hour_embedded = self.hour_embedding(input_seqs[:, :, 3].long())
        day_embedded = self.day_embedding(input_seqs[:, :, 4].long())
        month_embedded = self.month_embedding(input_seqs[:, :, 5].long())
        year_embedded = self.year_embedding(input_seqs[:, :, 6].long())
        dayofweek_embedded = self.dayofweek_embedding(input_seqs[:, :, 7].long())
        dayofyear_embedded = self.dayofyear_embedding(input_seqs[:, :, 8].long())
        is_weekend_embedded = self.is_weekend_embedding(input_seqs[:, :, 9].long())
        work_time_embedded = self.work_time_embedding(input_seqs[:, :, 10].long())

        one_hot_total_embedding = torch.cat((hour_embedded,
                                             day_embedded,
                                             month_embedded,
                                             year_embedded,
                                             dayofweek_embedded,
                                             dayofyear_embedded,
                                             is_weekend_embedded,
                                             work_time_embedded
                                                ), dim= 2)
        new_input_seqs = torch.cat((input_seqs[:, :, 0:3],one_hot_total_embedding), dim= 2)
        outputs, hidden = self.gru(new_input_seqs, hidden)

        hidden_tansformed = self.hidden_transform(hidden.transpose(0,1)).transpose(1,0)
        # hidden_tansformed = hidden 
        outputs_transformed = self.out(outputs.transpose(0,1))  # S = B x O

        return outputs_transformed, hidden, hidden_tansformed

   