import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Model.variation_rnn import rhn as rhn
from Model.embedding import FeatureEmbedding
from Model.norm import LayerNorm

class VanillaEncoder(nn.Module):

    def __init__(self, hidden_size, output_size, embedding_size, num_layers):
        """Define layers for a vanilla rnn encoder"""
        super(VanillaEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.feature_embedding = FeatureEmbedding(embedding_size)
        self.gru = nn.GRU(input_size= 6 + self.feature_embedding.embedding_size, 
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=False)

        self.hidden_transform = nn.Linear(hidden_size, output_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, hidden=None):
        feature_embedding = self.feature_embedding(input_seqs)
        input_seqs = torch.cat((input_seqs[:, :, 0:6], feature_embedding), dim=2)
        #print("R", new_input_seqs.size())
        #print(input_seqs.size())
        outputs, hidden = self.gru(input_seqs, hidden)
        hidden = hidden[-1].unsqueeze(0)
        #freeze_err = Variable(hidden.data, requires_grad=False).cuda()
        hidden_tansformed = self.hidden_transform(hidden.transpose(0, 1)).transpose(1, 0)
        outputs_transformed = self.out(outputs.transpose(0, 1))  # S = B x O

        return outputs, outputs_transformed, hidden, hidden_tansformed

class AVEncoder(nn.Module):

    def __init__(self, hidden_size, output_size, embedding_size, num_layers):
        """Define layers for a vanilla rnn encoder"""
        super(AVEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.feature_embedding = FeatureEmbedding(embedding_size)
        self.gru = nn.GRU(input_size= 6 + self.feature_embedding.embedding_size, 
                          hidden_size=hidden_size // 2,
                          num_layers=num_layers,
                          bidirectional=False)

        self.hidden_transform = nn.Linear(hidden_size, output_size)
        self.out = nn.Linear(hidden_size // 2, output_size)

    def forward(self, input_seqs, hidden=None):
        feature_embedding = self.feature_embedding(input_seqs)
        input_seqs = torch.cat((input_seqs[:, :, 0:6], feature_embedding), dim=2)
        #print("R", new_input_seqs.size())
        #print(input_seqs.size())
        outputs, hidden = self.gru(input_seqs, hidden)
        
        last_h = hidden[-1].unsqueeze(0)
        mean_h = torch.mean(outputs, dim=0).unsqueeze(0)
        #print("mean_h", mean_h.size())
        #print("last_h", last_h.size())
        av_h = torch.cat((last_h, mean_h), dim=2)

        hidden_tansformed = self.hidden_transform(av_h.transpose(0, 1)).transpose(1, 0)
        outputs_transformed = self.out(outputs.transpose(0, 1))  # S = B x O

        return outputs, outputs_transformed, av_h, hidden_tansformed

class VanillaResidualEncoder(nn.Module):

    def __init__(self, hidden_size, output_size, embedding_size, num_layers, window_size=2):
        """Define layers for a vanilla rnn encoder"""
        super(VanillaResidualEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.feature_embedding = FeatureEmbedding(embedding_size)
        self.gru = nn.GRU(input_size= 6 + self.feature_embedding.embedding_size, 
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=False)
        #self.layer_norm = LayerNorm(48)
        self.hidden_transform = nn.Linear(hidden_size, output_size)
        self.layer_norm = LayerNorm(window_size * 24)
        self.out = nn.Linear(6 + self.feature_embedding.embedding_size + hidden_size, output_size)

    def forward(self, input_seqs, hidden=None):
        feature_embedding = self.feature_embedding(input_seqs)
        input_seqs = torch.cat((input_seqs[:, :, 0:6], feature_embedding), dim=2)
        #print("R", new_input_seqs.size())
        #print(input_seqs.size())
        outputs, hidden = self.gru(input_seqs, hidden)
        outputs = self.layer_norm(outputs)
        hidden = hidden[-1].unsqueeze(0)
        hidden_tansformed = self.hidden_transform(hidden.transpose(0, 1)).transpose(1, 0)

        # for encoder's output
        residual = torch.cat((outputs, input_seqs), dim=2) # S = T * B * (H + I)
        outputs_transformed = self.out(residual.transpose(0, 1))  # S = B x O

        return outputs, outputs_transformed, hidden, hidden_tansformed
class BidirectionalGRUEncoder(nn.Module):
    
    def __init__(self, hidden_size, output_size, embedding_size, num_layers):
        """Define layers for a vanilla rnn encoder"""
        super(BidirectionalGRUEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.feature_embedding = FeatureEmbedding(embedding_size)
        self.gru = nn.GRU(input_size= 6 + self.feature_embedding.embedding_size, 
                          hidden_size= hidden_size // 2,
                          bidirectional=True,
                          num_layers=num_layers)

        self.hidden_transform = nn.Linear(hidden_size, output_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, hidden=None):
        feature_embedding = self.feature_embedding(input_seqs)
        #print("E", feature_embedding.size())
        new_input_seqs = torch.cat((input_seqs[:, :, 0:6], feature_embedding), dim=2)
        #print("R", new_input_seqs.size())
        #print(input_seqs.size())
        outputs, hidden = self.gru(new_input_seqs, hidden)
        hidden = hidden[-self.num_layers:]
        hidden = torch.cat(hidden, dim=1).unsqueeze(0)
        hidden_tansformed = self.hidden_transform(hidden.transpose(0,1)).transpose(1,0)
        outputs_transformed = self.out(outputs.transpose(0,1))  # S = B x O

        return outputs, outputs_transformed, hidden, hidden_tansformed

        
class HighwayEncoder(nn.Module):

    def __init__(self, hidden_size, output_size, embedding_size, depth):
        """Define layers for a vanilla rnn encoder"""
        super(HighwayEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.depth = depth

        self.feature_embedding = FeatureEmbedding(embedding_size)
        self.rhn = rhn.RecurrentHighwayNetwork(input_size=6 + self.feature_embedding.embedding_size, hidden_size=hidden_size, recurrence_length=5, recurrent_dropout=0)
        self.hidden_transform = nn.Linear(hidden_size, output_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, hidden=None):
        feature_embedding = self.feature_embedding(input_seqs)
        new_input_seqs = torch.cat((input_seqs[:, :, 0:6], feature_embedding), dim=2)
        outputs, hidden = self.rhn(new_input_seqs, hidden)
        hidden_tansformed = self.hidden_transform(hidden.transpose(0,1)).transpose(1,0)
        outputs_transformed = self.out(outputs.transpose(0,1))  # S = B x O

        return outputs, outputs_transformed, hidden, hidden_tansformed

