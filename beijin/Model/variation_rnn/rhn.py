import torch
from torch.autograd import Variable
from torch import nn
from collections import OrderedDict
from Model.variation_rnn import dropout as dr
import importlib

class RecurrentHighwayNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, recurrence_length, recurrent_dropout=0.2):
        super(RecurrentHighwayNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size // 2 # birdirectional
        if hidden_size %2 != 0:
            print("[RHN]: Note that the hidden size % 2 should = 0.")

        self.L = recurrence_length
        self.recurrent_dropout = recurrent_dropout
        self.highways = nn.ModuleList()
        self.highways.append(RHNCell(self.input_size, self.hidden_size, is_first_layer=True, recurrent_dropout=recurrent_dropout))

        for _ in range(self.L - 1):
            self.highways.append(RHNCell(self.input_size, self.hidden_size, is_first_layer=False, recurrent_dropout=recurrent_dropout))

    def init_state(self, batch_size):
        hidden = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        return hidden

    def forward(self, _input, hidden=None):
        _input = _input.transpose(0, 1) # B * T * V
        batch_size = _input.size(0)
        max_time = _input.size(1)
        
        if hidden is None:
            hidden = self.init_state(batch_size)

        lefts = []
        rights = []

        for time in range(max_time):
            for rhn_cell in self.highways:
                hidden = rhn_cell(_input[:, time, :], hidden)
            lefts.append(hidden.unsqueeze(1))
        lefts = torch.cat(lefts, 1)

        for rhn_cell in self.highways:
            rhn_cell.end_of_sequence()

        for time in range(max_time):
            for rhn_cell in self.highways:
                hidden = rhn_cell(_input[:, time, :], hidden)
            rights.append(hidden.unsqueeze(1))
        rights = torch.cat(rights, 1)

        for rhn_cell in self.highways:
            rhn_cell.end_of_sequence()

        outputs = torch.cat((lefts, rights), dim=2).transpose(0, 1) # T * B * V
        hiddens = outputs[-1].unsqueeze(0).contiguous() # 1 * B * V
        return outputs, hiddens


class RHNCell(nn.Module):

    def __init__(self, input_size, hidden_size, is_first_layer, recurrent_dropout):
        super(RHNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_first_layer = is_first_layer

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.set_dropout(recurrent_dropout)

        # input weight matrices
        if self.is_first_layer:
            self.W_H = nn.Linear(input_size, hidden_size)
            self.W_C = nn.Linear(input_size, hidden_size)

        # hidden weight matrices
        self.R_H = nn.Linear(hidden_size, hidden_size, bias=True)
        self.R_C = nn.Linear(hidden_size, hidden_size, bias=True)

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.drop_ir = dr.SequentialDropout(p=dropout)
        self.drop_ii = dr.SequentialDropout(p=dropout)
        self.drop_hr = dr.SequentialDropout(p=dropout)
        self.drop_hi = dr.SequentialDropout(p=dropout)

    def end_of_sequence(self):
        self.drop_ir.end_of_sequence()
        self.drop_ii.end_of_sequence()
        self.drop_hr.end_of_sequence()
        self.drop_hi.end_of_sequence()

    def forward(self, _input, prev_hidden):
        c_i = self.drop_hr(prev_hidden)
        h_i = self.drop_hi(prev_hidden)

        if self.is_first_layer:
            x_i = self.drop_ii(_input)
            x_r = self.drop_ir(_input)
            hl = self.tanh(self.W_H(x_i) + self.R_H(h_i))
            tl = self.sigmoid(self.W_C(x_r) + self.R_C(c_i))
        else:
            hl = self.tanh(self.R_H(h_i))
            tl = self.sigmoid(self.R_C(c_i))

        h = (hl * tl) + (prev_hidden * (1 - tl))
        return h
