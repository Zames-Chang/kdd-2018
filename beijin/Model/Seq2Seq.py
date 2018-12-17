import random
import torch
import torch.nn as nn

from torch.autograd import Variable


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_inputs, dec_inputs, target_vars, validate=False):
        encoder_outputs, encoder_prediction, encoder_hidden, encoder_transform = self.encoder.forward(enc_inputs)
        decoder_outputs, decoder_hidden = self.decoder.forward(context_vector=encoder_hidden, dec_entity=dec_inputs,
        decoder_first_input=encoder_transform, encoder_outputs=encoder_outputs,
        target_vars=target_vars, validate=validate)
        return encoder_prediction, decoder_outputs
    
