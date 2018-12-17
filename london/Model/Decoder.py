import random
import torch
import torch.nn as nn

from torch.autograd import Variable

class VanillaDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, use_cuda):
        """Define layers for a vanilla rnn decoder"""
        super(VanillaDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax()  # work with NLLLoss = CrossEntropyLoss
        self.output_length = 48
        self.use_cuda = use_cuda

    def forward_step(self, inputs, hidden):
        # inputs: (time_steps=1, batch_size)
        rnn_output, hidden = self.gru(inputs, hidden)  #rnn_output=T(1) X B X H  inputs = T(1) x B x H
        output = self.out(rnn_output.transpose(0,1).squeeze(1)).unsqueeze(1)  # S = B x O


        return output, rnn_output, hidden

    def forward(self, context_vector, decoder_first_input):

        # Prepare variable for decoder on time_step_0
        batch_size = context_vector.size(1)
        decoder_input = decoder_first_input

        # Pass the context vector
        decoder_hidden = context_vector

        decoder_outputs = Variable(torch.zeros(
            self.output_length,
            batch_size,
            self.output_size
        ))  # (time_steps, batch_size, vocab_size) contain every timestep air prediction 
        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()
        
       
        # Unfold the decoder RNN on the time dimension
        for t in range(self.output_length):
            output, _, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[t] = output
           

        return decoder_outputs.transpose(0,1), decoder_hidden

    def evaluation(self, context_vector):
        batch_size = context_vector.size(1) # get the batch size
        decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))
        decoder_hidden = context_vector

        decoder_outputs = Variable(torch.zeros(
            self.max_length,
            batch_size,
            self.output_size
        ))  # (time_steps, batch_size, vocab_size)

        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_outputs = decoder_outputs.cuda()

        # Unfold the decoder RNN on the time dimension
        for t in range(self.max_length):
            decoder_outputs_on_t, _, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[t] = decoder_outputs_on_t
            decoder_input = self._decode_to_index(decoder_outputs_on_t)  # select the former output as input

        return self._decode_to_indices(decoder_outputs)

    def _decode_to_index(self, decoder_output):
        """
        evaluate on the logits, get the index of top1
        :param decoder_output: S = B x V or T x V
        """
        value, index = torch.topk(decoder_output, 1)
        index = index.transpose(0, 1)  # S = 1 x B, 1 is the index of top1 class
        if self.use_cuda:
            index = index.cuda()
        return index

    def _decode_to_indices(self, decoder_outputs):
        """
        Evaluate on the decoder outputs(logits), find the top 1 indices.
        Please confirm that the model is on evaluation mode if dropout/batch_norm layers have been added
        :param decoder_outputs: the output sequence from decoder, shape = T x B x V 
        """
        decoded_indices = []
        batch_size = decoder_outputs.size(1)
        decoder_outputs = decoder_outputs.transpose(0, 1)  # S = B x T x V

        for b in range(batch_size):
            top_ids = self._decode_to_index(decoder_outputs[b])
            decoded_indices.append(top_ids.data[0])
        return decoded_indices