import random
import torch
import torch.nn as nn

from torch.autograd import Variable
from Model.embedding import FeatureEmbedding
from Model.norm import LayerNorm
from Model.attention import Attention

class VanillaDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, use_cuda, num_layers=1):
        """Define layers for a vanilla rnn decoder"""
        super(VanillaDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.feature_embedding = FeatureEmbedding(100, bias=6)
        self.gru = nn.GRU(input_size + self.feature_embedding.embedding_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax()  # work with NLLLoss = CrossEntropyLoss
        self.output_length = 48
        self.teacher_forcing_ratio = 0.0
        self.use_cuda = use_cuda

    def forward_step(self, inputs, hidden):
        # inputs: (time_steps=1, batch_size)
        rnn_output, hidden = self.gru(inputs, hidden)  #rnn_output=T(1) X B X H  inputs = T(1) x B x H
        output = self.out(rnn_output.transpose(0, 1).squeeze(1)).unsqueeze(1)  # S = B x O
        return output, rnn_output, hidden

    def forward(self, context_vector, dec_entity, decoder_first_input, encoder_outputs, target_vars, validate=False):
    
        # Prepare variable for decoder on time_step_0
        batch_size = context_vector.size(1)
        #print(decoder_first_input.size())
        entity_embedding = self.feature_embedding(dec_entity) # T * B * V
        #print("EMB", entity_embedding[0].unsqueeze(0))
        decoder_input = torch.cat((decoder_first_input, entity_embedding[0].unsqueeze(0)), dim=2) # 1, B, 3+embedding
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
        
        if not validate:
            use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        # Unfold the decoder RNN on the time dimension
        for t in range(self.output_length):
            output, _, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[t] = output
            if use_teacher_forcing:
                decoder_input = target_vars[t]
            else:
                decoder_input = output.squeeze(1) # B * 1 * O -> B * O
            #print("DEC", decoder_input.size()) # B * O
            #print("ENT", entity_embedding[t].size()) # B * O
            decoder_input = torch.cat((decoder_input, entity_embedding[t]), dim=1).unsqueeze(0)
           
        return decoder_outputs.transpose(0,1), decoder_hidden

class VanillaResidualDecoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, use_cuda, num_layers=1):
        """Define layers for a vanilla rnn decoder"""
        super(VanillaResidualDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.feature_embedding = FeatureEmbedding(100, bias=6)
        self.hidden_size = input_size + self.feature_embedding.embedding_size
        self.gru = nn.GRU(input_size + self.feature_embedding.embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size + input_size + self.feature_embedding.embedding_size, output_size)
        self.log_softmax = nn.LogSoftmax()  # work with NLLLoss = CrossEntropyLoss
        self.output_length = 48
        self.teacher_forcing_ratio = 0.0
        self.use_cuda = use_cuda
        self.layer_norm = LayerNorm(1)

    def forward_step(self, inputs, hidden):
        # inputs: (time_steps=1, batch_size)
        rnn_output, hidden = self.gru(inputs, hidden)  #rnn_output=T(1) X B X H  inputs = T(1) x B x H
        add_norm = self.layer_norm(torch.cat((rnn_output, inputs), dim=2))
        output = self.out(add_norm.transpose(0, 1).squeeze(1)).unsqueeze(1)  # S = B, 1, O
        return output, rnn_output, hidden

    def forward(self, context_vector, dec_entity, decoder_first_input, target_vars, validate=False):

        # Prepare variable for decoder on time_step_0
        batch_size = context_vector.size(1)
        entity_embedding = self.feature_embedding(dec_entity) # T * B * V
        decoder_input = torch.cat((decoder_first_input, entity_embedding[0].unsqueeze(0)), dim=2) # 1, B, 3+embedding
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
        
        if not validate:
            use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        # Unfold the decoder RNN on the time dimension
        for t in range(self.output_length):
            output, _, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[t] = output
            if use_teacher_forcing:
                decoder_input = target_vars[t]
            else:
                decoder_input = output.squeeze(1) # B * 1 * O -> B * O
            #print("DEC", decoder_input.size()) # B * O
            #print("ENT", entity_embedding[t].size()) # B * O
            decoder_input = torch.cat((decoder_input, entity_embedding[t]), dim=1).unsqueeze(0)
           
        return decoder_outputs.transpose(0,1), decoder_hidden

class AttentionDecoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, use_cuda, num_layers=1):
        """Define layers for a vanilla rnn decoder"""
        super(AttentionDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.feature_embedding = FeatureEmbedding(100, bias=6)
        self.gru = nn.GRU(input_size + self.feature_embedding.embedding_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax()  # work with NLLLoss = CrossEntropyLoss
        self.output_length = 48
        self.teacher_forcing_ratio = 0.0
        self.use_cuda = use_cuda
        self.attention = Attention(self.hidden_size)

    def forward_step(self, inputs, hidden, encoder_outputs):
        output, hidden = self.gru(inputs, hidden)
        output, attn = self.attention(output, encoder_outputs) # output: B * T * V
        #print("O", output.size())
        predicted = self.out(output.contiguous())
        #print("PS", predicted.size())
        return predicted, hidden, attn


    def forward(self, context_vector, dec_entity, decoder_first_input, encoder_outputs, target_vars, validate=False):

        # Prepare variable for decoder on time_step_0
        batch_size = context_vector.size(1)
        #print(decoder_first_input.size())
        entity_embedding = self.feature_embedding(dec_entity) # T * B * V
        #print("EMB", entity_embedding[0].unsqueeze(0))
        decoder_input = torch.cat((decoder_first_input, entity_embedding[0].unsqueeze(0)), dim=2) # 1, B, 3+embedding
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
        
        if not validate:
            use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        # Unfold the decoder RNN on the time dimension
        for t in range(self.output_length):
            output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[t] = output
            if use_teacher_forcing:
                decoder_input = target_vars[t]
            else:
                decoder_input = output.squeeze(1) # B * 1 * O -> B * O
            #print("DEC", decoder_input.size()) # B * O
            #print("ENT", entity_embedding[t].size()) # B * O
            decoder_input = torch.cat((decoder_input, entity_embedding[t]), dim=1).unsqueeze(0)
           
        return decoder_outputs.transpose(0,1), decoder_hidden