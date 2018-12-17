# London train process 

from torch.utils.data import DataLoader
from torch.autograd import Variable
from Dataloader.Seq2Seq_DataLoader import DataTransformer
import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn

from Model.Encoder import VanillaEncoder
from Model.Decoder import VanillaDecoder
from Model.Seq2Seq import Seq2Seq
from Trainer.Trainer import Trainer

from Tensorboard.Tensorboard import Logger

if __name__ == "__main__":


    data_transformer = DataTransformer(path = "Data/london/processed_data/Processed_london_aq_data.csv",
                                    use_cuda = True)


    Enc = VanillaEncoder(hidden_size = 300,
                        output_size = 3).cuda()


    Dec = VanillaDecoder(input_size = 3,
                        hidden_size = 300, 
                        output_size = 3,
                        use_cuda = True).cuda()

    seq2seq = Seq2Seq(encoder=Enc,
                    decoder=Dec)

    
    loggers = (train_logger, valid_logger)
    trainer = Trainer(seq2seq, data_transformer, loggers= loggers, learning_rate = 0.001, use_cuda= True)
    trainer.load_model()

    
    trainer.train(num_epochs=5, 
                batch_size=10, 
                window_size = 2,
                pretrained=False,
                valid_portion = 0.8)
