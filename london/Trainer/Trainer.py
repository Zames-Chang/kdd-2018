import torch

class Trainer(object):

    def __init__(self, model, data_transformer, loggers, learning_rate, use_cuda,
                 checkpoint_name= "Model_CheckPoint/seq2seqModel.pt"
                ):

        self.model = model
        self.train_logger = loggers[0]
        self.valid_logger = loggers[1]
        # record some information about dataset
        self.data_transformer = data_transformer
        self.use_cuda = use_cuda

        # optimizer setting
        self.learning_rate = learning_rate
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.checkpoint_name = checkpoint_name

    def train(self, num_epochs, batch_size, window_size, pretrained= False, valid_portion= 0.8):
        
        self.window_size = window_size
        if pretrained:
            self.load_model()

        self.global_step = 0
        for epoch in range(0, num_epochs):
            print("In %d epoch" %(epoch))
            for data in self.data_transformer.every_station_data:

                # validation
                train_data = data[: int(len(data)*valid_portion)]
                valid_data = data[ int(len(data)*valid_portion):]  
  
                for input_batch, target_batch in self.data_transformer.mini_batch_generator(train_data, batch_size=batch_size, window_size= self.window_size):

                    self.optimizer.zero_grad()
                    encoder_outputs, decoder_outputs = self.model(input_batch)

                    
                    # calculate the loss and back prop.
                    cur_loss = self.get_loss(encoder_outputs,
                                             decoder_outputs,
                                             input_batch[:,:,0:3].transpose(0, 1),
                                             target_batch.transpose(0, 1))

                    smape_loss = self.smape_loss(encoder_outputs,
                                                 decoder_outputs,
                                                 input_batch[:,:,0:3].transpose(0, 1),
                                                 target_batch.transpose(0, 1))
                    cur_loss.backward()                     
                    # logging
                    self.global_step += 1
                    if self.global_step % 10 == 0:
                        self.tensorboard_log(cur_loss, smape_loss)

                        print("Step: %d, Mse Loss : %f, SMAPE Loss : %f" %(self.global_step,cur_loss.data[0],smape_loss.data[0] ))
                        # self.save_model()
                        self.validation(valid_data)

                    

                    # optimize
                    self.optimizer.step()

        self.save_model()

    def smape_loss(self, encoder_outputs, decoder_outputs, input_batch, target_batch):  

        concat_predict = torch.cat((encoder_outputs, decoder_outputs), dim= 1)
        concat_label = torch.cat((input_batch, target_batch), dim= 1)
        
        concat_predict = concat_predict[:, :, :3]
        concat_label = concat_label[:, :, :3]
        # print(torch.abs(concat_predict - concat_label).size())
        loss = 2 *  (torch.abs(concat_predict - concat_label).sum(2) /  (concat_predict + concat_label).sum(2)) # B ＊　Ｔ
        loss = loss.sum()
        loss = loss / (concat_predict.size(0) * concat_predict.size(1))   # Divide the batch size and number of days to get mean 

        return loss

    def get_loss(self, encoder_outputs, decoder_outputs, input_batch, target_batch):  
        # print("Input batch size", input_batch.size())
        # print("Target batch size", target_batch.size())
        concat_predict = torch.cat((encoder_outputs, decoder_outputs), dim= 1)
        concat_label = torch.cat((input_batch, target_batch), dim= 1)

        
        loss = self.criterion(concat_predict,concat_label)
        # for i in range(concat_label.size(1)):
        #     i_timestep_predict = concat_predict[:,i,:].contiguous().view(concat_label.size(0),-1)
        #     i_timestep_label = concat_label[:,i,:].contiguous().view(concat_label.size(0),-1)
        #     loss += self.criterion(i_timestep_predict, i_timestep_label)
        return loss

    def validation(self, valid_data):
        total_mse_loss = 0
        total_smape_loss = 0
        number_of_batch =0
        for input_batch, target_batch in self.data_transformer.mini_batch_generator(valid_data, batch_size=32, window_size=self.window_size):
                    encoder_outputs, decoder_outputs = self.model(input_batch)

                    # calculate the loss and back prop.
                    cur_mse_loss = self.get_loss(encoder_outputs,
                                             decoder_outputs,
                                             input_batch[:,:,0:3].transpose(0, 1),
                                             target_batch.transpose(0, 1))

                    smape_loss = self.smape_loss(encoder_outputs,
                                                 decoder_outputs,
                                                 input_batch[:,:,0:3].transpose(0, 1),
                                                 target_batch.transpose(0, 1))
                    total_mse_loss += (cur_mse_loss*input_batch.size(1))  # Mulitply Batch number  input_batch size: T * B * H 
                    total_smape_loss += (smape_loss*input_batch.size(1))
                    number_of_batch += input_batch.size(1)
        if number_of_batch == 0:
            # print(valid_data)
            # print(total_mse_loss)
            # print(total_smape_loss)
            return 
        total_mse_loss /= number_of_batch
        total_smape_loss /= number_of_batch
        self.tensorboard_log(total_mse_loss, total_smape_loss, valid= True)
        print("Validation, Mse Loss : %f, SMAPE Loss : %f" %(total_mse_loss.data[0],total_smape_loss.data[0] ))


    def predictition(self, predict_data):
        for input_batch, _ in self.data_transformer.mini_batch_generator(predict_data, batch_size=32, window_size=self.window_size):
                    encoder_outputs, decoder_outputs = self.model(input_batch)

                    
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint_name)
        print("Model has been saved as %s.\n" % self.checkpoint_name)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name))
        print("Pretrained model has been loaded.\n")

    def tensorboard_log(self, mse_loss, smape_loss, valid= False):
        info = {
                'mse_loss': mse_loss.data[0],
                'smape_loss': smape_loss.data[0]
            }
        if not valid:
            
            for tag, value in info.items():
                self.train_logger.scalar_summary(tag, value, self.global_step)

        else:
            for tag, value in info.items():
                self.valid_logger.scalar_summary(tag, value, self.global_step)
