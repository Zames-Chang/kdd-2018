import torch
import numpy as np
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

    def train(self, num_epochs, batch_size, window_size, pretrained= False, valid_portion= 0.8, time_lag=10):
        
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

                print("Training on %d, Validation with %d" % (len(train_data), len(valid_data)))
                encoder_batches, decoder_batches, decoder_labels = self.data_transformer.create_mini_batch(train_data, batch_size=batch_size, window_size= self.window_size, time_lag=time_lag)
                for input_batch, target_batch, target_label in self.data_transformer.variables_generator(encoder_batches, decoder_batches, decoder_labels):

                    self.optimizer.zero_grad()
                    self.model.train()
                    encoder_outputs, decoder_outputs = self.model(input_batch, target_batch, target_label)

                    # calculate the loss and back prop.
                    cur_loss = self.get_loss(encoder_outputs,
                                             decoder_outputs,
                                             input_batch[:,:,0:6].transpose(0, 1),
                                             target_label.transpose(0, 1),
                                             val_only=True)

                    smape_loss = self.smape_loss(encoder_outputs,
                                                 decoder_outputs,
                                                 input_batch[:,:,0:6].transpose(0, 1),
                                                 target_label.transpose(0, 1),
                                                 val_only=True)
                    cur_loss.backward()                     
                    # logging
                    self.global_step += 1
                    self.tensorboard_log(cur_loss.data[0], smape_loss)
                    if self.global_step % 5 == 0:
                        print("Step: %d, Mse Loss : %4.4f, SMAPE Loss : %f" %(self.global_step, cur_loss.data[0], smape_loss), end='\t')
                    # self.save_model()
                        self.validation(valid_data, batch_size=72, time_lag=time_lag)

                    # optimize
                    self.optimizer.step()

        self.save_model()

    def smape_loss(self, encoder_outputs, decoder_outputs, input_batch, target_batch, val_only=False):
        concat_predict = decoder_outputs.data.cpu().numpy()
        #print("C", concat_predict.shape) # 32, 48, 3
        target_batch = target_batch.data.cpu().numpy()
        #print("D", target_batch.shape)
        concat_label = target_batch[:, -48:, :] # B, T, 3
        #print("E", concat_label.shape)

        #print("P", concat_predict)
        #print("L", concat_label)
        norm = float(concat_predict.shape[0])
        loss = 2 *  (np.abs(concat_predict - concat_label)) /  (np.abs(concat_predict) + concat_label) # B, T, 3
        loss = loss.sum(1)
        loss = loss.mean(-1).mean(-1)
        return (loss / norm)

    def get_loss(self, encoder_outputs, decoder_outputs, input_batch, target_batch, val_only=False):
        #print("E", encoder_outputs.size())
        #print("D", decoder_outputs.size())
        #print("T", target_batch.size())
        loss = self.criterion(torch.cat((encoder_outputs, decoder_outputs), dim=1), target_batch[:, :, [0,1,4]])
        # for i in range(concat_label.size(1)):
        #     i_timestep_predict = concat_predict[:,i,:].contiguous().view(concat_label.size(0),-1)
        #     i_timestep_label = concat_label[:,i,:].contiguous().view(concat_label.size(0),-1)
        #     loss += self.criterion(i_timestep_predict, i_timestep_label)
        return loss

    def validation(self, valid_data, batch_size, time_lag):
        total_mse_loss = 0
        total_smape_loss = 0
        number_of_batch =0
        self.model.eval()
        encoder_batches, decoder_batches, decoder_labels = self.data_transformer.create_mini_batch(valid_data, batch_size=batch_size, window_size=self.window_size, time_lag=time_lag)
        for input_batch, target_batch, target_label in self.data_transformer.variables_generator(encoder_batches, decoder_batches, decoder_labels):
            encoder_outputs, decoder_outputs = self.model(input_batch, target_batch, target_label, validate=True)

            # calculate the loss and back prop.
            cur_mse_loss = self.get_loss(encoder_outputs,
                                        decoder_outputs,
                                        input_batch[:,:,0:6].transpose(0, 1),
                                        target_label.transpose(0, 1),
                                        val_only=True).data[0]
            smape_loss = self.smape_loss(encoder_outputs,
                                            decoder_outputs,
                                            input_batch[:,:,0:6].transpose(0, 1),
                                            target_label.transpose(0, 1),
                                            val_only=True)
            total_mse_loss += (cur_mse_loss * input_batch.size(1))  # Mulitply Batch number  input_batch size: T * B * H 
            total_smape_loss += (smape_loss * input_batch.size(1))
            number_of_batch += input_batch.size(1)
        total_mse_loss /= number_of_batch
        total_smape_loss /= number_of_batch
        self.tensorboard_log(total_mse_loss, total_smape_loss, valid=True)
        print("Validation, Mse Loss : %4.4f, SMAPE Loss : %f" %(total_mse_loss, total_smape_loss))


    
    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint_name)
        print("Model has been saved as %s.\n" % self.checkpoint_name)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name))
        print("Pretrained model has been loaded.\n")

    def tensorboard_log(self, mse_loss, smape_loss, valid= False):
        info = {
                'mse_loss': mse_loss,
                'smape_loss': smape_loss
            }
        if not valid:
            
            for tag, value in info.items():
                self.train_logger.scalar_summary(tag, value, self.global_step)

        else:
            for tag, value in info.items():
                self.valid_logger.scalar_summary(tag, value, self.global_step)


class RandomTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super(RandomTrainer, self).__init__(*args, **kwargs)

    def train(self, num_epochs, batch_size, window_size, pretrained= False, valid_portion= 0.8, time_lag=10, valid_batch_size=128):
        
        self.window_size = window_size
        if pretrained:
            self.load_model()

        all_station_train_input, all_station_dec_input, all_station_train_label, all_station_valid_input, all_station_valid_dec_input, all_station_valid_label = self.data_transformer.prepare_all_station_data(
            all_station_data=self.data_transformer.every_station_data,
            training_portion=valid_portion,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            window_size=self.window_size,
            time_lag=time_lag,
            shuffle_input=True)

        valid_total_size = len(all_station_valid_input)
        print("Total validation size:", valid_total_size * batch_size)
        print("Training on %d, Validation with 1600." % (60 * batch_size))
        self.global_step = 0
        for epoch in range(1, num_epochs + 1):
            print("In %d epoch" %(epoch))
            for input_batch, target_batch, target_label in self.data_transformer.variables_generator(all_station_train_input, all_station_dec_input, all_station_train_label):
                self.optimizer.zero_grad()
                self.model.train()
                encoder_outputs, decoder_outputs = self.model(input_batch, target_batch, target_label)

                # calculate the loss and back prop.
                cur_loss = self.get_loss(encoder_outputs,
                                            decoder_outputs,
                                            input_batch[:,:,0:6].transpose(0, 1),
                                            target_label.transpose(0, 1),
                                            val_only=True)

                smape_loss = self.smape_loss(encoder_outputs,
                                            decoder_outputs,
                                            input_batch[:,:,0:6].transpose(0, 1),
                                            target_label.transpose(0, 1),
                                            val_only=True)
                cur_loss.backward()                     
                # logging
                self.global_step += 1
                self.tensorboard_log(cur_loss.data[0], smape_loss)
                if self.global_step % 5 == 0:
                    print("[Epoch #%d] Step: %d, Mse Loss : %4.4f, SMAPE Loss : %f" %(epoch, self.global_step, cur_loss.data[0], smape_loss), end='\t')
                # self.save_model()
                    all_id = np.random.permutation(valid_total_size)
                    valid_idx = np.random.choice(all_id, 60)
                    #print(valid_idx)
                    self.validation(all_station_valid_input[valid_idx], all_station_valid_dec_input[valid_idx], all_station_valid_label[valid_idx], valid_batch_size, time_lag)

                # optimize
                self.optimizer.step()

            self.save_model()

    def validation(self, valid_inputs, valid_dec_input, valid_labels, batch_size, time_lag):
        total_mse_loss = 0
        total_smape_loss = 0
        number_of_batch =0
        self.model.eval()
        for input_batch, target_batch, target_label in self.data_transformer.variables_generator(valid_inputs, valid_dec_input, valid_labels):
            encoder_outputs, decoder_outputs = self.model(input_batch, target_batch, target_label, validate=True)

            # calculate the loss and back prop.
            cur_mse_loss = self.get_loss(encoder_outputs,
                                        decoder_outputs,
                                        input_batch[:,:,0:6].transpose(0, 1),
                                        target_label.transpose(0, 1),
                                        val_only=True).data[0]
            smape_loss = self.smape_loss(encoder_outputs,
                                            decoder_outputs,
                                            input_batch[:,:,0:6].transpose(0, 1),
                                            target_label.transpose(0, 1),
                                            val_only=True)
            total_mse_loss += (cur_mse_loss * input_batch.size(1))  # Mulitply Batch number  input_batch size: T * B * H 
            total_smape_loss += (smape_loss * input_batch.size(1))
            number_of_batch += input_batch.size(1)
        total_mse_loss /= number_of_batch
        total_smape_loss /= number_of_batch
        self.tensorboard_log(total_mse_loss, total_smape_loss, valid= True)
        print("Validation, Mse Loss : %4.4f, SMAPE Loss : %f" %(total_mse_loss, total_smape_loss))