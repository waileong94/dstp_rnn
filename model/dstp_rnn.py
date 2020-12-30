from torch.autograd import Variable
import torch
from torch import cuda
# torch.cuda.is_available()
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau


class Encoder(nn.Module):
    

    def __init__(self, T ,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):

        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T
       
        
        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.encoder_num_hidden)

        self.encoder_lstm2 = nn.LSTM(
            input_size=self.input_size, hidden_size=self.encoder_num_hidden)

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1, out_features=1, bias=True) #1033
        
        # W_s[h_{t-1} ; s_{t-1}] + U_s[x^k ; y^k]
        self.encoder_attn2 = nn.Linear(
            in_features=2 * self.encoder_num_hidden + 2*self.T - 2, out_features=1, bias=True)

        
    def forward(self, X ,y_prev):
        """forward.

        Args:
            X

        """
        X_tilde = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())
        
        X_tilde2 = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())

        X_encoded2 = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

      
        # hidden, cell: initial states with dimention hidden_size

        h_n = self._init_states(X)
        s_n = self._init_states(X)

        hs_n = self._init_states(X)
        ss_n = self._init_states(X)
        # y_prev = y_prev.view()
       
        y_prev = y_prev.view(len(X) , self.T-1 ,1)
        
        # print(h_n.size())  # 1 233 512
        # print(s_n.size())
   
        
        for t in range(self.T - 1):
            #Phase one attention
            # batch_size * input_size * (2*hidden_size + T - 1)
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2), #233 363 1033
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)
            
           
       
            # test = x.view(-1, self.encoder_num_hidden * 2 + self.T - 1)
            # print(test.size()) #84579 1033
            
            x = self.encoder_attn( #84579 1  
                x.view(-1, self.encoder_num_hidden * 2 + self.T - 1)) 
         
            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size))# 233x363
            
            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :]) #233x363
            # print(x_tilde.size())
                
            # encoder LSTM 
            self.encoder_lstm.flatten_parameters()
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]
            

            #Phase two attention from DSTP-RNN Paper

            x2 = torch.cat((hs_n.repeat(self.input_size, 1, 1).permute(1, 0, 2), #233 363 1042
                           ss_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1),
                           y_prev.repeat(1, 1, self.input_size).permute(0, 2, 1)), dim=2)
            
            x2 = self.encoder_attn2( 
                x2.view(-1, self.encoder_num_hidden * 2 + 2*self.T - 2)) 
            
            alpha2 = F.softmax(x2.view(-1, self.input_size))# 233x363
            
            x_tilde2 = torch.mul(alpha2, x_tilde)
            

            self.encoder_lstm2.flatten_parameters()
            _, final_state2 = self.encoder_lstm2(
                x_tilde2.unsqueeze(0), (hs_n, ss_n))
            hs_n = final_state2[0]
            ss_n = final_state2[1]
            # print(x_tilde2.size())
            X_tilde2[:, t, :] = x_tilde2
            X_encoded2[:, t, :] = hs_n       

        return X_tilde2 , X_encoded2

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = Variable(X.data.new(
            1, X.size(0), self.encoder_num_hidden).zero_())
        return initial_states
        
class Decoder(nn.Module):


    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
                                        nn.Tanh(),
                                        nn.Linear(encoder_num_hidden, 1))
        self.lstm_layer = nn.LSTM(
            input_size=1, hidden_size=decoder_num_hidden)
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final_price = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)
        

        self.fc.weight.data.normal_()

    def forward(self, X_encoed, y_prev):
        """forward."""
        d_n = self._init_states(X_encoed)
        c_n = self._init_states(X_encoed)

        for t in range(self.T - 1):

            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoed), dim=2)

            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1))
            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoed)[:, 0, :]
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))
                # 1 * batch_size * decoder_num_hidden
                d_n = final_states[0]
                # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]
        # Eqn. 22: final output
        final_temp_y = torch.cat((d_n[0], context), dim=1)
        y_pred_price = self.fc_final_price(final_temp_y)
       
        return y_pred_price
    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = X.data.new(
            1, X.size(0), self.decoder_num_hidden).zero_()
        return initial_states
        
class DSTP_rnn(nn.Module):
    

    def __init__(self,input_size,T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 learning_rate,
                 weight_decay = 0,
                 learning_rate_decay_step = 100,
                 learning_rate_decay_alpha = 0.99,
                 learning_rate_plateau_alpha = 0.7,
                 learning_rate_plateau_patience = 50,
                 parallel=False):
       
        super().__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        # self.batch_size = batch_size
        self.parallel = parallel
        # self.shuffle = False
        self.T = T
        self.input_size = input_size
        self.weight_decay = weight_decay

        self.Encoder = Encoder(input_size=self.input_size,
                               encoder_num_hidden=encoder_num_hidden,
                               T=T)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T)
        self.Encoder = self.Encoder.cuda()
        self.Decoder = self.Decoder.cuda()
        # Loss function
        self.criterion_price = nn.MSELoss()
        
    


        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate,
                                            weight_decay = self.weight_decay)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate,
                                            weight_decay = self.weight_decay)
        
        self.encoder_step_scheduler = StepLR(self.encoder_optimizer, step_size = learning_rate_decay_step,gamma = learning_rate_decay_alpha)
        self.decoder_step_scheduler = StepLR(self.decoder_optimizer, step_size = learning_rate_decay_step,gamma = learning_rate_decay_alpha)

        self.encoder_plateau_scheduler = ReduceLROnPlateau(self.encoder_optimizer, 'min',factor=learning_rate_plateau_alpha,patience = learning_rate_plateau_patience)
        self.decoder_plateau_scheduler = ReduceLROnPlateau(self.decoder_optimizer, 'min',factor=learning_rate_plateau_alpha,patience = learning_rate_plateau_patience)
        
    def train_forward(self, X, y_prev, y_gt):
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()),Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).cuda())) #cuda
        y_pred_price = self.Decoder(input_encoded, Variable(
            torch.from_numpy(y_prev).type(torch.FloatTensor)).cuda())#cuda

        
        y_true_price = torch.from_numpy(
            y_gt).type(torch.FloatTensor)
        
        y_true_price =y_true_price.view(-1, 1).cuda() #cuda
        
        loss = self.criterion_price(y_pred_price, y_true_price)
        
        loss.backward()
       
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        

        return loss.item()        

