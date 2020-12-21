import torch.nn as nn
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import math
from torch.autograd import Variable
import torch.nn.functional as F


def get_norm(norm):
    no_norm = lambda x, dim: x
    if norm == 'weight':
        norm_layer = weight_norm
    elif norm == 'batch':
        norm_layer = nn.BatchNorm1d
    elif norm == 'layer':
        norm_layer = nn.LayerNorm
    elif norm == 'none':
        norm_layer = no_norm
    else:
        print("Invalid Normalization")
        raise Exception("Invalid Normalization")
    return norm_layer


def get_act(act):
    if act == 'ReLU':
        act_layer = nn.ReLU
    elif act == 'LeakyReLU':
        act_layer = nn.LeakyReLU
    elif act == 'PReLU':
        act_layer = nn.PReLU
    elif act == 'RReLU':
        act_layer = nn.RReLU
    elif act == 'ELU':
        act_layer = nn.ELU
    elif act == 'SELU':
        act_layer = nn.SELU
    elif act == 'Tanh':
        act_layer = nn.Tanh
    elif act == 'Hardtanh':
        act_layer = nn.Hardtanh
    elif act == 'Sigmoid':
        act_layer = nn.Sigmoid
    else:
        print("Invalid activation function")
        raise Exception("Invalid activation function")
    return act_layer



class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, dropout, norm, act):
        super(FCNet, self).__init__()

        norm_layer = get_norm(norm)
        act_layer = get_act(act)

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(norm_layer(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(act_layer())
            layers.append(nn.Dropout(p=dropout))
        layers.append(norm_layer(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(act_layer())
        layers.append(nn.Dropout(p=dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



class GTH(nn.Module):
    """Simple class for Gated Tanh
    """
    def __init__(self, in_dim, out_dim, dropout, norm, act):
        super(GTH, self).__init__()

        self.nonlinear = FCNet([in_dim, out_dim], dropout= dropout, norm= norm, act= act)
        self.gate = FCNet([in_dim, out_dim], dropout= dropout, norm= norm, act= 'Sigmoid')

    def forward(self, x):
        x_proj = self.nonlinear(x)
        gate = self.gate(x)
        x_proj = x_proj*gate
        return x_proj
        


class MHAtt(nn.Module):
    def __init__(self, HIDDEN_SIZE, DROPOUT_R = 0.1):
        super(MHAtt, self).__init__()
        self.MULTI_HEAD = 2
        self.linear_v = nn.Linear(HIDDEN_SIZE, self.MULTI_HEAD  * HIDDEN_SIZE)
        self.linear_k = nn.Linear(HIDDEN_SIZE, self.MULTI_HEAD  * HIDDEN_SIZE)
        self.linear_q = nn.Linear(HIDDEN_SIZE, self.MULTI_HEAD  * HIDDEN_SIZE)
        self.linear_merge = nn.Linear(self.MULTI_HEAD  * HIDDEN_SIZE, HIDDEN_SIZE)
        
        self.HIDDEN_SIZE_HEAD = HIDDEN_SIZE
        self.HIDDEN_SIZE = HIDDEN_SIZE

        self.dropout = nn.Dropout(DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
             self.MULTI_HEAD  * self.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class STRNN(nn.Module):

    # INPUT a list of seq [ [#neighbour, #time, feat], []...] [node_history]  batch, #time, feat
    def __init__(self, input_dim1, input_dim2, num_hid, rnn_type='LSTM'):
        """Module for question embedding
        """
        super(STRNN, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        norm_layer = get_norm('weight')
        self.rnn1 = rnn_cls( input_dim1 , num_hid, 1, bidirectional=False, dropout=0.1, batch_first=True)
        self.rnn2 = rnn_cls( input_dim2 , num_hid, 1, bidirectional=False, dropout=0.1, batch_first=True)
        self.num_neighbor = 10
        self.num_hid = num_hid
        self.rnn_type = rnn_type
        self.mhatt1 = MHAtt(num_hid)
        self.linear1 = nn.Linear(num_hid, input_dim1)
        self.linear2 = nn.Linear(input_dim2, input_dim2) 

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (1, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, input_seqs , node_history_):
        # # INPUT a list of seq [ [#neighbour, #time, feat], []...] [node_history]  batch, #time, feat
        # q_emb : [batch , q_dim ] 
        # c_emb : [batch * 5, 14 , c_dim]
        # v_emb : [batch , v_dim]
        
        batch_size = len(input_seqs)
        time_step = len(input_seqs[0][0])
        num_features = len(input_seqs[0][0][0])
        node_history = node_history_.detach().clone()
        node_history[torch.isnan(node_history)] = 0.
        
        input_tensor = torch.zeros((batch_size, self.num_neighbor, time_step, num_features)).float()
        for i in range(batch_size):
            input_tensor[i][:len(input_seqs[i])] = input_seqs[i][:self.num_neighbor]
        
        hidden1 = self.init_hidden(batch_size * self.num_neighbor)
        output1, hidden1 = self.rnn1(input_tensor.view(batch_size * self.num_neighbor, time_step, -1) , hidden1)
        output1 = output1.view(batch_size, self.num_neighbor, time_step, -1)
        output1 = output1.transpose(1, 2).reshape(batch_size * time_step, self.num_neighbor, -1)

        hidden2 = self.init_hidden(batch_size)
        output2, hidden2 = self.rnn2(node_history.view(batch_size, time_step, -1) , hidden2)
        output2 = output2.view(batch_size, 1, time_step, -1)
        output2 = output2.transpose(1, 2).reshape(batch_size * time_step, 1, -1)

        atted_output = self.mhatt1( output1,  output1, output2, None)
        atted_output = atted_output.view(batch_size, time_step, -1)

        joint_history = torch.cat([self.linear1(atted_output), self.linear2(node_history)], dim=-1)
        return joint_history

if __name__ == '__main__':
    model = STRNN(8, 8 ).cuda()
    input_seqs = [ torch.rand(2, 3, 8).float().cuda(), torch.rand(7, 3, 8).float().cuda(), torch.rand(11, 3, 8).float().cuda() ]
    node_history = torch.rand(3, 3, 8).float().cuda()
    aaa = model(input_seqs, node_history)
    aaaa = 0.

