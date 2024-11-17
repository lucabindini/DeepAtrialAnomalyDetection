import torch
from torch import nn
import math


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_filters,
                                     kernel_size=kernel_size,
                                     padding='same')
        self.bn1 = torch.nn.BatchNorm1d(num_filters)
        self.conv2 = torch.nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_filters,
                                     kernel_size=kernel_size,
                                     stride=2,
                                     padding=(kernel_size-1)//2)
        self.bn2 = torch.nn.BatchNorm1d(num_filters)
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = torch.nn.ReLU()(h)
        h = self.conv2(x)
        h = self.bn2(h)
        h = torch.nn.ReLU()(h)
        return h
        
class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size):
        super().__init__()
        # self.convt1 = torch.nn.ConvTranspose1d(in_channels=in_channels, 
        #                                         out_channels=num_filters, 
        #                                         kernel_size=kernel_size,
        #                                         padding=0,
        #                                         stride=2)
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, 
                                                out_channels=num_filters, 
                                                kernel_size=kernel_size, padding='same')

        self.upsampling = torch.nn.Upsample(scale_factor=2)
        

        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        self.bn2 = torch.nn.BatchNorm1d(num_filters)
    def forward(self, x):
        h = self.upsampling(x)
        h = self.bn1(h)
        h = torch.nn.ReLU()(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = torch.nn.ReLU()(h)
        return h
    
class MemoryUnit(torch.nn.Module):
    def __init__(self, mem_dim, fea_dim):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = torch.nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def hard_shrink_relu(self, input, lambd=0, epsilon=1e-12):
        output = (torch.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output

    def forward(self, input):
        input = torch.flatten(input, start_dim=1)
        att_weight = torch.nn.functional.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = torch.softmax(att_weight, dim=1)  # TxM
        att_weight = self.hard_shrink_relu(att_weight, lambd=0.05)
        att_weight = torch.nn.functional.normalize(att_weight, p=1, dim=1)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = torch.nn.functional.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output.unsqueeze(-1), 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


class Encoder(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size):
        super().__init__()
        self.e1 = EncoderBlock(in_channels, num_filters, kernel_size)
        self.e2 = EncoderBlock(num_filters, num_filters*2, kernel_size)
        self.e3 = EncoderBlock(num_filters*2, num_filters*4, kernel_size)
        self.e4 = EncoderBlock(num_filters*4, num_filters*8, kernel_size)
        self.e5 = EncoderBlock(num_filters*8, num_filters*16, kernel_size)
        self.e6 = EncoderBlock(num_filters*16, num_filters*16, kernel_size)
    def forward(self, x):
        h = self.e1(x)
        h = self.e2(h)
        h = self.e3(h)
        h = self.e4(h)
        h = self.e5(h)
        h = self.e6(h)
        return h

class Decoder(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size):
        super().__init__()
        self.d1 = DecoderBlock(in_channels, num_filters*16, kernel_size)
        self.d2 = DecoderBlock(num_filters*16, num_filters*16, kernel_size)
        self.d3 = DecoderBlock(num_filters*16, num_filters*8, kernel_size)
        self.d4 = DecoderBlock(num_filters*8, num_filters*4, kernel_size)
        self.d5 = DecoderBlock(num_filters*4, num_filters*2, kernel_size)
        self.d6 = DecoderBlock(num_filters*2, num_filters, kernel_size)
    def forward(self, x):
        h = self.d1(x)
        h = self.d2(h)
        h = self.d3(h)
        h = self.d4(h)
        h = self.d5(h)
        h = self.d6(h)
        return h


class Autoencoder(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, bottleneck_dim, with_memory=False):
        super().__init__()
        self.with_memory = with_memory
        self.encoder = Encoder(in_channels, num_filters, kernel_size)
        self.decoder = Decoder(bottleneck_dim, num_filters, kernel_size)
        if with_memory:
            self.memory = MemoryUnit(bottleneck_dim, bottleneck_dim)
        self.bottleneck = torch.nn.Conv1d(in_channels=num_filters*16, 
                                          out_channels=bottleneck_dim,
                                          kernel_size=1,
                                          padding="same")
        # self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=)
        self.final = torch.nn.Conv1d(in_channels=num_filters,
                                     out_channels=1,
                                     kernel_size=1,
                                     padding="same")
        
    def forward(self, x):
        h = self.encoder(x)
        h = self.bottleneck(h)
        if self.with_memory:
            mem = self.memory(h)
            h = mem['output']
            self.att_weight = mem['att']
        h = self.decoder(h)
        h = self.final(h)
        return h
    

class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, bottleneck_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, num_filters, kernel_size)
        self.decoder = Decoder(bottleneck_dim, num_filters, kernel_size)
        self.bottleneck = torch.nn.Conv1d(in_channels=num_filters*16, 
                                          out_channels=bottleneck_dim,
                                          kernel_size=1,
                                          padding="same")
        self.mu = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.logvar = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.final = torch.nn.Conv1d(in_channels=num_filters,
                                     out_channels=1,
                                     kernel_size=1,
                                     padding="same")
        
    def forward(self, x):
        h = self.encoder(x)
        h = self.bottleneck(h)

        h = torch.nn.Flatten()(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        # Sample from latent distribution from encoder
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z_reparametrized = mu + std*epsilon
        z = z_reparametrized.unsqueeze(-1)

        h = self.decoder(z)
        h = self.final(h)
        return h, mu, logvar
    

class DeepSVDDEncoder(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, bottleneck_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, num_filters, kernel_size)
        self.bottleneck = torch.nn.Conv1d(in_channels=num_filters*16, 
                                          out_channels=bottleneck_dim,
                                          kernel_size=1,
                                          padding="same")

    def forward(self, x):
        h = self.encoder(x)
        h = self.bottleneck(h)
        return h
    
