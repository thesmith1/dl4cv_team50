import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def pad(kernel_size, stride, height, width):
    ph = int(np.ceil((max(kernel_size - stride_conv, 0) if (height % stride_conv) == 0 else max(kernel_size - (height % stride_conv), 0))/2.0))
    pw = int(np.ceil((max(kernel_size - stride_conv, 0) if (width % stride_conv) == 0 else max(kernel_size - (width % stride_conv), 0))/2.0))
    return (ph, pw)
        

class AttentionCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), weight_scale=0.001):
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        self.conv1 = nn.Conv2d(channels, 64, 3, padding=pad(3, 1, height, width), stride=1, bias=True)
        self.conv1.weight.data *= weight_scale
        
        self.conv2 = nn.Conv2d(64, 1, 3, padding=pad(3, 1, height, width), stride=1, bias=True)
        self.conv2.weight.data *= weight_scale

        if torch.cuda.is_available():
            self.parameters.cuda()
        
    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        out = None
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
	s = out2.size()
        m = nn.Softmax()
        out = m(out2.view(-1)).view(s)

        return out

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

