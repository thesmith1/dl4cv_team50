"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ph = int(np.ceil((max(kernel_size - stride_conv, 0) if (height % stride_conv) == 0 else max(kernel_size - (height % stride_conv), 0))/2.0))
        pw = int(np.ceil((max(kernel_size - stride_conv, 0) if (width % stride_conv) == 0 else max(kernel_size - (width % stride_conv), 0))/2.0))
        
        self.conv = nn.Conv2d(channels, num_filters, kernel_size, padding=(ph, pw), stride=stride_conv, bias=True)
        self.conv.weight.data *= weight_scale
        
        convw = np.ceil(height/float(stride_conv))
        convh = np.ceil(width/float(stride_conv))
        self.out_size = int(num_filters * np.ceil((convw - pool + 1)/stride_pool) * np.ceil((convh - pool + 1)/stride_pool))        
        self.fc1 = nn.Linear(self.out_size, num_classes)
        
        self.pool = pool
        self.stride_pool = stride_pool
        
        if torch.cuda.is_available():
            self.parameters.cuda()
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        out = None
        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        out1 = F.max_pool2d(F.relu(self.conv(x)), self.pool, stride=self.stride_pool, padding=0)
        out1 = out1.view(-1, self.out_size)
        out = self.fc1(out1)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

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

