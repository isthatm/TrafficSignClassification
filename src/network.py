import torch
import torch.nn.functional as F
from torch import nn
import configparser

config = configparser.ConfigParser()
config.read(
    './src/capsnet_config.ini'
    )
# General info
num_class = int(config['network']['num_class'])
img_channels = int(config['network']['image_channels'])

# Primary caps
primary_num_caps = int(config['primary_caps']['num_caps'])
primary_channels = int(config['primary_caps']['channels'])

# Digit caps
digit_num_caps = num_class
digit_channels = int(config['digit_caps']['channels'])
num_iterations = int(config['network']['num_routing_iter'])

# Loss hyper params
m_plus = float(config['loss']['m_plus'])
m_minus = float(config['loss']['m_minus'])
lmbd = float(config['loss']['lambda'])
regularization_factor = float(config['loss']['regularization_factor'])

def squash(vector, axis=-1 ,epsilon=1e-7, squash=True):
        """
        normalize the length of the vector by 1 (squash)

        :param vector: the muliplication of coupling coefs and prediction vectors sum [ c(ij)u^(j|i) ]
        :param axis: the axis that would not be reduced
        :param epsilon: a workaround to prevent devision by zero
        :param squash: if squash is False, the length of the vector is calculated

        """
        squared_norm = torch.sum(torch.square(vector), dim=axis, 
                                keepdim=True)
        safe_norm = torch.sqrt(squared_norm + epsilon)

        if squash:
            squash_factor = squared_norm / (1. + squared_norm)
            unit_vector = vector / safe_norm
            return squash_factor * unit_vector
        else:
            return safe_norm

class CapsLayers(nn.Module):
    """ 
    Args:

    :param num_conv: number of filters/convolutional unit per capsule (dimension of a capsule)
    :param num_capsules: number of primary/digit caps
    :param num_routing_nodes: number of possible u(i), 
                            set to -1 if it's not a secondary capsule layer
    :param in_channels: output convolutional layers of the prev layer
    :param out_channels: output convolutional layers of the current layer
    """
    def __init__(self, num_capsules: int, in_channels: int, out_channels: int, 
                 kernel_size=None, stride=None, num_routing_nodes=None ,num_iterations=None):
        super(CapsLayers, self).__init__()
        self.num_capsules = num_capsules
        
        self.num_iterations = num_iterations
        self.num_routing_nodes = num_routing_nodes
        if num_routing_nodes is not None:
            self.weights = nn.Parameter(torch.randn(
                                        self.num_routing_nodes, num_capsules, out_channels, in_channels))
            self.b = nn.Parameter(torch.zeros(
                                    self.num_routing_nodes, num_capsules, 1, 1))
        else:
            self.primary_caps = nn.ModuleList(nn.Conv2d(in_channels, out_channels, kernel_size, stride) 
                                                    for _ in range(num_capsules))

    def forward(self, inputs):
        """
        Feed foward function for non-reconstruction layer
        :param inputs: 
            for the primary caps, the inputs are convolutional layer pixels
            for digit caps, the inputs are n-D vectors from a primary cap
                where n is the # of filters for one capsule  
            Required Paramteters:
            prior_logits(b) 
            primary layer prediction (requires u-layer 1 ouput, Weights)
        """
        if self.num_routing_nodes is not None:
            weights = self.weights[None, :, :, :].tile(inputs.size(0), 1, 1, 1, 1)
            b_ij = self.b[None, :, :, :].tile(inputs.size(0), 1, 1, 1, 1)
            inputs = inputs.tile(1, 1, self.num_capsules, 1, 1)
            
            # u_hat = [batch, num_routing_nodes, # digit_caps, digit_caps_dims, 1]
            u_hat = weights @ inputs 

            for i in range(self.num_iterations):
                c_ij = F.softmax(b_ij, dim=2)
                outputs = squash((c_ij*u_hat).sum(dim=1, keepdim=True))

                if i < self.num_iterations - 1 :
                    # v_j OR outputs = [batch, 1 -> num_routing_nodes, num_digit_caps, digit_caps_dims, 1 )]
                    b_ij +=  (b_ij + torch.transpose(u_hat, 3, 4) @ outputs.tile(1, self.num_routing_nodes, 1, 1, 1))

        else:
            outputs = [
                capsule(inputs)[:, None, :, :, :].permute(0, 1, 3, 4, 2) for capsule in self.primary_caps]
            outputs = torch.cat(outputs, dim=1)
            # u(i) = [batch, num_prim_caps*prim_caps_2D_size, prim_caps_output_dimension]
            outputs = outputs.view(outputs.size(0), -1, outputs.size(4)) 
            outputs = squash(outputs)[:, :, None, :, None]
        
        # outputs = [batch, 1, num_digit_caps/num_class, digit_caps_dims, 1]
        return outputs            


class CapsNet(nn.Module):
    """
        This class contains the full CapsNet architecture:
        Convolutional -> primary capsules -> digit capsules -> (3) fully connected
    """
    def __init__(self):
        """
        Params: 
        `inputs`: a 4D tensor (grey scale or RGB)
        """
        super(CapsNet, self).__init__()

        self.conv_1 = nn.Conv2d(img_channels, 256, 
                                kernel_size=9, stride=1)
        self.primary_caps = CapsLayers(primary_num_caps, 256, primary_channels, 
                                        kernel_size=5, stride=2)
        self.digit_caps = CapsLayers(digit_num_caps, primary_channels, digit_channels, 
                                     num_routing_nodes=10*10*primary_num_caps, num_iterations=num_iterations)
        self.grey_scale_decoder = nn.Sequential(
                nn.Linear(digit_channels*digit_num_caps, 576),
                nn.ReLU(),
                nn.Linear(576, 1600),
                nn.ReLU(),
                nn.Linear(1600, 1024),
                nn.Sigmoid(),
        )
        self.linear_trans = nn.Linear(digit_channels*digit_num_caps, 400)
        self.RGB_decoder = nn.Sequential(
                nn.Upsample(size=(8, 8)),
                nn.Conv2d(16, 4, 3, padding='same'),
                nn.ReLU(),

                nn.Upsample(scale_factor=2),
                nn.Conv2d(4, 8, 3, padding='same'),
                nn.ReLU(),

                nn.Upsample(scale_factor=2),
                nn.Conv2d(8, 16, 3, padding='same'),
                nn.ReLU(),

                nn.Conv2d(16, 3, 3, padding='same'),
                nn.Sigmoid()
        )

    def forward(self, images, labels=None): # labels should be applied `one_hot` function
        self.batch_size = images.size()[0]

        conv_1_ouputs = F.relu(self.conv_1(images))
        primary_caps_outputs =  self.primary_caps(conv_1_ouputs) 
        digit_caps_outputs = self.digit_caps(primary_caps_outputs).squeeze(1)
        
        assert list(digit_caps_outputs.size()) == [images.size()[0], num_class, digit_channels, 1]
               
        v_norm = squash(digit_caps_outputs, axis=-2, squash=False) # [batch, num_classes, 1, 1]
        v_prob = F.softmax(v_norm, dim=1)

        self.img = images
        self.v_norm = v_norm

        idx = torch.zeros(images.size()[0], 1, 1)
        # Masking
        if labels is None: # Testing mode
            _, idx = torch.max(v_prob, dim=1)
            labels = torch.eye(num_class).index_select(dim=0, index = idx.squeeze())

        # masked_v = [batch_size, digit_channels*classes])
        masked_v = (labels[:, :, None, None] * digit_caps_outputs).view(images.size(0), -1)
        
        # Reconstruction
        if images.size()[1] == 1:
            reconstructed_img = self.grey_scale_decoder(masked_v).view(self.batch_size, 32, 32) # [batch, 32x32]
        else:
  
            fc_out = self.linear_trans(masked_v).view(self.batch_size, 16, 5 ,5) # masked_v
            reconstructed_img = self.RGB_decoder(fc_out) # [batch, channel, height, width]

        return idx, reconstructed_img 
        
    def loss_fn(self, reconstructed_img, labels):
        # Margin loss
        max_1 = F.relu(m_plus - self.v_norm)
        max_2 = F.relu(self.v_norm - m_minus)
        T_k = labels[:, :, None, None]
        
        L_k = T_k * torch.square(max_1) + lmbd * (1 - T_k) * torch.square(max_2)

        assert L_k.size() == self.v_norm.size()
        margin_loss = L_k.sum(dim=1).mean()
        
        # Reconstruction loss
        reconstruction_loss_obj = nn.MSELoss()

        # original_img = [batch size, flatten image (pixels are flatten into arrays)]
        original_img = self.img.contiguous().view(self.batch_size, -1)
        reconstructed_img = reconstructed_img.view(self.batch_size, -1)

        reconstruction_loss = reconstruction_loss_obj(reconstructed_img, original_img)
        total_loss = margin_loss + regularization_factor * reconstruction_loss

        return margin_loss, reconstruction_loss, total_loss