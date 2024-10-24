import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import math
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure that the input tensor size is [batch_size, channel, width, heigh]
# Ensure that the labels are int64
# Ensure that the ouput of the classfier matches the number of classes

supportedArch = [
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

label_name_mapping = {
    0: "3-junction",
    1: "One-way road",
    2: "Side to obe followed",
    3: "Cross road",
    4: "Intersection with Uncontrolled Road",
    5: "Dangerous turn",
    6: "No Left turn",
    7: "Bus stop",
    8: "Roundabout",
    9: "No parking and stopping",
    10: "U-turn",
    11: "Lane-allocation",
    12: "No left turn for motorcycles",
    13: "Slow Down",
    14: "No Trucks Allowed",
    15: "Narrow Road on the Right",
    16: "No Passenger Cars and Trucks",
    17: "Height Limit",
    18: "No U-Turn",
    19: "No U-Turn and No Right Turn",
    20: "No Cars Allowed",
    21: "Narrow Road on the Left",
    22: "Uneven Road",
    23: "No Two or Three-wheeled Vehicles",
    24: "Customs Checkpoint",
    25: "Motorcycles Only",
    26: "Obstacle on the Road",
    27: "Children Presence",
    28: "Trucks and Containers",
    29: "No Motorcycles Allowed",
    30: "Trucks Only",
    31: "Road with Surveillance Camera",
    32: "No Right Turn",
    33: "Series of Dangerous Turns",
    34: "No Containers Allowed",
    35: "No Left or Right Turn",
    36: "No Straight and Right Turn",
    37: "Intersection with T-Junction",
    38: "Speed limit (50km/h)",
    39: "Speed limit (60km/h)",
    40: "Speed limit (80km/h)",
    41: "Speed limit (40km/h)",
    42: "Left Turn",
    43: "Low Clearance",
    44: "Other Danger",
    45: "Go Straight",
    46: "No Parking",
    47: "No Left or U-turn",
    48: "No U-Turn for Cars",
    49: "Level Crossing with Barriers"
}

class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, vgg_blocks):
        super(VGG, self).__init__()
        self.layers = nn.ModuleList(vgg_blocks)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1408, 512), # nn.Linear(1408, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 50),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
      output = []
      tmpInput = x
      for layerIdx, layer in enumerate(self.layers):
        current_output = layer(tmpInput)
        output.append(current_output)
        tmpInput = current_output

      output_m1 = torch.cat((self.maxPool (output[1]), output[2]), 1)
      output_m2 = torch.cat((self.maxPool (output_m1), output[3]), 1)
      output_m3 = torch.cat((self.maxPool (output_m2), output[4]), 1)

      ouput_pre_classification = output_m3.view(output_m3.size(0), -1)
      #ouput_pre_classification = output[-1].view(output[-1].size(0), -1)
      #print(ouput_pre_classification.size())
      final_ouput = self.classifier(ouput_pre_classification)
      return final_ouput

    def vgg11():
        """VGG 11-layer model (configuration "A")"""
        return VGG(make_layers(cfg['A']))


    def vgg11_bn():
        """VGG 11-layer model (configuration "A") with batch normalization"""
        return VGG(make_layers(cfg['A'], batch_norm=True))


    def vgg13():
        """VGG 13-layer model (configuration "B")"""
        return VGG(make_layers(cfg['B']))


    def vgg13_bn():
        """VGG 13-layer model (configuration "B") with batch normalization"""
        return VGG(make_layers(cfg['B'], batch_norm=True))


    def vgg16():
        """VGG 16-layer model (configuration "D")"""
        return VGG(make_layers(cfg['D']))


    def vgg16_bn():
        """VGG 16-layer model (configuration "D") with batch normalization"""
        return VGG(make_layers(cfg['D'], batch_norm=True))


    def vgg19():
        """VGG 19-layer model (configuration "E")"""
        return VGG(make_layers(cfg['E']))


    def vgg19_bn():
        """VGG 19-layer model (configuration 'E') with batch normalization"""
        return VGG(make_layers(cfg['E'], batch_norm=True))

def make_layers(cfg, batch_norm=False):
    layers = []
    vgg_blocks = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            vgg_sequential = nn.Sequential(*layers)
            vgg_blocks.append(vgg_sequential)
            layers = [] # empty the current block
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return vgg_blocks

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}