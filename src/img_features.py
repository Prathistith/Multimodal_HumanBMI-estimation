import torch
import gdown,os
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils import FeatureExtractor
from utils import get_model

class VGGFace(nn.Module):
    def __init__(self):
        super(VGGFace, self).__init__()
        self.features = nn.ModuleDict(OrderedDict(
            {
                # === Block 1 ===
                'conv_1_1': nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                'relu_1_1': nn.ReLU(inplace=True),
                'conv_1_2': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                'relu_1_2': nn.ReLU(inplace=True),
                'maxp_1_2': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 2 ===
                'conv_2_1': nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                'relu_2_1': nn.ReLU(inplace=True),
                'conv_2_2': nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                'relu_2_2': nn.ReLU(inplace=True),
                'maxp_2_2': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 3 ===
                'conv_3_1': nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                'relu_3_1': nn.ReLU(inplace=True),
                'conv_3_2': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_2': nn.ReLU(inplace=True),
                'conv_3_3': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_3': nn.ReLU(inplace=True),
                'maxp_3_3': nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                # === Block 4 ===
                'conv_4_1': nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                'relu_4_1': nn.ReLU(inplace=True),
                'conv_4_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_2': nn.ReLU(inplace=True),
                'conv_4_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_3': nn.ReLU(inplace=True),
                'maxp_4_3': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 5 ===
                'conv_5_1': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_1': nn.ReLU(inplace=True),
                'conv_5_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_2': nn.ReLU(inplace=True),
                'conv_5_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_3': nn.ReLU(inplace=True),
                'maxp_5_3': nn.MaxPool2d(kernel_size=2, stride=2)
            }))

        self.fc = nn.ModuleDict(OrderedDict(
            {
                'fc6': nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                #'fc6-relu': nn.ReLU(inplace=True),
                #'fc6-dropout': nn.Dropout(p=0.5),
                #'fc7': nn.Linear(in_features=4096, out_features=4096),
                #'fc7-relu': nn.ReLU(inplace=True),
                #'fc7-dropout': nn.Dropout(p=0.5),
                #'fc8': nn.Linear(in_features=4096, out_features=2622),
            }))

    def forward(self, x):
        # Forward through feature layers
        for k, layer in self.features.items():
            x = layer(x)

        # Flatten convolution outputs
        x = x.view(x.size(0), -1)

        # Forward through FC layers
        for k, layer in self.fc.items():
            x = layer(x)

        return x

def load_pretrained_vggface(args):
    '''Build VGGFace model and load pre-trained weights'''
    model = VGGFace()
    if not os.path.exists(args['vggface_path']):
        print("Downloading vggface.pth file .....",sep=" ",end="",flush=True)
        gdown.download("https://drive.google.com/uc?export=download&confirm=c8KK&id=1-XjP7EPy8rwx9ivr4g7fZ9cTKekhAJ6v",args['vggface_path'],False)
        print("completed !\n")

    pretrained_dict = torch.load(args['vggface_path'],map_location=args['device'])
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    
    for param in model.parameters():
        param.requires_grad = False

    return model


class FaceFeatures(nn.Module):
    def __init__(self,args):
        super(FaceFeatures,self).__init__()
        self.base_model =  load_pretrained_vggface(args)

        self.linear1 = nn.Linear(4096, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dp1 = nn.Dropout(p=args['dropout'])

        self.linear2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dp2 = nn.Dropout(p=args['dropout'])

        self.linear3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self,x):
        x = self.base_model(x)
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)

        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = F.relu(self.bn3(self.linear3(x)))

        return x


class BodyFeatures(nn.Module):
    def __init__(self,args):
        super(BodyFeatures,self).__init__()
        self._args = args
        self.features =  get_model(args['body_model'])

        self.linear1 = nn.Linear(2048, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dp1 = nn.Dropout(p=args['dropout'])

        self.linear2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dp2 = nn.Dropout(p=args['dropout'])

        self.linear3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self,x):
        x = self.features(x)
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)

        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = F.relu(self.bn3(self.linear3(x)))

        return x
