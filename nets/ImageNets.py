import torch
import torch.nn as nn
import torch.nn.functional as F

from .VisionNavNet import VisionNavNet

#
# Same networks as Image and Traj models from PAAD (https://github.com/tianchenji/PAAD),
# but where we bring the feature map linear layers into the models, rather than requiring the fusion model to do it (if include_linear=True).
#


class ImageModel(nn.Module):
    def __init__(self, freeze_features, pretrained_file, feature_size=64, include_linear=True):
        """ Image model from PAAD
         
        :param freeze_features: if True, the features in pretrained network are frozen.
        :param pretrained_file: pretrained file to use for VisionNavNet (or None if not pretrained)
        :param feature_size: number of output features (only applicable if include_linear=True)
        :param include_linear: if True, linear layers are included in this model
        """
        super().__init__()
        # load pretrained weights
        vision_nav_net = VisionNavNet()
        if pretrained_file is not None:
            checkpoint = torch.load(pretrained_file)
            vision_nav_net.load_state_dict(checkpoint['state_dict'])

        if freeze_features:
            for param in vision_nav_net.parameters():
                param.requires_grad = False

        self.features = nn.Sequential(*list(vision_nav_net.children())[:-3])
        self.conv_L_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_A_1 = nn.ReLU()
        self.include_linear = include_linear
        if self.include_linear:
            self.linear = nn.Linear(640, feature_size)
            self.relu = nn.ReLU()

    def forward(self, img):
        """ Compute image features
        
        :param img: image (shape: [B, 3, 240, 320])
        :returns: features
        """
        img_features = self.features(img)
        img_features = self.conv_A_1(self.conv_L_1(img_features))
        if self.include_linear:
            img_features = torch.flatten(img_features, 1)
            img_features = self.relu(self.linear(img_features))
        return img_features


class TrajModel(nn.Module):
    def __init__(self, feature_size=64, include_linear=True):
        """ Trajectory image model from PAAD
        
        :param feature_size: number of output features (only applicable if include_linear=True)
        :param include_linear: if True, linear layers are included in this model
        """
        super().__init__()
        self.conv_L_1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv_A_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv_L_2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv_A_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_L_3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv_A_3 = nn.ReLU()
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_linear = include_linear
        if self.include_linear:
            self.linear = nn.Linear(640, feature_size)
            self.relu = nn.ReLU()

    def forward(self, traj):
        """ Compute trajectory features
        
        :param traj: trajectory image (shape: [B, 1, 128, 320]
        :returns: features
        """
        traj_features = self.conv_A_1(self.conv_L_1(traj))
        traj_features = self.pool_1(traj_features)
        traj_features = self.conv_A_2(self.conv_L_2(traj_features))
        traj_features = self.pool_2(traj_features)
        traj_features = self.conv_A_3(self.conv_L_3(traj_features))
        traj_features = self.pool_3(traj_features)
        if self.include_linear:
            traj_features = torch.flatten(traj_features, 1)
            traj_features = self.relu(self.linear(traj_features))
        return traj_features
