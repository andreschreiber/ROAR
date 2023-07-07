import torch
import torch.nn as nn
import torch.nn.functional as F

from .SVAE import SVAE
from .ImageNets import ImageModel, TrajModel


class ROAR(nn.Module):
    """ ROAR model """
    def __init__(self, device, freeze_features, pretrained_file, horizon):
        """ ROAR implementation
        
        :param device: device to use
        :param freeze_features: if true, image features are frozen
        :param pretrained_file: pretrained file for image extractor
        :param horizon: prediction horizon (in time-steps)
        """
        super().__init__()

        self.variant = "roar"

        self.image_model = ImageModel(freeze_features, pretrained_file)
        self.traj_model  = TrajModel()
        self.lidar_model = SVAE(device)

        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)

        self.linear_lidar_occl_1 = nn.Linear(64, 32)
        self.activate_lidar_occl_1 = nn.ReLU()
        self.linear_lidar_occl_2 = nn.Linear(32, 1)
        self.activate_lidar_occl_2 = nn.Sigmoid()
        
        self.linear_image_occl_1 = nn.Linear(64, 32)
        self.activate_image_occl_1 = nn.ReLU()
        self.linear_image_occl_2 = nn.Linear(32, 1)
        self.activate_image_occl_2 = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.5)

        self.fc_L_1 = nn.Linear(64*4, 128)
        self.fc_A_1 = nn.ReLU()
        self.fc_L_2 = nn.Linear(128, horizon)
        self.fc_A_2 = nn.Sigmoid()

    def forward(self, imgs, trajs, lidar_scans, initial_state=None):
        """ Predict anomalies
        
        :param img: image to predict (shape: [B, S, 3, 240, 320])
        :param traj: trajectory data (shape: [B, S, 128, 320])
        :param lidar_scan: lidar scan (shape: [B, S, 1081])
        :param initial_state: initial state (shape: [B, 64]). Can be "None" for default initial state.
        :returns: dictionary with "pred_inv_scores" (shape: [B, S, Horizon]), "state" (shape: [B, 64]),
                  "lidar" (shape: [B, S, 1081]), "lidar_mean" (shape: [B, S, 32]), "lidar_log_var" (shape: [B, S, 32]),
                  "pred_img_score" (shape: [B, S, 1]) and "pred_lidar_score" (shape: [B, S, 1]).
        """
        batch_size = imgs.shape[0]
        seq_len = imgs.shape[1]

        # Construct initial state
        state = initial_state if initial_state is not None else torch.zeros((batch_size, 64), device=imgs.device)

        recon_lidars, means, log_vars, pred_inv_scores, image_occls, lidar_occls = [], [], [], [], [], []
        for step in range(seq_len): # Compute for each time-step
            img, traj, lidar_scan = imgs[:,step], trajs[:,step], lidar_scans[:,step]

            # Input features
            img_features  = self.image_model(img)
            traj_features = self.traj_model(traj)
            recon_lidar, mean, log_var = self.lidar_model(lidar_scan)
            lidar_features = torch.cat((mean, log_var), dim=-1)

            # Occlusion features
            image_occl_feat = self.activate_image_occl_1(self.linear_image_occl_1(self.dropout(img_features)))
            image_occl = self.activate_image_occl_2(self.linear_image_occl_2(image_occl_feat))
            lidar_occl_feat = self.activate_lidar_occl_1(self.linear_lidar_occl_1(self.dropout(lidar_features)))
            lidar_occl = self.activate_lidar_occl_2(self.linear_lidar_occl_2(lidar_occl_feat))

            # Perform attention
            kv_features = torch.cat((state.unsqueeze(0),
                                     traj_features.unsqueeze(0),
                                     img_features.unsqueeze(0),
                                     lidar_features.unsqueeze(0)), dim=0)
            q_features = torch.cat((
                torch.cat((image_occl_feat, lidar_occl_feat), dim=-1).unsqueeze(0),
                traj_features.unsqueeze(0),
                img_features.unsqueeze(0),
                lidar_features.unsqueeze(0)
            ), dim=0)
            attn_features, _ = self.mha(q_features, kv_features, kv_features) # (4, b, 64)
            state = F.hardtanh(attn_features[0], min_val=-10.0, max_val=10.0)
            attn_features = attn_features.permute(1, 0, 2)
            attn_features = torch.flatten(attn_features, 1) # (b, 4*64)

            # FC layers
            attn_features = self.fc_A_1(self.fc_L_1(attn_features))
            attn_features = self.dropout(attn_features)
            pred_inv_score = self.fc_A_2(self.fc_L_2(attn_features))

            recon_lidars.append(recon_lidar)
            means.append(mean)
            log_vars.append(log_var)
            pred_inv_scores.append(pred_inv_score)
            image_occls.append(image_occl)
            lidar_occls.append(lidar_occl)

        return {
            "lidar": torch.stack(recon_lidars, dim=1),
            "lidar_mean": torch.stack(means, dim=1),
            "lidar_log_var": torch.stack(log_vars, dim=1),
            "pred_inv_score": torch.stack(pred_inv_scores, dim=1),
            'pred_img_score': torch.stack(image_occls, dim=1),
            'pred_lidar_score': torch.stack(lidar_occls, dim=1),
            "state": state
        }


class IOROAR(nn.Module):
    """ Image-Only Variant of ROAR (IO-ROAR) """
    def __init__(self, device, freeze_features, pretrained_file, horizon):
        """ Image-only version of ROAR
        
        :param device: device to use
        :param freeze_features: if true, image features are frozen
        :param pretrained_file: pretrained file for image extractor
        :param horizon: prediction horizon (in time-steps)
        """
        super().__init__()

        self.variant = "ioroar"

        self.image_model = ImageModel(freeze_features, pretrained_file)
        self.traj_model  = TrajModel()

        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        
        self.linear_image_occl_1 = nn.Linear(64, 32)
        self.activate_image_occl_1 = nn.ReLU()
        self.linear_image_occl_2 = nn.Linear(32, 1)
        self.activate_image_occl_2 = nn.Sigmoid()

        self.fc_L_1 = nn.Linear(192, 128)
        self.fc_A_1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc_L_2 = nn.Linear(128, horizon)
        self.fc_A_2 = nn.Sigmoid()

    def forward(self, imgs, trajs, lidar_scans=None, initial_state=None):
        """ Predict anomalies
        
        :param img: image to predict (shape: [B, S, 3, 240, 320])
        :param traj: trajectory data (shape: [B, S, 128, 320])
        :param lidar_scan: lidar scan (kept for consistency but unused)
        :param initial_state: initial state (shape: [B, 64]). Can be "None" for default initial state.
        :returns: dictionary with "pred_inv_scores" (shape: [B, S, Horizon]), "state" (shape: [B, 64]),
                  and "pred_img_score" (shape: [B, S, 1]).
        """

        batch_size = imgs.shape[0]
        seq_len = imgs.shape[1]

        # Construct initial state
        state = initial_state if initial_state is not None else torch.zeros((batch_size, 64), device=imgs.device)

        pred_inv_scores, image_occls = [], []
        for step in range(seq_len): # Compute for each time-step
            img, traj = imgs[:,step], trajs[:,step]
            
            # Input features
            img_features = self.image_model(img)
            traj_features = self.traj_model(traj)

            # Occlusion features
            image_occl_feat = self.activate_image_occl_1(self.linear_image_occl_1(self.dropout(img_features)))
            image_occl = self.activate_image_occl_2(self.linear_image_occl_2(image_occl_feat))

            # Perform attention
            kv_features = torch.cat((state.unsqueeze(0),
                                     traj_features.unsqueeze(0),
                                     img_features.unsqueeze(0)), dim=0)
            q_features = torch.cat((
                torch.cat((image_occl_feat, image_occl_feat), dim=-1).unsqueeze(0),
                traj_features.unsqueeze(0),
                img_features.unsqueeze(0),
            ), dim=0)
            attn_features, _ = self.mha(q_features, kv_features, kv_features) # (4, b, 64)
            state = F.hardtanh(attn_features[0], min_val=-10.0, max_val=10.0)
            attn_features = attn_features.permute(1, 0, 2)
            attn_features = torch.flatten(attn_features, 1) # (b, 4*64)

            # FC layers
            attn_features = self.fc_A_1(self.fc_L_1(attn_features))
            attn_features = self.dropout(attn_features)
            pred_inv_score = self.fc_A_2(self.fc_L_2(attn_features))

            pred_inv_scores.append(pred_inv_score)
            image_occls.append(image_occl)

        return {
            "pred_inv_score": torch.stack(pred_inv_scores, dim=1),
            'pred_img_score': torch.stack(image_occls, dim=1),
            "state": state
        }
