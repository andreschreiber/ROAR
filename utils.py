import tqdm
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.ROAR import ROAR, IOROAR
from custom_dataset import SequentialInterventionDataset
from pathlib import Path
from sklearn.metrics import average_precision_score, confusion_matrix


#
# Dataset and model creation
#


def create_dataset(network, image_path, csv_path, mode, seq_len):
    """ Create dataset
    
    :param network: neural network model
    :param image_path: path to images
    :param csv_path: path to csv
    :param mode: dataset mode
    :param seq_len: max sequence length
    """
    ignore_partial = (mode == 'train')
    return SequentialInterventionDataset(image_path, csv_path, dataset_type=mode, ignore_partial=ignore_partial, max_sequence_length=seq_len)


def get_model_from_args(args, device):
    """ Fetch model using data from command line arguments
    
    :param args: command line args
    :param device: device to use for model
    """
    if args.model_variant == "roar":
        return ROAR(device=device, freeze_features=args.freeze_features, pretrained_file=args.pretrained_file, horizon=args.horizon).to(device)
    elif args.model_variant == "ioroar":
        return IOROAR(device=device, freeze_features=args.freeze_features, pretrained_file=args.pretrained_file, horizon=args.horizon).to(device)
    else:
        raise ValueError("Invalid model provided")


def get_model_from_dict(d, device):
    """ Fetch model using data from experiment dictionary 
    
    :param d: experiment dictionary
    :param device: device to use for model
    """
    if d['model_variant'] == "roar":
        return ROAR(device=device, freeze_features=d['freeze_features'], pretrained_file=d['pretrained_file'], horizon=d['horizon']).to(device)
    elif d['model_variant'] == "ioroar":
        return IOROAR(device=device, freeze_features=d['freeze_features'], pretrained_file=d['pretrained_file'], horizon=d['horizon']).to(device)
    else:
        raise ValueError("Invalid model provided")
    

def count_parameters(model, only_trainable=False):
    """ Counts the number of neural network parameters
    
    Uses the implementation from:
    https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model/62764464#62764464

    :param model: model to count parameters for
    :param only_trainable: if True, only trainable parameters are counted
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


#
# Loss
#


def get_loss_fn(settings):
    """ Return the loss function 
    
    :param settings: settings
    """
    return ROARLoss(**settings)


class ROARLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        """ Loss function
        
        :param alpha: weight for cross entropy of interventions
        :param beta: weight for image occlusion
        :param gamma: weight for lidar occlusion
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """ Compute loss 
        
        :param predictions: predictions from model (dictionary of tensors)
        :param targets: ground truth targets (dictionary of tensors)
        :returns: loss
        """

        # Prediction loss
        loss = self.alpha * F.binary_cross_entropy(predictions['pred_inv_score'], targets['intervention_label'], reduction='sum')

        # LiDAR loss
        if 'lidar' in targets and 'lidar' in predictions:
            loss += F.mse_loss(predictions['lidar'], targets['lidar'], reduction='sum')
        if 'lidar_log_var' in predictions and 'lidar_mean' in predictions:
            loss += -0.5 * torch.sum(1 + predictions['lidar_log_var'] - predictions['lidar_mean'].pow(2) - predictions['lidar_log_var'].exp())
        
        # Image occlusion
        if 'pred_img_score' in predictions and 'img_label' in targets:
            img_occ_label = targets['img_label']
            loss += self.beta * F.binary_cross_entropy(predictions['pred_img_score'], img_occ_label, reduction='sum')
        
        # Lidar occlusion
        if 'pred_lidar_score' in predictions and 'lidar_label' in targets:
            lidar_occ_label = targets['lidar_label']
            loss += self.gamma * F.binary_cross_entropy(predictions['pred_lidar_score'], lidar_occ_label, reduction='sum')
        
        return loss / predictions['pred_inv_score'].shape[0]


# 
# Experiment
#


class Experiment:
    """ Experiment object """
    def __init__(self, path, should_exist=False):
        """ Create experiment
        
        :param path: path to experiment folder
        :param should_exist: if True, will report an error if folder does not exist.
        """
        if should_exist and not path.exists():
            raise FileNotFoundError("Experiment directory does not exist but should")
        elif not should_exist and path.exists():
            raise FileNotFoundError("Experiment directory exists but should not")
        
        path.mkdir(parents=True, exist_ok=True)
        self.folder = path
        self.model_file = path / Path('model.pth')
        self.settings_file = path / Path('settings.json')

    def read_settings(self):
        """ Reads and returns (as dict) the settings JSON file """
        settingsJson = None
        with open(self.settings_file, 'r') as f:
            settingsJson = json.load(f)
        return settingsJson

    def write_settings(self, args):
        """ Writes the settings json file from command line arguments """
        settingsJson = {
            'seed': args.seed,
            'epochs': args.epochs,
            'patience': args.patience,
            'train_batch_size': args.train_batch_size,
            'max_sequence_length': args.max_sequence_length,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'experiment_name': args.experiment_name,
            'freeze_features': args.freeze_features,
            'pretrained_file': args.pretrained_file,
            'horizon': args.horizon,
            'model_variant': args.model_variant
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settingsJson, f)


#
# Training and evalution
#


def model_train(data_loader, model, loss_fn, device, optimizer):
    """ Trains a model
    
    :param data_loader: data loader to use
    :param model: model to use
    :param loss_fn: loss function to use
    :param device: device to use
    :param optimizer: optimizer to use
    :returns: mean loss
    """

    model.train()
    running_loss = 0.0
    iterations = 0
    for (img, pred_traj, lidar_scan, label, img_label, lidar_label, seq) in tqdm.tqdm(data_loader):
        img, pred_traj = img.to(device), pred_traj.to(device)
        img_label, lidar_label = img_label.to(device), lidar_label.to(device)
        lidar_scan, label  = lidar_scan.to(device), label.to(device)

        predictions = model(img, pred_traj, lidar_scan)
        
        targets = {'lidar': lidar_scan.flatten(0,1), 'intervention_label': label.flatten(0,1),
                   'img_label': img_label.flatten(0,1), 'lidar_label': lidar_label.flatten(0,1)}
        for field in ['lidar', 'lidar_log_var', 'lidar_mean',
                      'pred_inv_score', 'pred_img_score', 'pred_lidar_score']:
            if field in predictions:
                predictions[field] = predictions[field].flatten(0,1)

        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.cpu().item()
        iterations += 1
    mean_loss = running_loss / iterations
    return mean_loss


def model_evaluation(data_loader, model, device, threshold=0.5):
    """ Evaluates a model
    
    :param data_loader: data loader to use
    :param model: model to use
    :param device: device to use
    :param threshold: threshold for threshold-dependent metrics
    """

    gt_list = []
    pred_score_list = []
    pred_label_list = []
    
    model.eval()
    state = None
    last_seq = None # used to determine when to reset the state

    infer_times = []
    total_time = time.time()

    with torch.no_grad():
        for (img, pred_traj, lidar_scan, label, img_label, lidar_label, seq) in tqdm.tqdm(data_loader):
            if seq != last_seq or last_seq is None: # reset the state if new sequence
                state = None
                last_seq = seq

            img, pred_traj = img.to(device), pred_traj.to(device)
            lidar_scan = lidar_scan.to(device)

            # Infer
            infer_time = time.time()
            predictions = model(img, pred_traj, lidar_scan, initial_state=state)
            infer_time = time.time() - infer_time
            infer_times.append(infer_time)

            # Update state
            state = predictions['state']
            # Get predictions
            pred_score = predictions['pred_inv_score']
            pred_label = predictions['pred_inv_score'] > threshold

            # Prepare labels
            label = label.flatten(0,1).cpu().numpy().flatten()
            pred_score = pred_score.flatten(0,1).cpu().numpy().flatten()
            pred_label = pred_label.flatten(0,1).cpu().numpy().flatten()

            gt_list.extend(list(label))
            pred_score_list.extend(list(pred_score))
            pred_label_list.extend(list(pred_label))
    
    total_time = time.time() - total_time

    # Compute metrics
    ap_measure = average_precision_score(gt_list, pred_score_list)
    tn, fp, fn, tp = confusion_matrix(gt_list, pred_label_list).ravel()
    if tp == 0:
        precision, recall, f1_measure = 0.0, 0.0, 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_measure = (2 * precision * recall) / (precision + recall)
    
    return {'f1': f1_measure, 'ap': ap_measure, 'precision': precision, 'recall': recall,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'infer_times': infer_times, 'total_time': total_time}
