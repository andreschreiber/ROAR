import os
import csv
import torch
import numpy as np
from skimage import io
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset
import random

class SequentialInterventionDataset(Dataset):
    def __init__(self, image_path, csv_path, dataset_type, max_sequence_length=8, ignore_partial=True):
        """ Create sequential intervention dataset
        
        :param image_path: path to image files
        :param csv_path: path to csv file
        :param dataset_type: type of dataset ("train" if training modifications should be used)
        :param max_sequence_length: length of sequences (should be 8 for sequential models, 1 for non-sequential models).
        :param ignore_partial: if True, partial sequences (i.e., sequences not of length max_sequence_length) are not included.
        """
        self.samples = []
        self.image_path = image_path
        self.csv_path = csv_path 
        self.dataset_type = dataset_type
        self.lidar_clip = 1.85
        self.f = 460
        self.image_width, self.image_height = 320, 240
        self.cam_height = -0.23
        
        self.normal_failure_ratio = self.compute_ratio() - 1
        self.max_sequence_length = max_sequence_length
        self.ignore_partial = ignore_partial
        self.read_data()

    def __len__(self):
        """ Returns number of samples """
        return len(self.samples)

    def __getitem__(self, idx):
        """ Fetch sample
        
        :param idx: index of sample
        :returns: tuple of (images, trajectories, lidars, labels, image_occlusions, lidar_occlusions, sequence_num)
        """
        sample = self.samples[idx]
        images = [img for img in sample['images']]

        # Apply color jitter to the training set
        if self.dataset_type == 'train':
            color_jitter_tx = transforms.RandomApply(
                        [transforms.ColorJitter(0.5, 0.25, 0.25, 0.1)], p=0.5)
            images = [color_jitter_tx(img) for img in images]

        # Image normalization transform
        image_tx = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # Create trajectory data
        pred_traj = sample['pred_traj']
        final_traj = []
        for traj in pred_traj:
            ego_traj = self.get_ego_trajectory(traj)
            traj_image = Image.new(mode="L", size=(self.image_width, self.image_height))
            traj_draw = ImageDraw.Draw(traj_image)
            traj_draw.line(ego_traj, fill="white", width=6, joint="curve")
            traj_image = traj_image.crop((0, 112, 320, 240))
            traj_image = transforms.ToTensor()(traj_image)
            final_traj.append(traj_image)

        # Create lidar data
        final_lidar = sample['lidar_scan']
        
        # Create final tensors
        images = torch.stack([image_tx(img) for img in images])
        final_traj = torch.stack(final_traj)
        final_lidar = torch.stack(final_lidar)
        label = torch.stack(sample['label'])
        image_occl = torch.stack(sample['image_occluded'])
        lidar_occl = torch.stack(sample['lidar_occluded'])
        sequence = sample['sequence']
        
        return (images, final_traj, final_lidar, label, image_occl, lidar_occl, sequence)

    def compute_ratio(self):
        """ Compute ratio of normal to augmented failure """
        num_normal  = 0
        num_failure = 0
        map_to_int = lambda x: np.array(list(map(int, x)))
        with open(self.csv_path, newline='') as dataCSV:
            data_reader = csv.DictReader(dataCSV)
            for datapoint in data_reader:
                label = datapoint['label'][1:-1].split(',')
                label = map_to_int(label)
                if max(label) == 0:
                    num_normal += 1
                else:
                    num_failure += 1
        # Make sure to include flipped
        num_failure = num_failure * 2
        nf_ratio = round(num_normal / num_failure)
        if self.dataset_type == 'train':
            print("The ratio of normal to augmented failure is: {:d}".format(nf_ratio))
        return nf_ratio

    def get_ego_trajectory(self, pred_traj):
        """ Computes the ego trajectory """
        predXs = pred_traj[:,0]
        predYs = pred_traj[:,1]
        xs = self.f * predYs / predXs
        ys = self.f * self.cam_height / predXs
        xs = -xs + self.image_width / 2
        ys = -ys + self.image_height / 2
        ego_traj = [(x,y) for x,y in zip (xs, ys)]
        return ego_traj

    def read_datapoint(self, datapoint):
        """ Reads a provided datapoint 
        
        :param datapoint: datapoint in form of iterate from csv.DictReader
        :returns: tuple of (image, traj, lidar, label, image occlusion, lidar occlusion, sequence num)
        """
        map_to_float = lambda x: np.array(list(map(float, x)))
        map_to_int = lambda x: np.array(list(map(int, x)))
        
        # Read image
        image_name = os.path.join(self.image_path, datapoint['image_name'])
        image = io.imread(image_name)
        image = Image.fromarray(image)
        
        # Create data for traj image
        predX = datapoint['pred_traj_x'][1:-1].split(',')
        predY = datapoint['pred_traj_y'][1:-1].split(',')
        predX = abs(map_to_float(predX))
        predY = map_to_float(predY)
        pred_traj = torch.tensor(np.stack([predX, predY])).to(torch.float32).permute(1,0)

        # Get LiDAR data
        lidar_scan = datapoint['lidar_scan'][1:-1].split(',')
        lidar_scan = map_to_float(lidar_scan)
        # Clip LiDAR
        lidar_scan = np.clip(lidar_scan, a_min=0, a_max=self.lidar_clip) / self.lidar_clip
        lidar_scan = torch.as_tensor(lidar_scan, dtype=torch.float32)

        # Get label
        label = datapoint['label'][1:-1].split(',')
        label = map_to_int(label)
        label = torch.as_tensor(label, dtype=torch.float32)
        
        # Get occlusion labels and sequence number
        image_occl = torch.as_tensor(int(datapoint['image_occluded']), dtype=torch.float32).unsqueeze(0)
        lidar_occl = torch.as_tensor(int(datapoint['lidar_occluded']), dtype=torch.float32).unsqueeze(0)
        sequence = torch.as_tensor(int(datapoint['sequence']), dtype=torch.int).unsqueeze(0)
        
        return (image, pred_traj, lidar_scan, label, image_occl, lidar_occl, sequence)

    def read_data(self):
        """ Reads the data provided in the CSV file """
        raw_samples = []
        sequences = []
        interventions = []
        
        # Collect all data samples
        with open(self.csv_path, newline='') as dataCSV:
            data_reader = csv.DictReader(dataCSV)
            for datapoint in data_reader:
                image, pred_traj, lidar_scan, label, image_occl, lidar_occl, sequence = self.read_datapoint(datapoint)
                seq = int(sequence.cpu().item())
                intervention = int(label.max().item() != 0)
                raw_samples.append({'image': image, 'pred_traj': pred_traj, 'lidar_scan': lidar_scan, 'label': label,
                                    'image_occluded': image_occl, 'lidar_occluded': lidar_occl, 'intervention': intervention, 'sequence': seq})
                sequences.append(seq)
                interventions.append(intervention)
        
        sequences = np.array(sequences)
        interventions = np.array(interventions)
        
        clean_samples = []
        interv_samples = []
        all_samples = []
        data_counter = [0, 0]
        
        index = 0
        while index < len(raw_samples):
            sample = raw_samples[index]
            # Find start of sequence (with same number as this sequence) and length.
            sequence_start = (sequences == sample['sequence']).argmax()
            sequence_len = (sequences == sample['sequence']).sum()
            
            # If training, randomly modify sequence start if not a multiple of sequence length
            if self.dataset_type == 'train':
                if index == sequence_start and sequence_len % self.max_sequence_length != 0:
                    index += random.randint(0, sequence_len % self.max_sequence_length)
                    if index >= len(raw_samples): # Exceeded samples
                        break
            
            # Fetch sample with (potentially) updated index
            sample = raw_samples[index]
            sequence_number = sample['sequence']
            sequence_start = (sequences == sample['sequence']).argmax()
            sequence_len = (sequences == sample['sequence']).sum()
            # Get end of this subsequence
            index_end = min(sequence_start + sequence_len, index + self.max_sequence_length)
            if self.ignore_partial and (index_end - index) != self.max_sequence_length:
                # Skip if index end is not multiple of max_sequence_length and we want to ignore partial
                index = index_end
                continue
            
            samples = raw_samples[index:index_end]
            is_intervention = max([s['intervention'] for s in samples]) > 0

            # Construct the samples dictionary for the subsequence
            samples = {
                'images': [s['image'] for s in samples],
                'pred_traj': [s['pred_traj'] for s in samples],
                'lidar_scan': [s['lidar_scan'] for s in samples],
                'label': [s['label'] for s in samples],
                'image_occluded': [s['image_occluded'] for s in samples],
                'lidar_occluded': [s['lidar_occluded'] for s in samples],
                'sequence': sequence_number
            }
            data_counter[int(is_intervention)] += len(samples['images'])
            if self.dataset_type == 'train':
                # If train, interventions produce two subsequences (intervention and flipped intervention)
                if is_intervention:
                    interv_samples.append(samples)
                    interv_samples.append({
                        'images': [transforms.functional.hflip(s) for s in samples['images']],
                        'pred_traj': [s * torch.Tensor([[ 1., -1.]]) for s in samples['pred_traj']],
                        'lidar_scan': [torch.flip(s, [0]) for s in samples['lidar_scan']],
                        'label': [s for s in samples['label']],
                        'image_occluded': [s for s in samples['image_occluded']],
                        'lidar_occluded': [s for s in samples['lidar_occluded']],
                        'sequence': -samples['sequence'] # negative sequence numbers are flipped sequences
                    })
                else:
                    clean_samples.append(samples)
            else:
                if is_intervention:
                    interv_samples.append(samples)
                else:
                    clean_samples.append(samples)
                all_samples.append(samples)
            # Update index to end index
            index = index_end
        
        if self.dataset_type == 'train':
            # Undersample non anomaly cases for "train" mode
            clean_samples = random.sample(clean_samples, len(clean_samples)//int(self.normal_failure_ratio))
        
        # Calculate the augmented statistics
        data_counter_aug = [sum([len(s['images']) for s in (clean_samples)]), sum([len(s['images']) for s in (interv_samples)])]
        # Set samples and print diagnostics
        if self.dataset_type == 'train':
            self.samples = clean_samples + interv_samples
        else:
            self.samples = all_samples
        print("All data have been loaded! Total dataset size: {:d}".format(sum(data_counter_aug)))
        print("The number of normal cases / failures: {:d} / {:d}".format(data_counter_aug[0], data_counter_aug[1]))
        print("The number of normal cases / failures before augmentation: {:d} / {:d}".format(data_counter[0], data_counter[1]))
