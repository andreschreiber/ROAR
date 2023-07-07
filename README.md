# ROAR
This repository contains the code for ROAR from the IROS 2023 paper:

**An Attentional Recurrent Neural Network for Occlusion-Aware Proactive Anomaly Detection in Field Robot Navigation** (Andre Schreiber, Tianchen Ji, D. Livingston McPherson, and Katherine Driggs-Campbell).

Code was tested using Python 3.9.12 and PyTorch 1.12.1 (CUDA 11.6) on Windows 10.

### Abstract
The use of mobile robots in unstructured environments like the agricultural field is becoming increasingly common. The ability for such field robots to proactively identify and avoid failures is thus crucial for ensuring efficiency and avoiding damage; however, the cluttered field environment introduces various sources of noise (such as sensor occlusions) that make proactive anomaly detection difficult. Existing approaches can show poor performance in sensor occlusion scenarios as they typically do not explicitly model occlusions and only leverage current sensory inputs. In this work, we present an attention-based recurrent neural network architecture for proactive anomaly detection that fuses current sensory inputs and planned control actions with a latent representation of prior robot state. We enhance our model with an explicitly-learned model of sensor occlusion that is used to modulate the use of our latent representation of prior robot state. Our method shows improved anomaly detection performance and enables mobile field robots to display increased resilience to predicting false positives regarding navigation failure during periods of sensor occlusion, particularly in cases where all sensors are briefly occluded.

### Code Description

- nets/
  - This folder contains the code for the neural networks
- custom_dataset.py
  - Contains code for the custom sequential intervention dataset
- test.py
  - Code for testing a model (computing PR-AUC and F1-score)
- train.py
  - Code for training a model
- utils.py
  - Utility code for training, evaluation, loading models, etc.

### Running the Code

#### Dependencies
The code has dependencies for torch, PIL, numpy, tqdm, and sklearn.
These can be installed by using conda or pip (and following the installation instructions from the respective sources).

These can be installed (for CUDA machines) as follows:

    conda create -n roar python=3.9.12
    conda activate roar
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
    conda install -c conda-forge tqdm
    conda install -c anaconda scikit-image
    conda install -c anaconda scikit-learn

#### Required Files
Make sure a dataset has been downloaded (and ideally located in a data/ folder unless you modify data file paths), and that you have a folder "experiments" in the same directory as train.py and test.py. Also make sure that you have a weights/ folder containing the "VisionNavNet_state_hd.pth.tar" pretrained Vision Nav Net weights.

#### Training
To run a training of a model, use the following structure for the command line arguments (assuming the downloaded CSVs and images are in the data/ folder under data/train_set and data/test_set).

    python train.py --seed <SEED> --experiment_name <EXPERIMENT_NAME> --epochs <EPOCHS> --patience <PATIENCE> --train_batch_size <BATCH_SIZE> --max_sequence_length <MAX_SEQ_LENGTH> --model_variant <MODEL_VARIANT>

where the model_variant setting can be "ioroar" (image-only ROAR) or "roar" (complete ROAR), and the max_sequence_length should be 8 to match the paper.

#### Testing
To run a testing of an existing experiment, use the following structure for the command line arguments (again, assuming the test data is located under data/test_set).

    python test.py --experiment_name <EXPERIMENT_NAME> --results_file <RESULTS_FILE_TXT> --max_sequence_length <MAX_SEQUENCE_LENGTH>
(Note: max sequence length will not cause issues with states; using a max_sequence_length=1 will still use prior prediction state if still on same sequence.)

#### Changing the Data Paths
The data paths can be changed using --train_image_path, --train_csv_path, --test_image_path, and --test_csv_path.

### Dataset

The dataset was originally released in the paper "Proactive Anomaly Detection for Robot Navigation with Multi-Sensor Fusion" (Ji et al., 2022).

We add additional labels for occlusions of the LiDAR and RGB camera. The original images for the dataset can be found [here](https://uofi.app.box.com/s/n1qhun9u7lwgtgeyb6hd0tzxpbyxgpl7/folder/155298878008), while the updated CSV files containing the additional occlusion labels can be found [here](https://uofi.box.com/s/fiz5bf99vd6lk92vy94ztn4nxz9qp0dl).

The pretrained weights for the ROAR and IO-ROAR models from the paper (as well as the pretrained visual navigation model) can be found [here](https://uofi.box.com/s/v2jqlh615tv9eeazadff64t63w5bu1kf).

### Acknowledgements
The code structure is based on a prior implementation of PAAD (https://github.com/tianchenji/PAAD):

    @article{ji2022proactive,
    title={Proactive Anomaly Detection for Robot Navigation With Multi-Sensor Fusion},
    author={Ji, Tianchen and Sivakumar, Arun Narenthiran and Chowdhary, Girish and Driggs-Campbell, Katherine},
    journal={IEEE Robotics and Automation Letters},
    year={2022},
    volume={7},
    number={2},
    pages={4975-4982},
    doi={10.1109/LRA.2022.3153989}}

### Contact
If you have any questions or comments, please feel free to open an issue or email the author (andrems2@illinois.edu).
