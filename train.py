import copy
import torch
import utils
import argparse
import time
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader


def main(args):
    """ Train a model
    
    :param args: command line arguments
    """

    # Load experiment
    experiment_path = Path("experiments") / Path(args.experiment_name)
    experiment = utils.Experiment(experiment_path)
    experiment.write_settings(args)
    
    # Setup seed
    if args.seed:
        print("Provided seed will be used")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    else:
        print("Using system's random seed")

    # Report command line parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = utils.get_model_from_args(args, device)

    # Create train and test set
    train_set = utils.create_dataset(network, args.train_image_path, args.train_csv_path, 'train', args.max_sequence_length)
    test_set  = utils.create_dataset(network, args.test_image_path, args.test_csv_path, 'test', args.max_sequence_length)
    train_loader = DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    print("Data loaded")

    alpha = 6.21 # from PAAD
    beta = 0.1 * alpha
    gamma = 0.1 * alpha

    loss_fn = utils.get_loss_fn(settings={'alpha': alpha, 'beta': beta, 'gamma': gamma})
    parameters = filter(lambda p: p.requires_grad, network.parameters())
    optimizer  = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Start training
    patience_tries = 0
    best_ap = 0.0
    train_loss_over_epochs = []
    test_ap_over_epochs = []
    for epoch in range(args.epochs):
        elapsed = time.time()
        epoch_loss = utils.model_train(train_loader, network, loss_fn, device, optimizer)
        evaluation = utils.model_evaluation(test_loader, network, device, threshold=0.5)
        ap = evaluation['ap']
        f1 = evaluation['f1']
        elapsed = time.time() - elapsed

        train_loss_over_epochs.append(epoch_loss)
        test_ap_over_epochs.append(ap)

        print("Finished training epoch {:02d}/{:02d} in {:.1f} seconds".format(epoch+1, args.epochs, elapsed))
        print("Loss: {:.3f}. AP: {:.3f}, F1: {:.3f}".format(epoch_loss, ap, f1))
        if ap > best_ap: # Save model if AP exceeds prior best
            torch.save(network.state_dict(), experiment.model_file)
            best_ap = copy.deepcopy(ap)
            print("New best model found!")
            patience_tries = 0
        else: # Check pateince
            patience_tries += 1
            if patience_tries >= args.patience:
                print("Patience has been exceeded. Aborting remainder of training...")
                break
    
    # Save training data into plot
    print("Training has finished with best AP = {:.3f}".format(best_ap))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--train_image_path", type=str, default='data/train_set/images_train/')
    parser.add_argument("--train_csv_path", type=str, default='data/train_set/labeled_data_train.csv')
    parser.add_argument("--test_image_path", type=str, default='data/test_set/images_test/')
    parser.add_argument("--test_csv_path", type=str, default='data/test_set/labeled_data_test.csv')

    # Training parameters
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.00015)
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--max_sequence_length", type=int, default=8)

    # Model parameters
    parser.add_argument("--freeze_features", type=bool, default=True)
    parser.add_argument("--pretrained_file", type=str, default="weights/VisionNavNet_state_hd.pth.tar")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--model_variant", type=str)

    args = parser.parse_args()
    main(args)
