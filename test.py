import torch
import argparse
import utils
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader


def test_all(args):
    """ Test on entire dataset
    
    :param args: command line arguments object (from argparse)
    """

    # Load experiment
    experiment_path = Path("experiments") / Path(args.experiment_name)
    experiment = utils.Experiment(experiment_path, should_exist=True)
    settings = experiment.read_settings()

    # Load network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = utils.get_model_from_dict(settings, device)
    network.load_state_dict(torch.load(experiment.model_file))

    # Max sequence length will be that of experiment if it is not provided as command line arg
    max_seq_len = args.max_sequence_length
    if max_seq_len is None:
        max_seq_len = settings['max_sequence_length']

    # Create dataset and evaluate
    test_set = utils.create_dataset(network, args.test_image_path, args.test_csv_path, 'test', max_seq_len)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    evaluation = utils.model_evaluation(test_loader, network, device, threshold=args.threshold)

    # Report metrics of interest
    print("Average precision on the test set: {:.6f}".format(evaluation['ap']))
    print("F1 measure on the test set: {:.6f}".format(evaluation['f1']))
    print("Test set precision: {:.6f}".format(evaluation['precision']))
    print("Test set recall: {:.6f}".format(evaluation['recall']))
    print("Test set confusion: TP = {}, TN = {}, FP = {}, FN = {}".format(evaluation['tp'], evaluation['tn'], evaluation['fp'], evaluation['fn']))
    print("Average inference time = {:.4f} ms (sequence length used = {}). Infer time: Min = {:.4f} ms, Median = {:.4f} ms, Max = {:.4f} ms".format(
        np.array(evaluation['infer_times']).mean()*1000.0,
        max_seq_len,
        min(evaluation['infer_times'])*1000.0,
        np.median(np.array(evaluation['infer_times']))*1000.0,
        max(evaluation['infer_times'])*1000.0
    ))
    print("Total inference time = {:.4f} ms".format(sum(evaluation['infer_times'])*1000.0))
    print("Total time = {:.4f} ms for {} inferences".format(evaluation['total_time']*1000.0, len(evaluation['infer_times'])))
    print("Number of parameters: {}".format(utils.count_parameters(network, False)))
    print("Number of trainable parameters: {}".format(utils.count_parameters(network, True)))

    # Save metrics of interest to file
    if args.results_file:
        with open(args.results_file, 'w') as f:
            f.write("Average precision on the test set: {:.6f}\n".format(evaluation['ap']))
            f.write("F1 measure on the test set: {:.6f}\n".format(evaluation['f1']))
            f.write("Test set precision: {:.6f}\n".format(evaluation['precision']))
            f.write("Test set recall: {:.6f}\n".format(evaluation['recall']))
            f.write("Test set confusion: TP = {}, TN = {}, FP = {}, FN = {}\n".format(evaluation['tp'], evaluation['tn'], evaluation['fp'], evaluation['fn']))
            f.write("Average inference time = {:.4f} ms (sequence length used = {})\nInfer time: Min = {:.4f} ms, Median = {:.4f} ms, Max = {:.4f} ms\n".format(
                np.array(evaluation['infer_times']).mean()*1000.0,
                max_seq_len,
                min(evaluation['infer_times'])*1000.0,
                np.median(np.array(evaluation['infer_times']))*1000.0,
                max(evaluation['infer_times'])*1000.0
            ))
            f.write("Total inference time = {:.4f} ms\n".format(sum(evaluation['infer_times'])*1000.0))
            f.write("Total time = {:.4f} ms for {} inferences\n".format(evaluation['total_time']*1000.0, len(evaluation['infer_times'])))
            f.write("Number of parameters: {}\n".format(utils.count_parameters(network, False)))
            f.write("Number of trainable parameters: {}\n".format(utils.count_parameters(network, True)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--test_image_path", type=str, default='data/test_set/images_test/')
    parser.add_argument("--test_csv_path", type=str, default='data/test_set/labeled_data_test.csv')
    parser.add_argument("--threshold", type=bool, default=0.5)
    parser.add_argument("--results_file", type=str)
    parser.add_argument("--max_sequence_length", type=int, required=False)
    args = parser.parse_args()

    test_all(args)
