"""Evaluate accuracy for 1000 episodes on test set."""

import os
import numpy as np

from src.agents.nlp import *
from src.utils.setup import process_config, process_config_from_json
from src.datasets.text import *


def evaluate(args, gpu_device=-1):
    config_path = os.path.join(args.exp_dir, 'config.json')
    checkpoint_dir = os.path.join(args.exp_dir, 'checkpoints')
    analysis_dir = os.path.join(args.exp_dir, 'analysis')

    if not os.path.isdir(analysis_dir):
        os.makedirs(analysis_dir)

    config = process_config(config_path)

    if gpu_device >= 0: config.gpu_device = gpu_device
    config = process_config_from_json(config)

    AgentClass = globals()[config.agent]
    agent = AgentClass(config)

    agent.load_checkpoint(
        args.checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        load_model=True,
        load_optim=False,
        load_epoch=True,
    )
    # turn on eval mode
    agent.model.eval()

    class_dict = {
        'fs_20news': FewShot20News,
        'fs_amazon': FewShotAmazon,
        'fs_huffpost': FewShotHuffPost,
        'fs_rcv1': FewShotRCV1,
        'fs_reuters': FewShotReuters,
        'fs_fewrel': FewShotFewRel,
    }
    DatasetClass = class_dict[config.dataset.name]
    test_dataset = DatasetClass(
        data_root=config.dataset.data_root,
        n_ways=config.dataset.test.n_ways,
        n_shots=config.dataset.test.n_shots,
        n_queries=config.dataset.test.n_queries,
        split='test',
    )
    test_loader, _ = agent._create_test_dataloader(
        test_dataset,
        config.optim.batch_size,
    )
    _, accuracies, acc_stdevs = agent.eval_split('Test', test_loader, verbose=True)
    print(acc_stdevs)
    print(accuracies)

    checkpoint_name = args.checkpoint_file.replace('.pth.tar', '')
    accuracy_fpath = os.path.join(analysis_dir, f'{checkpoint_name}_accuracies.csv')
    np.savez(accuracy_fpath, accuracies=accuracies)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='path to experiment directory')
    parser.add_argument('checkpoint_file', type=str, help='name of checkpoint')
    parser.add_argument('--gpu-device', type=int, default=-1)
    args = parser.parse_args()
    evaluate(args, gpu_device=args.gpu_device)
