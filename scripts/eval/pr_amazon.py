"""
Compute the Precision-Recall Curve on Amazon.
"""

import os
import torch
import numpy as np
from pprint import pprint
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.agents.nlp import *
from src.utils.setup import process_config, process_config_from_json
from src.objectives.prototype import batch_euclidean_dist
from src.datasets.amazon import EvalFewShotAmazonSentiment
from src.utils import utils

from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    roc_curve,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pr_curve(args, gpu_device=-1):
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
    agent.model.eval()

    task_test_acc = []
    for i in range(12):
        print(f'Evaluating dataset ({i+1}/12)')
        test_dataset = EvalFewShotAmazonSentiment(config.dataset.data_root, i,
                                                  n_shots=config.dataset.test.n_shots)
        test_loader = DataLoader(test_dataset, batch_size=32,  # might be too high
                                 shuffle=False, pin_memory=True,
                                 num_workers=config.data_loader_workers)
        test_accuracy = evaluate(agent, config, test_loader)
        task_test_acc.append(test_accuracy)
    task_test_acc = np.array(task_test_acc)

    mean_test_acc = float(np.mean(task_test_acc))
    std_test_acc = float(np.std(task_test_acc))
    print(f'Test acc: {mean_test_acc} +- {std_test_acc}')

    checkpoint_name = args.checkpoint_file.replace('.pth.tar', '')
    out_file = os.path.join(analysis_dir, f'{checkpoint_name}_test_acc.npy')
    np.save(out_file, task_test_acc)


@torch.no_grad()
def evaluate(agent, config, test_loader):
    num_correct, num_total = 0, 0
    tqdm_batch = tqdm(total=len(test_loader))
    for index, batch in enumerate(test_loader):
        support_toks = test_loader.dataset.support_toks.to(agent.device)    # n_ways x n_shots x 512
        support_lens = test_loader.dataset.support_lens.to(agent.device)    # n_ways x n_shots
        support_masks = test_loader.dataset.support_masks.to(agent.device)  # n_ways x n_shots x 512
        support_labs = test_loader.dataset.support_labs.to(agent.device)    # n_ways x n_shots
        n_ways, n_shots, seq_len = support_toks.size()

        support_toks = support_toks.view(-1, seq_len)            # n_ways*n_shots x 512
        support_lens = support_lens.view(-1)                     # n_ways*n_shots
        support_masks = support_masks.view(-1, seq_len).long()   # n_ways*n_shots x 512

        query_toks = batch['query_toks'].to(agent.device)      # mb x 512
        query_lens = batch['query_lens'].to(agent.device)      # mb
        query_masks = batch['query_masks'].to(agent.device)    # mb x 512
        query_labs = batch['query_labs'].to(agent.device)      # mb
        side_info = batch['side_info'].to(agent.device)        # mb x 768
        mb = query_toks.size(0)

        if config.model.task_tam:
            # support_sides : n_ways*n_shots x 768
            support_sides = side_info[0].unsqueeze(0).repeat(n_ways * n_shots, 1)
            query_sides = side_info  # mb x 768
        
            # support_sides: n_ways*n_shots x 1 x 768
            support_sides = support_sides.unsqueeze(1)
            # query_sides: mb x 1 x 768
            query_sides = query_sides.unsqueeze(1)
        else:
            support_sides, query_sides = None, None

        if config.model.name == 'lstm':
            support_features = agent.model(support_toks, support_lens, tam_embeds=support_sides)
            query_features = agent.model(query_toks, query_lens, tam_embeds=query_sides)
        else:
            support_features = agent.model(
                input_ids=support_toks, attention_mask=support_masks, tam_embeds=support_sides)[0]
            query_features = agent.model(
                input_ids=query_toks, attention_mask=query_masks, tam_embeds=query_sides)[0]
            support_features = agent.compute_masked_means(support_features, support_masks)
            query_features = agent.compute_masked_means(query_features, query_masks)

        support_features = support_features.view(n_ways, n_shots, -1)  # n_ways x n_shots x dim
        support_labs = support_labs.view(n_ways, n_shots)              # n_ways x n_shots
        query_features = query_features.view(mb, -1)                   # mb x dim
        query_labs = query_labs.view(mb)                               # mb

        prototypes = torch.mean(support_features, dim=1)               # n_ways x dim
        dists = agent.tau * batch_euclidean_dist(                      # mb x 1 x n_ways
            query_features.unsqueeze(1),                               # mb x 1 x dim
            prototypes.unsqueeze(0).repeat(mb, 1, 1))                  # mb x n_ways x dim
        dists = dists.squeeze(1)                                       # mb x n_ways
        logprobas = F.log_softmax(-dists, dim=1)                       # mb x n_ways

        pred = torch.argmax(logprobas, dim=1)                          # mb
        correct = pred.eq(query_labs).sum()
        num_correct += correct.item()
        num_total += mb
        tqdm_batch.update()
    tqdm_batch.close()
    return num_correct / float(num_total)


def precision_at_recall(precisions, recalls, at=0.5):
    return precisions[np.argmin(np.abs(recalls - at))]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='path to experiment directory')
    parser.add_argument('checkpoint_file', type=str, help='name of checkpoint')
    parser.add_argument('--gpu-device', type=int, default=-1)
    args = parser.parse_args()
    pr_curve(args, gpu_device=args.gpu_device)
