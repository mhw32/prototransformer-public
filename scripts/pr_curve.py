"""
Compute the Precision-Recall Curve given a trained model.
"""

import os
import torch
from pprint import pprint
from copy import deepcopy
from src.agents.purplebook import *
from src.utils.setup import process_config, process_config_from_json
from src.utils import utils

from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    roc_curve,
)


def pr_curve(args, gpu_device=-1):
    config_path = os.path.join(args.exp_dir, 'config.json')
    checkpoint_dir = os.path.join(args.exp_dir, 'checkpoints')
    analysis_dir = os.path.join(args.exp_dir, 'analysis')

    if not os.path.isdir(analysis_dir):
        os.makedirs(analysis_dir)

    config = utils.load_json(config_path)

    for key in ['exp_base', 'exp_dir', 'checkpoint_dir', 'out_dir', 'log_dir']:
        config[key] = config[key].replace('/iris/u', '/mnt/fs5')
    config['data_root'] = '/data5/wumike'

    config = process_config_from_json(config)

    if gpu_device >= 0: config.gpu_device = gpu_device
    config.override_n_shots = args.n_shots  # override how many shots to use
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
    skipped = 0
    agent.model.eval()
    with torch.no_grad():
        probas_db, labels_db, preds_db = [], [], []  # only binary
        all_preds_db, all_labels_db = [], []
        task_indices = []  # so we can trace back to the problem

        tqdm_batch = tqdm(total=len(agent.test_loader))
        for index, batch in enumerate(agent.test_loader):
            n_shots = agent.config.dataset.train.n_shots
            n_queries = agent.config.dataset.test.n_queries

            task_type = batch['task_type'].cpu().numpy()[0]
            if task_type != 0: 
                tqdm_batch.update()
                continue

            if config.agent == 'CodeRelationNetAgent':
                _, _, scores = agent.forward(batch, n_shots, n_queries)
                probas = torch.clamp(scores, 0, 1)
                labels = batch['query_labs'].to(agent.device)
                mb, nway, nquery = labels.size()
                ndim = probas.size(-1)
                labels = labels.view(mb * nway * nquery)
                preds = torch.argmax(probas, dim=-1)
            else:
                # probas : batch_size, nway, nquery, -1
                _, _, logprobas = agent.forward(batch, n_shots, n_queries)
                probas = torch.exp(logprobas)   # these have been log_softmax'd
                # labels : batch_size, nway, nquery
                labels = batch['query_labs'].to(agent.device)

                mb, nway, nquery, ndim = probas.size()
                probas = probas.view(mb * nway * nquery, -1)
                labels = labels.view(mb * nway * nquery)
                preds = torch.argmax(probas, dim=-1)

            if ndim == 2:
                probas_db.append(probas.detach().cpu())
                labels_db.append(labels.detach().cpu())
                preds_db.append(preds.detach().cpu())
                task_indices.append(index)
            else:
                skipped += 1
            
            all_preds_db.append(preds.detach().cpu())
            all_labels_db.append(labels.detach().cpu())

            tqdm_batch.update()
        tqdm_batch.close()

    print(f'Skipped {skipped} items')
    probas_db = torch.cat(probas_db, dim=0).numpy()
    labels_db = torch.cat(labels_db, dim=0).numpy()
    preds_db = torch.cat(preds_db, dim=0).numpy()
    task_indices = np.array(task_indices)

    all_preds_db = torch.cat(all_preds_db, dim=0).numpy()
    all_labels_db = torch.cat(all_labels_db, dim=0).numpy()

    precision, recall, pr_thresholds = precision_recall_curve(labels_db, probas_db[:, 1]) 
    average_precision = average_precision_score(labels_db, probas_db[:, 1])
    precision_50 = precision_at_recall(precision, recall, at=0.5)
    precision_75 = precision_at_recall(precision, recall, at=0.75)
    roc_auc = roc_auc_score(labels_db, probas_db[:, 1])
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(labels_db, probas_db[:, 1])
    acc_binary = accuracy_score(labels_db, preds_db)
    acc_all = accuracy_score(all_labels_db, all_preds_db)

    results = {
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'roc_fpr': roc_fpr,
        'roc_tpr': roc_tpr,
        'roc_thresholds': roc_thresholds,
        'chance': len(labels_db[labels_db == 1]) / float(len(labels_db)),
        'human': 0.8254,
        'average_precision': average_precision,
        'precision@50': precision_50,
        'precision@75': precision_75,
        'roc_auc_score': roc_auc,
        'roc_curve': roc_curve,
        'acc_binary': acc_binary,
        'acc_all': acc_all,
    }
    pprint({'average_precision': average_precision, 'roc_auc_score': roc_auc})
    checkpoint_name = args.checkpoint_file.replace('.pth.tar', '')
    out_file = os.path.join(
        analysis_dir, 
        f'{checkpoint_name}_pr_curve_{config.override_n_shots}shot.npz')
    np.savez(out_file, **results)
    print(f'Saved metrics to {out_file}.')

    # save the raw numbers of accuracy
    raw_out = {
        'indices': task_indices,
        'labels': labels_db,
        'probas': probas_db,
    }
    out_file = os.path.join(
        analysis_dir, 
        f'{checkpoint_name}_raw_{config.override_n_shots}shot.npz')
    np.savez(out_file, **raw_out)
    print(f'Saved raw outputs to {out_file}')


def precision_at_recall(precisions, recalls, at=0.5):
    return precisions[np.argmin(np.abs(recalls - at))]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='path to experiment directory')
    parser.add_argument('checkpoint_file', type=str, help='name of checkpoint')
    parser.add_argument('--n-shots', type=int, default=10, help='number of shots')
    parser.add_argument('--gpu-device', type=int, default=-1)
    args = parser.parse_args()
    pr_curve(args, gpu_device=args.gpu_device)
