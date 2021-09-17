import os
import copy
import time
import numpy as np
from tqdm import tqdm
from dotmap import DotMap
from itertools import chain
from sklearn import metrics
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup

from src.utils import utils
from src.agents.base import BaseAgent
from src.objectives.prototype import euclidean_dist, batch_euclidean_dist
from src.datasets.toy import ToySequenceClassification
from src.models.contracode import CodeTransformerEncoder
from src.models.context import ContextEncoder, AttentionEncoder
from src.objectives.prototype import batch_euclidean_dist


class BaseToyMetaAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.train_loss = []
        self.train_acc = []
        self.test_acc = []
        self.temp = []

    def _load_datasets(self):
        self.train_dataset = ToySequenceClassification(
            self.config.dataset.n_shots,
            self.config.dataset.n_queries,
            vocab_size=12,
            max_seq_len=5,
            num_class=4,
            num_examples_per_class=500,
            num_train_tasks=500,
            num_val_tasks=16,
            num_test_tasks=64,
            split='train',
            random_seed=1337,
        )
        self.val_dataset = ToySequenceClassification(
            self.config.dataset.n_shots,
            self.config.dataset.n_queries,
            vocab_size=12,
            max_seq_len=5,
            num_class=4,
            num_examples_per_class=500,
            num_train_tasks=500,
            num_val_tasks=16,
            num_test_tasks=64,
            split='val',
            random_seed=1337,
            data_dict=self.train_dataset.data_dict,
        )
        self.test_dataset = ToySequenceClassification(
            self.config.dataset.n_shots,
            self.config.dataset.n_queries,
            vocab_size=12,
            max_seq_len=5,
            num_class=4,
            num_examples_per_class=500,
            num_train_tasks=500,
            num_val_tasks=16,
            num_test_tasks=64,
            split='test',
            random_seed=1337,
            data_dict=self.train_dataset.data_dict,
        )

    def _load_loaders(self):
        self.train_loader, self.train_len = self._create_dataloader(
            self.train_dataset,
            self.config.optim.batch_size, 
            shuffle=True,
        )
        self.val_loader, self.val_len = self._create_dataloader(
            self.val_dataset,
            self.config.optim.batch_size, 
            shuffle=True,
        )
        self.test_loader, self.test_len = self._create_test_dataloader(
            self.test_dataset,
            self.config.optim.batch_size,
        )

    def _create_model(self):
        model = CodeTransformerEncoder(
            self.config.dataset.vocab_size,
            d_model=128,
            n_head=4,
            n_encoder_layers=4,
            d_ff=2048,
            dropout=0.1,
            activation="relu",
            norm=True,
            is_tam=self.config.model.task_tam,
            is_adapter=self.config.model.task_adapter,
        )
        self.model = model.to(self.device)

        if self.config.model.task_aware:
            num_tasks = self.train_len + self.val_len + self.test_len
            task_layer = nn.Embedding(num_tasks, 128)
            self.task_layer = task_layer.to(self.device)

        tau = nn.Parameter(torch.ones(1)).to(self.device)
        tau = tau.detach().requires_grad_(True)
        self.tau = tau

    def _all_parameters(self):
        parameters = [self.model.parameters(), [self.tau]]
        if self.config.model.task_aware:
            parameters.append(self.task_layer.parameters())
        return chain(*parameters)

    def _create_optimizer(self):
        optimizer = torch.optim.AdamW(
            self._all_parameters(),
            lr=self.config.optim.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.01,
        )
        num_training_steps = len(self.train_dataset) * self.config.optim.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.optim.warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.optim = optimizer
        self.scheduler = scheduler
        self.config.optim.use_scheduler = True

    def train_one_epoch(self):
        raise NotImplementedError

    def eval_test(self):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.current_epoch, self.config.optim.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()

            if (self.config.validate and epoch % self.config.validate_freq == 0):
                self.eval_test()

            self.save_checkpoint()

            if self.iter_with_no_improv > self.config.optim.patience:
                self.logger.info("Exceeded patience. Stop training...")
                break

    def save_metrics(self):
        out_dict = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'tau': self.tau,
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'loss': self.current_loss,
            'val_iteration': self.current_val_iteration,
            'val_metric': self.current_val_metric,
            'config': self.config,
            'train_acc': np.array(self.train_acc),
            'train_loss': np.array(self.train_loss),
            'test_acc': np.array(self.test_acc),
            'temp': np.array(self.temp),
        }
        if self.config.model.task_aware:
            out_dict['task_layer_state_dict'] = self.task_layer.state_dict()
        return out_dict

    def load_checkpoint(self, filename, checkpoint_dir=None, load_model=True, load_optim=False, load_epoch=False):
        if checkpoint_dir is None:
            checkpoint_dir = self.config.checkpoint_dir

        filename = os.path.join(checkpoint_dir, filename)

        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')

            if load_epoch:
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.current_val_iteration = checkpoint['val_iteration']
                self.train_loss = list(checkpoint['train_loss'])
                self.train_acc = list(checkpoint['train_acc'])
                self.test_acc = list(checkpoint['test_acc'])
                self.temp = list(checkpoint['temp'])
                self.current_val_metric = checkpoint['val_metric']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)
                if self.config.model.task_aware:
                    task_layer_state_dict = checkpoint['task_layer_state_dict']
                    self.task_layer.load_state_dict(task_layer_state_dict)
                self.tau.data = checkpoint['tau'].to(self.tau.device)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))

            return checkpoint

        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e


class ToyPrototypeNetAgent(BaseToyMetaAgent):

    def forward(self, batch, n_shots, n_queries):
        support_toks = batch['support_toks'].to(self.device)
        support_masks = batch['support_masks'].to(self.device)
        support_labs = batch['support_labs'].to(self.device)
        query_toks  = batch['query_toks'].to(self.device)
        query_masks = batch['query_masks'].to(self.device)
        query_labs = batch['query_labs'].to(self.device)

        batch_size = support_toks.size(0)
        n_ways = support_toks.size(1)
        seq_len = support_toks.size(-1)

        # support_toks: batch_size * n_ways * n_support x seq_len
        support_toks = support_toks.view(-1, seq_len)
        support_masks = support_masks.view(-1, seq_len)
        query_toks = query_toks.view(-1, seq_len)
        query_masks = query_masks.view(-1, seq_len)

        if self.config.model.task_aware:
            task = batch['task'].to(self.device)
            task_features = self.task_layer(task)
        else:
            task_features = None
        
        support_features = self.model(
            input_ids=support_toks, 
            attention_mask=support_masks,
            tam_embeds=task_features,
        )[0]
        query_features = self.model(
            input_ids=query_toks, 
            attention_mask=query_masks,
            tam_embeds=task_features,
        )[0]

        support_features = torch.mean(support_features, dim=1)
        query_features = torch.mean(query_features, dim=1)

        loss, top1, logprobas  = self.compute_loss(
            support_features.view(batch_size, n_ways, n_shots, -1),
            support_labs.view(batch_size, n_ways, n_shots),
            query_features.view(batch_size, n_ways, n_queries, -1),
            query_labs.view(batch_size, n_ways, n_queries),
        )
        return loss, top1, logprobas

    def compute_loss(self, support_features, support_targets, query_features, query_targets):
        batch_size, nway, nquery, dim = query_features.size()
        prototypes = torch.mean(support_features, dim=2)
        query_features_flat = query_features.view(batch_size, nway * nquery, dim)

        dists = self.tau * batch_euclidean_dist(query_features_flat, prototypes)
        logprobas = F.log_softmax(-dists, dim=2).view(batch_size, nway, nquery, -1)

        loss = -logprobas.gather(3, query_targets.unsqueeze(3)).squeeze()
        loss = loss.view(-1).mean()

        acc = utils.get_accuracy(logprobas.view(batch_size, nway*nquery, -1),
                                 query_targets.view(batch_size, nway*nquery)) 
        return loss, acc, logprobas

    def train_one_epoch(self):
        tqdm_batch = tqdm(total=len(self.train_loader),
                          desc="[Epoch {}]".format(self.current_epoch))
        self.model.train()
        loss_meter = utils.AverageMeter() 
        acc_meter = utils.AverageMeter()

        for batch in self.train_loader:
            n_shots = self.config.dataset.n_shots
            n_queries = self.config.dataset.n_queries
            loss, acc, _ = self.forward(batch, n_shots, n_queries)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.config.optim.use_scheduler:
                self.scheduler.step()

            with torch.no_grad():
                loss_meter.update(loss.item())
                acc_meter.update(acc)
                postfix = {"Loss": loss_meter.avg, "Acc": acc_meter.avg}
                self.current_iteration += 1

            tqdm_batch.set_postfix(postfix)
            tqdm_batch.update()
        tqdm_batch.close()

        self.current_loss = loss_meter.avg
        self.train_loss.append(loss_meter.avg)
        print(f'Meta-Train Tasks: {acc_meter.avg}')
        self.train_acc.append(acc_meter.avg)
        self.temp.append(self.tau.item())
        print(f'Temperature: {self.tau.item()}')

    def eval_split(self, name, loader):
        tqdm_batch = tqdm(total=len(loader), desc=f"[{name}]")
        self.model.eval()
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()

        with torch.no_grad():
            for batch in loader:
                n_shots = self.config.dataset.n_shots
                n_queries = self.config.dataset.n_queries
                loss, acc, _ = self.forward(batch, n_shots, n_queries)

                loss_meter.update(loss.item())
                acc_meter.update(acc)
                postfix = {"Loss": loss_meter.avg, "Acc": acc_meter.avg}

                tqdm_batch.set_postfix(postfix)
                tqdm_batch.update()
            tqdm_batch.close()

        return loss_meter.avg, acc_meter.avg

    def eval_test(self):
        _, acc = self.eval_split('Test', self.test_loader)
        print(f'Meta-Val Tasks: {acc}')

        self.current_val_iteration += 1
        self.current_val_metric = sum(acc)
        self.test_acc.append(acc)

        if self.current_val_metric >= self.best_val_metric:
            self.best_val_metric = self.current_val_metric
            self.iter_with_no_improv = 0
        else:
            self.iter_with_no_improv += 1


class ToyMatchingNetAgent(ToyPrototypeNetAgent):

    def _create_model(self):
        super()._create_model()
        self.fce_f = ContextEncoder(128, num_layers=self.config.model.fce.n_encoder_layers)
        self.fce_g = AttentionEncoder(128, unrolling_steps=self.config.model.fce.unrolling_steps)
        self.fce_f = self.fce_f.to(self.device)
        self.fce_g = self.fce_g.to(self.device)

    def _all_parameters(self):
        parameters = [self.model.parameters(), self.fce_f.parameters(), self.fce_g.parameters(), [self.tau]]
        if self.config.model.task_aware:
            parameters.append(self.task_layer.parameters())
        return chain(*parameters)

    def compute_loss(self, support_features, support_targets, query_features, query_targets):
        batch_size, n_ways, n_shots, d_model = support_features.size()
        n_queries = query_features.size(2)

        support_features = support_features.view(batch_size, n_ways * n_shots, d_model)
        query_features = query_features.view(batch_size, n_ways * n_queries, d_model)
        support_features = self.fce_f(support_features)
        query_features = self.fce_g(support_features, query_features)

        dists = self.tau * batch_euclidean_dist(query_features, support_features)
        attentions = F.softmax(-dists, dim=2)

        support_targets = support_targets.view(batch_size, n_ways * n_shots)
        support_targets_1hot = torch.zeros(batch_size, n_ways * n_shots, n_ways)
        support_targets_1hot = support_targets_1hot.to(support_targets.device)
        support_targets_1hot.scatter_(2, support_targets.unsqueeze(2), 1)

        probas = torch.bmm(attentions, support_targets_1hot)
        probas = probas.view(batch_size, n_ways, n_queries, n_ways)

        probas = probas.clamp(1e-8, 1 - 1e-8)
        logprobas = torch.log(probas)
       
        loss = -logprobas.gather(3, query_targets.unsqueeze(3)).squeeze()
        loss = loss.view(-1).mean()

        acc = utils.get_accuracy(logprobas.view(batch_size, n_ways * n_queries, -1),
                                 query_targets.view(batch_size, n_ways * n_queries)) 

        return loss, acc, logprobas

    def save_metrics(self):
        out_dict = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'fce_f_state_dict': self.fce_f.state_dict(),
            'fce_g_state_dict': self.fce_g.state_dict(),
            'tau': self.tau,
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'loss': self.current_loss,
            'val_iteration': self.current_val_iteration,
            'val_metric': self.current_val_metric,
            'config': self.config,
            'train_acc': np.array(self.train_acc),
            'train_loss': np.array(self.train_loss),
            'test_acc': np.array(self.test_acc),
            'temp': np.array(self.temp),
        }
        if self.config.model.task_aware:
            out_dict['task_layer_state_dict'] = self.task_layer.state_dict()
        return out_dict

    def load_checkpoint(self, filename, checkpoint_dir=None, load_model=True, load_optim=False, load_epoch=False):
        if checkpoint_dir is None:
            checkpoint_dir = self.config.checkpoint_dir
        filename = os.path.join(checkpoint_dir, filename)

        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')

            if load_epoch:
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.current_val_iteration = checkpoint['val_iteration']
                self.train_loss = list(checkpoint['train_loss'])
                self.train_acc = list(checkpoint['train_acc'])
                self.test_acc = list(checkpoint['test_acc'])
                self.temp = list(checkpoint['temp'])
                self.current_val_metric = checkpoint['val_metric']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                fce_f_state_dict = checkpoint['fce_f_state_dict']
                fce_g_state_dict = checkpoint['fce_g_state_dict']
                self.model.load_state_dict(model_state_dict)
                self.fce_f.load_state_dict(fce_f_state_dict)
                self.fce_g.load_state_dict(fce_g_state_dict)
                if self.config.model.task_aware:
                    task_layer_state_dict = checkpoint['task_layer_state_dict']
                    self.task_layer.load_state_dict(task_layer_state_dict)
                self.tau.data = checkpoint['tau'].to(self.tau.device)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))

            return checkpoint

        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e
