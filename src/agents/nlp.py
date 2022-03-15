import os
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    RobertaConfig,
    RobertaModel,
    get_linear_schedule_with_warmup,
)
from src.utils import utils
from src.models.codelstm import CodeLSTMEncoder
from src.models.monkeypatch import RobertaModel
from src.models.context import ContextEncoder, AttentionEncoder
from src.agents.base import BaseAgent
from src.objectives.prototype import batch_euclidean_dist
from src.datasets.nlp import MetaNewsGroup, SupNewsGroup
from src.datasets.amazon import FewShotAmazonSentiment
from src.datasets.text import (
    FewShot20News,
    FewShotAmazon,
    FewShotHuffPost,
    FewShotRCV1,
    FewShotReuters,
    FewShotFewRel,
)


class BaseNLPMetaAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.train_loss = []
        self.train_acc = []
        self.test_acc = []
        self.test_acc_stdevs = []
        self.temp = []

    def _load_datasets(self):
        if self.config.dataset.name == 'newsgroup':
            self.train_dataset = MetaNewsGroup(
                data_root=self.config.data_root,
                n_ways=self.config.dataset.train.n_ways,
                n_shots=self.config.dataset.train.n_shots,
                n_queries=self.config.dataset.train.n_queries,
                smlmt_tasks_factor=self.config.dataset.train.smlmt_tasks_factor,
                train=True,
            )
            self.test_dataset = MetaNewsGroup(
                data_root=self.config.data_root,
                n_ways=self.config.dataset.train.n_ways,
                n_shots=self.config.dataset.train.n_shots,
                n_queries=self.config.dataset.test.n_queries,
                smlmt_tasks_factor=self.config.dataset.train.smlmt_tasks_factor,
                train=False,
            )
        elif self.config.dataset.name == 'amazon':
            self.train_dataset = FewShotAmazonSentiment(
                data_root=self.config.dataset.data_root,
                n_shots=self.config.dataset.train.n_shots,
                n_queries=self.config.dataset.train.n_queries,
                smlmt_tasks_factor=self.config.dataset.train.smlmt_tasks_factor,
                train=True,
            )
            # NOTE: val/dev dataset.
            self.test_dataset = FewShotAmazonSentiment(
                data_root=self.config.dataset.data_root,
                n_shots=self.config.dataset.test.n_shots,
                n_queries=self.config.dataset.test.n_queries,
                smlmt_tasks_factor=self.config.dataset.train.smlmt_tasks_factor,
                train=False,
            )
        elif 'fs' in self.config.dataset.name:
            class_dict = {
                'fs_20news': FewShot20News,
                'fs_amazon': FewShotAmazon,
                'fs_huffpost': FewShotHuffPost,
                'fs_rcv1': FewShotRCV1,
                'fs_reuters': FewShotReuters,
                'fs_fewrel': FewShotFewRel,
            }
            DatasetClass = class_dict[self.config.dataset.name]
            self.train_dataset = DatasetClass(
                data_root=self.config.dataset.data_root,
                n_ways=self.config.dataset.train.n_ways,
                n_shots=self.config.dataset.train.n_shots,
                n_queries=self.config.dataset.train.n_queries,
                split='train',
            )
            # validation (test is not used here)
            self.test_dataset = DatasetClass(
                data_root=self.config.dataset.data_root,
                n_ways=self.config.dataset.test.n_ways,
                n_shots=self.config.dataset.test.n_shots,
                n_queries=self.config.dataset.test.n_queries,
                split='val',
            )
        else:
            raise Exception(f'Dataset {self.config.dataset.name} not supported.')

    def _load_loaders(self):
        self.train_loader, self.train_len = self._create_dataloader(
            self.train_dataset,
            self.config.optim.batch_size,
            shuffle=True,
        )
        self.test_loader, self.test_len = self._create_test_dataloader(
            self.test_dataset,
            self.config.optim.batch_size,
        )

    def _create_model(self):
        if self.config.model.name == 'roberta':
            model = RobertaModel.from_pretrained(self.config.model.config, is_tam=self.config.model.task_tam)
            utils.reset_model_for_training(model)

            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.pooler.parameters():
                    param.requires_grad = True
                # only allow some parameters to be finetuned
                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

            self.model = model.to(self.device)

        elif self.config.model.name == 'roberta_scratch':
            config = RobertaConfig.from_pretrained(self.config.model.config)
            model = RobertaModel(config,  is_tam=self.config.model.task_tam)
            utils.reset_model_for_training(model)

            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False

                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

        elif self.config.model.name == 'lstm':
            model = CodeLSTMEncoder(
                self.train_dataset.vocab_size,
                d_model=768,
                n_encoder_layers=4,
                dropout=0.1,
            )

        else:
            raise Exception(f'Model {self.config.model.name} not supported.')

        self.model = model.to(self.device)

        tau = nn.Parameter(torch.ones(1)).to(self.device)
        tau = tau.detach().requires_grad_(True)
        self.tau = tau

    def _all_parameters(self):
        return chain(self.model.parameters(), [self.tau])

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
            print(f"Epoch: {epoch}")
            self.current_epoch = epoch
            self.write_to_file(str(epoch) + self.train_one_epoch())

            if (self.config.validate and epoch % self.config.validate_freq == 0):
                self.write_to_file(self.eval_test())

            self.save_checkpoint()

            if self.iter_with_no_improv > self.config.optim.patience:
                self.logger.info("Exceeded patience. Stop training...")
                break

            # increase the number of ways on a patience-based system
            print(f"self.wi_mode; {self.wi_mode}")
            print(f"self.iter_with_no_improv_since_wi: {self.iter_with_no_improv_since_wi}")
            print(f"self.config.dataset.train.ways_patience: {self.config.dataset.train.ways_patience}")
            print(f"self.config.dataset.train.n_ways: {self.config.dataset.train.n_ways}")
            print(f"self.config.dataset.train.max_ways: {self.config.dataset.train.max_ways}")
            if self.wi_mode == "patience":
                if self.iter_with_no_improv_since_wi > self.config.dataset.train.ways_patience:
                    self.iter_with_no_improv_since_wi = 0
                    self.best_val_metric_since_wi = 0
                    if self.config.dataset.train.n_ways < self.config.dataset.train.max_ways:
                        self.config.dataset.train.n_ways = min(self.config.dataset.train.max_ways,
                                                               self.config.dataset.train.n_ways +
                                                               self.config.dataset.train.ways_inc_by)
                        self.train_dataset.update_n_ways(self.config.dataset.train.n_ways)

            # decrease the number of shots on a patience-based system
            if self.shot_mode == "patience":
                if self.iter_with_no_improv_since_sd > self.config.dataset.train.sd_patience:
                    self.iter_with_no_improv_since_sd = 0
                    self.best_val_metric_since_sd = 0
                    if self.config.dataset.train.n_shots > self.config.dataset.train.min_shots:
                        self.config.dataset.train.n_shots = max(self.config.dataset.train.min_shots,
                                                          self.config.dataset.train.n_shots -
                                                          self.config.dataset.train.decay_by)
                        self.train_dataset.update_n_shots(self.config.dataset.train.n_shots)

            # Decay the shot
            if self.shot_mode == "step_decay":
                if epoch % self.config.dataset.train.decay_every == self.config.dataset.train.decay_every - 1:
                    self.config.dataset.train.n_shots = max(self.config.dataset.train.min_shots,
                                                            self.config.dataset.train.n_shots -
                                                            self.config.dataset.train.decay_by)
                    self.train_dataset.update_n_shots(self.config.dataset.train.n_shots)


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
            'test_acc_stdevs': np.array(self.test_acc_stdevs),
            'temp': np.array(self.temp),
        }
        return out_dict

    def load_checkpoint(
            self,
            filename,
            checkpoint_dir=None,
            load_model=True,
            load_optim=False,
            load_epoch=False,
        ):
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
                self.test_acc_stdevs = list(checkpoint['test_acc_stdevs'])
                self.temp = list(checkpoint['temp'])
                self.current_val_metric = checkpoint['val_metric']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)
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


class NLPPrototypeNetAgent(BaseNLPMetaAgent):

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

    def compute_masked_means(self, outputs, masks):
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        masked_outputs = outputs * masks_dim
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def forward(self, batch, n_shots, n_queries):
        support_toks = batch['support_toks'].to(self.device)
        support_lens = batch['support_lens'].to(self.device)
        support_masks = batch['support_masks'].to(self.device)
        support_labs = batch['support_labs'].to(self.device)
        query_toks  = batch['query_toks'].to(self.device)
        query_lens = batch['query_lens'].to(self.device)
        query_masks = batch['query_masks'].to(self.device)
        query_labs = batch['query_labs'].to(self.device)

        batch_size = support_toks.size(0)
        n_ways = support_toks.size(1)
        seq_len = support_toks.size(-1)

        # support_toks: batch_size*n_ways*n_shots x seq_len
        support_toks = support_toks.view(-1, seq_len)
        support_lens = support_lens.view(-1)
        support_masks = support_masks.view(-1, seq_len).long()

        # query_toks: batch_size*n_ways*n_queries x seq_len
        query_toks = query_toks.view(-1, seq_len)
        query_lens = query_lens.view(-1)
        query_masks = query_masks.view(-1, seq_len).long()

        if self.config.model.task_tam:
            side_info = batch['side_info'].to(self.device)  # 1 x 768
            # support_sides : batch_size*n_ways*n_shots x bert_dim
            # query_sides   : batch_size*n_ways*n_queries x bert_dim
            support_sides = side_info.repeat(batch_size * n_ways * n_shots, 1)
            query_sides = side_info.repeat(batch_size * n_ways * n_queries, 1)

            # support_sides: batch_size*n_ways*n_shots x 1 x bert_dim
            # query_sides  : batch_size*n_ways*n_queries x 1 x bert_dim
            support_sides = support_sides.unsqueeze(1)
            query_sides = query_sides.unsqueeze(1)
        else:
            support_sides, query_sides = None, None

        if self.config.model.name == 'lstm':
            support_features = self.model(support_toks, support_lens, tam_embeds=support_sides)
            query_features = self.model(query_toks, query_lens, tam_embeds=query_sides)
        else:
            support_features = self.model(input_ids=support_toks, attention_mask=support_masks, tam_embeds=support_sides)[0]
            query_features = self.model(input_ids=query_toks, attention_mask=query_masks, tam_embeds=query_sides)[0]
            support_features = self.compute_masked_means(support_features, support_masks)
            query_features = self.compute_masked_means(query_features, query_masks)

        loss, top1, logprobas  = self.compute_loss(
            support_features.view(batch_size, n_ways, n_shots, -1),
            support_labs.view(batch_size, n_ways, n_shots),
            query_features.view(batch_size, n_ways, n_queries, -1),
            query_labs.view(batch_size, n_ways, n_queries),
        )
        return loss, top1, logprobas

    def train_one_epoch(self):
        if not self.config.fasrc:
            tqdm_batch = tqdm(total=len(self.train_loader),
                              desc="[Epoch {}]".format(self.current_epoch))
        self.model.train()
        loss_meter = utils.AverageMeter()
        all_task_types = range(2)
        acc_meters = [utils.AverageMeter() for _ in all_task_types]

        for batch in self.train_loader:
            n_shots = self.config.dataset.train.n_shots
            n_queries = self.config.dataset.train.n_queries
            loss, acc, _ = self.forward(batch, n_shots, n_queries)
            task_type = batch['task_type'].cpu().numpy()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.config.optim.use_scheduler:
                self.scheduler.step()

            with torch.no_grad():
                loss_meter.update(loss.item())
                postfix = {"Loss": loss_meter.avg}
                for t_, t in enumerate(all_task_types):
                    if sum(task_type == t) > 0:
                        acc_meters[t_].update(acc[task_type == t].mean())
                        postfix[f"Acc{t}"] = acc_meters[t_].avg
                self.current_iteration += 1
            if not self.config.fasrc:
                tqdm_batch.set_postfix(postfix)
                tqdm_batch.update()
        if not self.config.fasrc:
            tqdm_batch.close()

        self.current_loss = loss_meter.avg
        self.train_loss.append(loss_meter.avg)
        accuracies = [acc_meters[t].avg for t in all_task_types]
        print(f'Meta-Train Tasks: {accuracies}')
        self.train_acc.append(accuracies)
        self.temp.append(self.tau.item())
        print(f'Temperature: {self.tau.item()}')
        return f'Meta-Train Tasks: {accuracies}'

    def eval_split(self, name, loader, verbose=False):
        if not self.config.fasrc:
            tqdm_batch = tqdm(total=len(loader), desc=f"[{name}]")
        self.model.eval()
        loss_meter = utils.AverageMeter()
        all_task_types = range(2)
        acc_meters = [utils.AverageMeter() for _ in all_task_types]
        acc_stores = [[] for _ in all_task_types]
        if verbose:
            accuracies = [[] for _ in all_task_types]

        with torch.no_grad():
            for batch in loader:
                n_shots = self.config.dataset.train.n_shots
                if self.shot_mode == "step_decay":
                    n_shots = self.config.dataset.test.n_shots
                n_queries = self.config.dataset.test.n_queries
                loss, acc, _ = self.forward(batch, n_shots, n_queries)
                task_type = batch['task_type'].cpu().numpy()

                loss_meter.update(loss.item())
                postfix = {"Loss": loss_meter.avg}
                for t_, t in enumerate(all_task_types):
                    if sum(task_type == t) > 0:
                        acc_meters[t_].update(acc[task_type == t].mean())
                        postfix[f"Acc{t}"] = acc_meters[t_].avg
                        acc_stores[t_].append(acc[task_type == t].mean())
                        if verbose:
                            accuracies[t_].append(acc[task_type == t])
                if not self.config.fasrc:
                    tqdm_batch.update()
            if not self.config.fasrc:
                tqdm_batch.close()

        if not verbose:
            accuracies = [acc_meters[t].avg for t in all_task_types]
        elif verbose:
            max_accuracy_len = max(list(map(lambda x: max(list(map(lambda y: len(y), x)), default=0), accuracies)), default=0)
            max_task_len = max(list(map(lambda x: len(x), accuracies)))
            accuracies = np.stack(list(map(lambda x: np.vstack(x) if len(x) > 0 else np.zeros((max_task_len, max_accuracy_len)), accuracies)))
        accuracy_stdevs = [np.std(acc_stores[t]) for t in all_task_types]
        return loss_meter.avg, accuracies, accuracy_stdevs

    def eval_test(self):
        _, acc_means, acc_stdevs = self.eval_split('Test', self.test_loader)
        print(f'Meta-Val Tasks: {acc_means}')

        self.current_val_iteration += 1
        self.current_val_metric = sum(acc_means)
        self.test_acc.append(acc_means)
        self.test_acc_stdevs.append(acc_stdevs)

        if self.current_val_metric >= self.best_val_metric:
            self.best_val_metric = self.current_val_metric
            self.iter_with_no_improv = 0
        else:
            self.iter_with_no_improv += 1

        # Patience for WI
        if self.current_val_metric >= self.best_val_metric_since_wi:
            self.best_val_metric_since_wi = self.current_val_metric
            self.iter_with_no_improv_since_wi = 0
        else:
            self.iter_with_no_improv_since_wi += 1

        # Patience for SD
        if self.current_val_metric >= self.best_val_metric_since_sd:
            self.best_val_metric_since_sd = self.current_val_metric
            self.iter_with_no_improv_since_sd = 0
        else:
            self.iter_with_no_improv_since_sd += 1

        return f'Meta-Val Tasks: {acc_means}'


class NLPMatchingNetAgent(NLPPrototypeNetAgent):

    def _create_model(self):
        super()._create_model()
        self.fce_f = ContextEncoder(768, num_layers=self.config.model.fce.n_encoder_layers)
        self.fce_g = AttentionEncoder(768, unrolling_steps=self.config.model.fce.unrolling_steps)
        self.fce_f = self.fce_f.to(self.device)
        self.fce_g = self.fce_g.to(self.device)

    def _all_parameters(self):
        return chain(self.model.parameters(), self.fce_f.parameters(), self.fce_g.parameters(), [self.tau])

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
            'test_acc_stdevs': np.array(self.test_acc_stdevs),
            'temp': np.array(self.temp),
        }
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
                self.test_acc_stdevs = list(checkpoint['test_acc_stdevs'])
                self.temp = list(checkpoint['temp'])
                self.current_val_metric = checkpoint['val_metric']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                fce_f_state_dict = checkpoint['fce_f_state_dict']
                fce_g_state_dict = checkpoint['fce_g_state_dict']
                self.model.load_state_dict(model_state_dict)
                self.fce_f.load_state_dict(fce_f_state_dict)
                self.fce_g.load_state_dict(fce_g_state_dict)
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


class BaseNLPSupAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.train_loss = []
        self.train_acc  = []
        self.test_acc   = []

    def _load_datasets(self):
        if self.config.dataset.name == 'newsgroup':
            self.train_dataset = SupNewsGroup(
                task_index=self.config.dataset.task_index,
                data_root=self.config.data_root,
                n_ways=self.config.dataset.train.n_ways,
                n_shots=self.config.dataset.train.n_shots,
                n_queries=self.config.dataset.train.n_queries,
                train=True,
            )
            self.test_dataset = SupNewsGroup(
                task_index=self.config.dataset.task_index,
                data_root=self.config.data_root,
                n_ways=self.config.dataset.train.n_ways,
                n_shots=self.config.dataset.train.n_shots,
                n_queries=self.config.dataset.test.n_queries,
                train=False,
            )
        else:
            raise Exception(f'Dataset {self.config.dataset.name} not supported.')

    def _load_loaders(self):
        self.train_loader, self.train_len = self._create_dataloader(
            self.train_dataset,
            self.config.optim.batch_size,
            shuffle=True,
        )
        self.test_loader, self.test_len = self._create_test_dataloader(
            self.test_dataset,
            self.config.optim.batch_size,
        )

    def _create_model(self):
        model = RobertaModel.from_pretrained(self.config.model.config, is_tam=self.config.model.task_tam)
        utils.reset_model_for_training(model)

        if self.config.model.finetune:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.pooler.parameters():
                param.requires_grad = True
            # only allow some parameters to be finetuned
            for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                param.requires_grad = True

        self.model = model.to(self.device)

        classifier = nn.Linear(self.config.model.d_model, 1)
        self.classifier = classifier.to(self.device)

    def _create_optimizer(self):
        optimizer = torch.optim.AdamW(
            chain(self.model.parameters(), self.classifier.parameters()),
            lr=self.config.optim.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.01,
        )
        self.optim = optimizer

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def forward(self, batch):
        batch_size, n_ways, n_shots, max_seq_len = batch['tokens'].size()

        tokens = batch['tokens'].to(self.device)
        masks = batch['masks'].to(self.device)
        labels = batch['labels'].to(self.device)

        tokens = tokens.view(batch_size * n_ways * n_shots, max_seq_len)
        masks = masks.view(batch_size * n_ways * n_shots, max_seq_len)
        labels = labels.view(batch_size * n_ways * n_shots)

        if self.config.model.task_tam:
            side_info = batch['side_info'].to(self.device)
            side_info = side_info.repeat(batch_size * n_ways * n_shots, 1)
            features = self.model(input_ids=tokens, attention_mask=masks,
                                  tam_embeds=side_info.unsqueeze(1))[0]
        else:
            features = self.model(input_ids=tokens, attention_mask=masks)[0]

        features = self.compute_masked_means(features, masks)

        logits = self.classifier(F.relu(features))
        probas = torch.sigmoid(logits)
        labels = labels.unsqueeze(1).float()
        loss = F.binary_cross_entropy(probas, labels)

        with torch.no_grad():
            preds = torch.round(probas)
            correct = preds.eq(labels).sum().item()
            acc = 100. / (batch_size * n_ways * n_shots) * correct

        return loss, acc, probas

    def train_one_epoch(self):
        if not self.config.fasrc:
            tqdm_batch = tqdm(total=len(self.train_loader),
                              desc="[Epoch {}]".format(self.current_epoch))
        self.model.train()
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()

        for batch in self.train_loader:
            loss, acc, _ = self.forward(batch)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            with torch.no_grad():
                loss_meter.update(loss.item())
                acc_meter.update(acc)
                postfix = {"Loss": loss_meter.avg, "Acc": acc_meter.avg}
                self.current_iteration += 1
                if not self.config.fasrc:
                    tqdm_batch.set_postfix(postfix)
            if not self.config.fasrc:
                tqdm_batch.update()
        if not self.config.fasrc:
            tqdm_batch.close()

        self.current_loss = loss_meter.avg
        self.train_loss.append(loss_meter.avg)
        print(f'Meta-Train Tasks: {acc_meter.avg}')
        self.train_acc.append(acc_meter.avg)

    def eval_split(self, name, loader):
        if not self.config.fasrc:
            tqdm_batch = tqdm(total=len(loader), desc=f"[{name}]")
        self.model.eval()
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()

        with torch.no_grad():
            for batch in loader:
                loss, acc, _ = self.forward(batch)
                loss_meter.update(loss.item())
                acc_meter.update(acc)
                postfix = {"Loss": loss_meter.avg, "Acc": acc_meter.avg}
                if not self.config.fasrc:
                    tqdm_batch.set_postfix(postfix)
                    tqdm_batch.update()
            if not self.config.fasrc:
                tqdm_batch.close()

        return loss_meter.avg, acc_meter.avg

    def eval_test(self):
        _, acc = self.eval_split('Test', self.test_loader)
        print(f'Meta-Val Tasks: {acc}')

        self.current_val_iteration += 1
        self.current_val_metric = acc
        self.test_acc.append(acc)

        if self.current_val_metric >= self.best_val_metric:
            self.best_val_metric = self.current_val_metric
            self.iter_with_no_improv = 0
        else:
            self.iter_with_no_improv += 1

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
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'loss': self.current_loss,
            'val_iteration': self.current_val_iteration,
            'val_metric': self.current_val_metric,
            'config': self.config,
            'train_acc': np.array(self.train_acc),
            'train_loss': np.array(self.train_loss),
            'test_acc': np.array(self.test_acc),
        }
        return out_dict

    def load_checkpoint(
            self,
            filename,
            checkpoint_dir=None,
            load_model=True,
            load_optim=False,
            load_epoch=False,
        ):
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
                self.current_val_metric = checkpoint['val_metric']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))

            return checkpoint

        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e
