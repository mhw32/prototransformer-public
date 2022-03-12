import os
import numpy as np
from tqdm import tqdm
from dotmap import DotMap
from itertools import chain
from collections import OrderedDict
from sklearn.cluster import KMeans
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaForMaskedLM,
    get_linear_schedule_with_warmup,
)

from src.utils import utils
from src.models.codelstm import CodeLSTMEncoder
# from src.models.contracode import CodeTransformerEncoder # I think this is residual for a model class they don't use
from src.models.monkeypatch import RobertaModel, RobertaForMaskedLM
from src.models.context import ContextEncoder, AttentionEncoder
from src.models.relation import RelationNetwork
from src.models.task import TaskEmbedding
from src.models.signatures import DistSign
from src.agents.base import BaseAgent
from src.objectives.prototype import batch_euclidean_dist
from src.datasets.feedback import MetaExamSolutions, SupervisedExamSolutions, MetaDTSolutions


class BaseCodeMetaAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.train_loss = []
        self.train_acc  = []
        self.test_acc   = []
        self.temp       = []

    def _load_datasets(self):
        if not self.config.cuda:
            roberta_device = 'cpu'
        else:
            roberta_device = f'cuda:{self.config.gpu_device}'

        self.train_dataset = MetaExamSolutions(
            data_root=self.config.data_root,
            n_shots=self.config.dataset.train.n_shots,
            n_queries=self.config.dataset.test.n_queries,
            train=True,
            vocab=None,
            train_frac=self.config.dataset.train_frac,
            obfuscate_names=self.config.dataset.obfuscate_names,
            max_num_var=self.config.dataset.max_num_var,
            max_num_func=self.config.dataset.max_num_func,
            max_seq_len=self.config.dataset.max_seq_len,
            min_occ=self.config.dataset.min_occ,
            augment_by_names=self.config.dataset.train.augment_by_names,
            augment_by_rubric=self.config.dataset.train.augment_by_rubric,
            roberta_rubric=self.config.dataset.train.roberta_rubric,
            roberta_prompt=self.config.dataset.train.roberta_prompt,
            roberta_tokenize=self.config.dataset.roberta_tokenize,
            roberta_config=self.config.model.config,
            roberta_device=roberta_device,
            conservative=self.config.dataset.train.conservative,
            cloze_tasks_factor=self.config.dataset.train.cloze_tasks_factor,
            execution_tasks_factor=self.config.dataset.train.execution_tasks_factor,
            smlmt_tasks_factor=self.config.dataset.train.smlmt_tasks_factor,
            pad_to_max_num_class=self.config.optim.batch_size > 1,
            hold_out_split=self.config.dataset.hold_out_split,
            hold_out_category=self.config.dataset.hold_out_category,
            enforce_binary=self.config.dataset.enforce_binary,
        )
        self.test_dataset = MetaExamSolutions(
            data_root=self.config.data_root,
            n_shots=self.config.dataset.train.n_shots,
            n_queries=self.config.dataset.test.n_queries,
            train=False,
            vocab=self.train_dataset.vocab,
            train_frac=self.config.dataset.train_frac,
            obfuscate_names=self.config.dataset.obfuscate_names,
            max_num_var=self.config.dataset.max_num_var,
            max_num_func=self.config.dataset.max_num_func,
            max_seq_len=self.config.dataset.max_seq_len,
            min_occ=self.config.dataset.min_occ,
            roberta_rubric=self.train_dataset.roberta_rubric,
            roberta_prompt=self.config.dataset.train.roberta_prompt,
            roberta_tokenize=self.config.dataset.roberta_tokenize,
            roberta_config=self.config.model.config,
            roberta_device=roberta_device,
            pad_to_max_num_class=self.config.optim.batch_size > 1,
            conservative=self.config.dataset.train.conservative,
            cloze_tasks_factor=self.train_dataset.cloze_tasks_factor,
            execution_tasks_factor=self.train_dataset.execution_tasks_factor,
            smlmt_tasks_factor=self.config.dataset.train.smlmt_tasks_factor,
            hold_out_split=self.config.dataset.hold_out_split,
            hold_out_category=self.config.dataset.hold_out_category,
            enforce_binary=self.config.dataset.enforce_binary,
        )

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
        if self.config.model.name == 'transformer':
            vocab_size = self.train_dataset.vocab_size
            model = CodeTransformerEncoder(
                vocab_size,
                d_model=self.config.model.d_model,
                n_head=self.config.model.n_head,
                n_encoder_layers=self.config.model.n_encoder_layers,
                d_ff=self.config.model.d_ff,
                dropout=0.1,
                activation="relu",
                norm=True,
                pad_id=self.train_dataset.pad_index,
                is_tam=self.config.model.task_tam,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )

        elif self.config.model.name == 'roberta':
            model = RobertaModel.from_pretrained(
                self.config.model.config,
                is_tam=self.config.model.task_tam,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )
            # set everything to requires_grad = True
            utils.reset_model_for_training(model)

            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.pooler.parameters():
                    param.requires_grad = True
                # only allow some parameters to be finetuned
                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

        elif self.config.model.name == 'roberta_codesearch':
            model = RobertaForMaskedLM.from_pretrained(
                'roberta-base',
                is_tam=self.config.model.task_tam,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )
            # load the codesearch checkpoint
            checkpoint = torch.load(
                self.config.model.codesearch_checkpoint_path,
                map_location='cpu',
            )
            raw_state_dict = checkpoint['state_dict']
            state_dict = OrderedDict()
            for k, v in raw_state_dict.items():
                new_k = '.'.join(k.split('.')[1:])
                state_dict[new_k] = v
            model.load_state_dict(state_dict, strict=False)
            model = model.roberta  # only keep roberta
            utils.reset_model_for_training(model)
            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False
                # only allow some parameters to be finetuned
                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

        elif self.config.model.name == 'roberta_scratch':
            config = RobertaConfig.from_pretrained(self.config.model.config)
            model = RobertaModel(
                config,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )
            # set everything to requires_grad = True
            utils.reset_model_for_training(model)

            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False
                # only allow some parameters to be finetuned
                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

        elif self.config.model.name == 'lstm':
            assert not self.config.model.task_tadam, "TADAM not support for LSTMs."
            assert not self.config.model.task_adapter, "Adapter not support for LSTMs."

            vocab_size = len(self.train_dataset.vocab['w2i'])
            model = CodeLSTMEncoder(
                vocab_size,
                d_model=self.config.model.d_model,
                n_encoder_layers=self.config.model.n_encoder_layers,
                dropout=0.1,
                is_tadam=self.config.model.task_tadam,
            )
        else:
            raise Exception(f'Model {self.config.model.name} not supported.')

        self.model = model.to(self.device)
        d_model = self.config.model.d_model
        bert_dim = 768

        if self.config.model.task_concat:
            # combine program embedding and rubric/question at the end of the forward pass
            concat_fusor = TaskEmbedding(d_model+bert_dim*2, d_model, hid_dim=d_model)
            self.concat_fusor = concat_fusor.to(self.device)

        tau = nn.Parameter(torch.ones(1)).to(self.device)
        tau = tau.detach().requires_grad_(True)
        self.tau = tau

    def _all_parameters(self):
        all_parameters = [self.model.parameters(), [self.tau]]
        if self.config.model.task_concat:
            all_parameters.append(self.concat_fusor.parameters())
        return chain(*all_parameters)

    def _create_optimizer(self):
        if self.config.model.name in ['roberta', 'roberta_mlm', 'roberta_scratch']:
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
        else:
            # this is the one used for Adam
            self.optim = torch.optim.AdamW(
                self._all_parameters(),
                lr=self.config.optim.learning_rate,
                betas=(0.9, 0.98),
                weight_decay=self.config.optim.weight_decay,
            )

            if self.config.optim.use_scheduler:

                def schedule(step_num):
                    d_model = self.config.model.d_model
                    warmup_steps = self.config.optim.warmup_steps
                    step_num += 1
                    lrate = d_model**(-0.5) * min(step_num**(-0.5), step_num * warmup_steps**(-1.5))
                    return lrate

                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, schedule)

    def train_one_epoch(self):
        raise NotImplementedError

    def eval_test(self):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.current_epoch, self.config.optim.num_epochs):
            print(f"Epoch: {epoch}") # to write to tee
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
        if self.config.model.task_concat:
            out_dict['concat_fusor_state_dict'] = self.concat_fusor.state_dict()
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
                self.temp = list(checkpoint['temp'])
                self.current_val_metric = checkpoint['val_metric']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)
                self.tau.data = checkpoint['tau'].to(self.tau.device)

                if self.config.model.task_concat:
                    concat_fusor_state_dict = checkpoint['concat_fusor_state_dict']
                    self.concat_fusor.load_state_dict(concat_fusor_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))

            return checkpoint

        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e


class CodePrototypeNetAgent(BaseCodeMetaAgent):

    def compute_loss(
            self,
            support_features,
            support_targets,
            query_features,
            query_targets,
        ):
        batch_size, nway, nquery, dim = query_features.size()
        prototypes = torch.mean(support_features, dim=2)
        query_features_flat = query_features.view(batch_size, nway * nquery, dim)

        # batch-based euclidean dist between prototypes and query_features_flat
        # dists: batch_size x nway * nquery x nway
        dists = self.tau * batch_euclidean_dist(query_features_flat, prototypes)
        logprobas = F.log_softmax(-dists, dim=2).view(batch_size, nway, nquery, -1)

        loss = -logprobas.gather(3, query_targets.unsqueeze(3)).squeeze()
        loss = loss.view(-1).mean()

        acc = utils.get_accuracy(logprobas.view(batch_size, nway*nquery, -1),
                                 query_targets.view(batch_size, nway*nquery))
        return loss, acc, logprobas

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

    def forward(self, batch, n_shots, n_queries):
        # NOTE: n_shots, n_queries are unused

        support_toks = batch['support_toks'].to(self.device)
        support_lens = batch['support_lens'].to(self.device)
        support_masks = batch['support_masks'].to(self.device)
        support_labs = batch['support_labs'].to(self.device)
        query_toks  = batch['query_toks'].to(self.device)
        query_lens = batch['query_lens'].to(self.device)
        query_masks = batch['query_masks'].to(self.device)
        query_labs = batch['query_labs'].to(self.device)
        rubric_embs = batch['rubric_embs'].to(self.device)
        prompt_embs = batch['prompt_embs'].to(self.device)
        bert_dim = batch['rubric_embs'].size(-1)

        if self.config.override_n_shots != DotMap():  # NOTE: used in test time to vary supervision
            assert self.config.override_n_shots <= support_toks.size(2)

            if self.config.override_n_shots == 0:
                # separate procedure for zero-shot
                return self.zero_shot_forward(batch, n_shots, n_queries)

            # if > 0, we can just pretend like we have less
            support_toks = support_toks[:, :, :self.config.override_n_shots, :].contiguous()
            support_lens = support_lens[:, :, :self.config.override_n_shots].contiguous()
            support_masks = support_masks[:, :, :self.config.override_n_shots, :].contiguous()
            support_labs = support_labs[:, :, :self.config.override_n_shots].contiguous()

        batch_size = support_toks.size(0)
        n_ways = support_toks.size(1)
        n_support = support_toks.size(2)
        n_query = query_toks.size(2)
        seq_len = support_toks.size(-1)

        # support_toks: batch_size*n_ways*n_support x seq_len
        support_toks = support_toks.view(-1, seq_len)
        support_lens = support_lens.view(-1)
        support_masks = support_masks.view(-1, seq_len).long()
        query_toks = query_toks.view(-1, seq_len)
        query_lens = query_lens.view(-1)
        query_masks = query_masks.view(-1, seq_len).long()

        # rubric_embs: batch_size*n_ways x bert_dim
        rubric_embs = rubric_embs.view(-1, bert_dim)
        support_rubric_embs = rubric_embs.unsqueeze(1).repeat(1, n_support, 1)
        # support_rubric_embs: batch_size*n_ways*n_support x bert_dim
        support_rubric_embs = support_rubric_embs.view(-1, bert_dim)
        # query_rubric_embs: batch_size*n_ways*n_query x bert_dim
        query_rubric_embs = rubric_embs.unsqueeze(1).repeat(1, n_query, 1)
        query_rubric_embs = query_rubric_embs.view(-1, bert_dim)

        # prompt_embs: batch_size*n_ways x bert_dim
        prompt_embs = prompt_embs.view(-1, bert_dim)
        support_prompt_embs = prompt_embs.unsqueeze(1).repeat(1, n_support, 1)
        # support_rubric_embs: batch_size*n_ways*n_support x bert_dim
        support_prompt_embs = support_prompt_embs.view(-1, bert_dim)
        query_prompt_embs = prompt_embs.unsqueeze(1).repeat(1, n_query, 1)
        # query_rubric_embs: batch_size*n_ways*n_prompt x bert_dim
        query_prompt_embs = query_prompt_embs.view(-1, bert_dim)

        if self.config.model.name == 'lstm':
            # support_tam_features : ... x 2 x bert_dim
            # query_tam_features   : ... x 2 x bert_dim
            support_tam_features = torch.cat([support_rubric_embs.unsqueeze(1),
                                              support_prompt_embs.unsqueeze(1)], dim=1)
            query_tam_features = torch.cat([query_rubric_embs.unsqueeze(1),
                                            query_prompt_embs.unsqueeze(1)], dim=1)
            # support_features: batch_size*n_ways*n_support x dim
            # query_features: batch_size*n_ways*n_query x dim
            support_features = self.model(
                support_toks,
                support_lens,
                tam_embeds=support_tam_features,
            )
            query_features = self.model(
                query_toks,
                query_lens,
                tam_embeds=query_tam_features,
            )
        else:
            # support_features: batch_size*n_ways*n_support x T x dim
            # query_features: batch_size*n_ways*n_query x T x dim
            if self.config.model.task_tam:
                # support_tam_features : ... x 2 x bert_dim
                # query_tam_features   : ... x 2 x bert_dim
                support_tam_features = torch.cat([support_rubric_embs.unsqueeze(1),
                                                  support_prompt_embs.unsqueeze(1)], dim=1)
                query_tam_features = torch.cat([query_rubric_embs.unsqueeze(1),
                                                query_prompt_embs.unsqueeze(1)], dim=1)
                support_features = self.model(
                    input_ids=support_toks,
                    attention_mask=support_masks,
                    tam_embeds=support_tam_features,
                )[0]
                query_features = self.model(
                    input_ids=query_toks,
                    attention_mask=query_masks,
                    tam_embeds=query_tam_features,
                )[0]
            elif self.config.model.task_adapter or self.config.model.task_tadam:
                # NOTE: we assume we don't use adapter/tadam/tam at the same time.
                support_task_features = torch.cat([support_rubric_embs, support_prompt_embs], dim=1)
                query_task_features = torch.cat([query_rubric_embs, query_prompt_embs], dim=1)
                support_features = self.model(
                    input_ids=support_toks,
                    attention_mask=support_masks,
                    tadam_or_adapter_embeds=support_task_features,
                )[0]
                query_features = self.model(
                    input_ids=query_toks,
                    attention_mask=query_masks,
                    tadam_or_adapter_embeds=query_task_features,
                )[0]
            else:
                support_features = self.model(input_ids=support_toks, attention_mask=support_masks)[0]
                query_features = self.model(input_ids=query_toks, attention_mask=query_masks)[0]

            # support_features: batch_size*n_ways*n_support x dim
            # query_features: batch_size*n_ways*n_query x dim
            support_features = self.compute_masked_means(support_features, support_masks)
            query_features = self.compute_masked_means(query_features, query_masks)

        if self.config.model.task_concat:
            # support_features: batch_size*n_ways*n_query x (bert_dim*2+dim)
            support_features = torch.cat([support_features, support_rubric_embs, support_prompt_embs], dim=1)
            # query_features: batch_size*n_ways*n_query x (bert_dim*2+dim)
            query_features = torch.cat([query_features, query_rubric_embs, query_prompt_embs], dim=1)
            # support_features: batch_size*n_ways*n_support x dim
            # query_features: batch_size*n_ways*n_query x dim
            support_features = self.concat_fusor(support_features)
            query_features = self.concat_fusor(query_features)

        loss, top1, logprobas  = self.compute_loss(
            support_features.view(batch_size, n_ways, n_support, -1),
            support_labs.view(batch_size, n_ways, n_support),
            query_features.view(batch_size, n_ways, n_query, -1),
            query_labs.view(batch_size, n_ways, n_query),
        )
        return loss, top1, logprobas

    @torch.no_grad()
    def zero_shot_embed_task_examples(self, task, rubric_embs, prompt_embs, device):
        # NOTE: assumes batch_size == 1
        indices = self.test_dataset.indices_by_task[task]
        toks = [self.test_dataset.token_seqs[i] for i in indices]
        lens = [self.test_dataset.token_lens[i] for i in indices]
        toks, lens = np.array(toks), np.array(lens)
        toks = torch.from_numpy(toks).long()
        lens = torch.from_numpy(lens).long()
        masks = self.test_dataset.build_attention_masks(lens)
        labs = np.array(self.test_dataset.labels_by_task[task])

        toks, masks = toks.to(device), masks.to(device)
        rubric_embs = rubric_embs[:, 0].repeat(toks.size(0), 1)
        prompt_embs = prompt_embs[:, 0].repeat(toks.size(0), 1)
        tam_embs = torch.cat([rubric_embs.unsqueeze(1), prompt_embs.unsqueeze(1)], dim=1)

        batch_size = 128
        num_total = toks.size(0)
        num_iters = (num_total // batch_size) + (num_total % batch_size != 0)

        features = []
        start_index = 0
        for i in range(num_iters):
            toks_i = toks[start_index:start_index+batch_size]
            masks_i = masks[start_index:start_index+batch_size]
            tam_embs_i = tam_embs[start_index:start_index+batch_size]
            features_i = self.model(input_ids=toks_i, attention_mask=masks_i, tam_embeds=tam_embs_i)[0]
            features_i = self.compute_masked_means(features_i, masks_i)
            features.append(features_i)
            start_index += batch_size
        features = torch.cat(features, dim=0)
        features = features.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=2).fit(features)
        preds = kmeans.labels_

        true_mode = stats.mode(labs)
        pred_mode = stats.mode(preds)
        flip = pred_mode != true_mode
        return kmeans, flip

    def zero_shot_forward(self, batch, n_shots, n_queries):
        # NOTE: assume no assume to a support set
        task = batch['task'].item()

        query_toks  = batch['query_toks'].to(self.device)
        query_lens = batch['query_lens'].to(self.device)
        query_masks = batch['query_masks'].to(self.device)
        query_labs = batch['query_labs'].to(self.device)
        rubric_embs = batch['rubric_embs'].to(self.device)
        prompt_embs = batch['prompt_embs'].to(self.device)
        bert_dim = batch['rubric_embs'].size(-1)

        device = query_toks.device
        kmeans, cluster_flip = self.zero_shot_embed_task_examples(task, rubric_embs, prompt_embs, device)

        batch_size = query_toks.size(0)
        n_ways = query_toks.size(1)
        n_query = query_toks.size(2)
        seq_len = query_toks.size(-1)

        query_toks = query_toks.view(-1, seq_len)
        query_lens = query_lens.view(-1)
        query_masks = query_masks.view(-1, seq_len).long()

        rubric_embs = rubric_embs.view(-1, bert_dim)
        query_rubric_embs = rubric_embs.unsqueeze(1).repeat(1, n_query, 1)
        query_rubric_embs = query_rubric_embs.view(-1, bert_dim)

        prompt_embs = prompt_embs.view(-1, bert_dim)
        query_prompt_embs = prompt_embs.unsqueeze(1).repeat(1, n_query, 1)
        query_prompt_embs = query_prompt_embs.view(-1, bert_dim)

        if self.config.model.name == 'lstm':
            raise NotImplementedError  # no support for this atm
        else:
            if self.config.model.task_tam:
                # support_tam_features : ... x 2 x bert_dim
                # query_tam_features   : ... x 2 x bert_dim
                query_tam_features = torch.cat([query_rubric_embs.unsqueeze(1),
                                                query_prompt_embs.unsqueeze(1)], dim=1)
                query_features = self.model(
                    input_ids=query_toks,
                    attention_mask=query_masks,
                    tam_embeds=query_tam_features,
                )[0]
            else:  # no support for other mechanisms for now
                raise NotImplementedError

        query_features = self.compute_masked_means(query_features, query_masks)

        with torch.no_grad():
            # cluster query features into labels
            device = query_features.device
            query_features_npy = query_features.detach().cpu().numpy()
            cluster_labels = kmeans.predict(query_features_npy)
            if cluster_flip:
                cluster_labels = 1 - cluster_labels
            uniq_labels = np.unique(cluster_labels)
            cluster_labels = torch.LongTensor(cluster_labels).to(device)

        if self.config.model.task_concat:
            raise NotImplementedError  # TODO: add functionality later

        prototypes = torch.stack([torch.mean(query_features[cluster_labels == c], dim=0)
                                  for c in uniq_labels])
        prototypes = prototypes.unsqueeze(0)
        query_features = query_features.view(batch_size, n_ways, n_query, -1)
        query_labs = query_labs.view(batch_size, n_ways, n_query)

        query_features_flat = query_features.view(batch_size, n_ways * n_query, -1)
        dists = self.tau * batch_euclidean_dist(query_features_flat, prototypes)
        probas = F.softmax(-dists, dim=2).view(batch_size, n_ways, n_query, -1)

        if len(uniq_labels) == 1:  # only predicting one label
            pads = torch.zeros_like(probas) + 1e-6
            probas = torch.cat([probas - 1e-6, pads], dim=-1)

        logprobas = torch.log(probas)
        loss = -logprobas.gather(3, query_labs.unsqueeze(3)).squeeze()
        loss = loss.view(-1).mean()

        acc = utils.get_accuracy(logprobas.view(batch_size, n_ways*n_query, -1),
                                 query_labs.view(batch_size, n_ways*n_query))

        return loss, acc, logprobas

    def train_one_epoch(self):
        tqdm_batch = tqdm(total=len(self.train_loader),
                          desc="[Epoch {}]".format(self.current_epoch))
        self.model.train()
        loss_meter = utils.AverageMeter()
        all_task_types = list(set(self.train_dataset.task_types))
        num_task_types = self.train_dataset.num_task_types
        acc_meters = [utils.AverageMeter() for _ in range(num_task_types)]

        for batch in self.train_loader:
            n_shots = self.config.dataset.train.n_shots
            n_queries = self.config.dataset.test.n_queries
            loss, acc, _ = self.forward(batch, n_shots, n_queries)
            task_type = batch['task_type'].cpu().numpy()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # I THINK: temperature scheduling (warmup and then cooldown)
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
            # I THINK: update tqdm with loss
            tqdm_batch.set_postfix(postfix)
            tqdm_batch.update()
        tqdm_batch.close()

        self.current_loss = loss_meter.avg
        self.train_loss.append(loss_meter.avg)
        accuracies = [acc_meters[t].avg for t in range(num_task_types)]
        print(f'Meta-Train Tasks: {accuracies}')
        self.train_acc.append(accuracies)
        self.temp.append(self.tau.item())
        print(f'Temperature: {self.tau.item()}')
        return f'Meta-Train Tasks: {accuracies}'

    def eval_split(self, name, loader):
        tqdm_batch = tqdm(total=len(loader), desc=f"[{name}]")
        self.model.eval()
        loss_meter = utils.AverageMeter()
        all_task_types = list(set(self.test_dataset.task_types))
        num_task_types = self.test_dataset.num_task_types
        acc_meters = [utils.AverageMeter() for _ in range(num_task_types)]

        with torch.no_grad():
            for batch in loader:
                n_shots = self.config.dataset.train.n_shots
                n_queries = self.config.dataset.test.n_queries
                loss, acc, _ = self.forward(batch, n_shots, n_queries)
                task_type = batch['task_type'].cpu().numpy()

                loss_meter.update(loss.item())
                postfix = {"Loss": loss_meter.avg}
                for t_, t in enumerate(all_task_types):
                    if sum(task_type == t) > 0:
                        acc_meters[t_].update(acc[task_type == t].mean())
                        postfix[f"Acc{t}"] = acc_meters[t_].avg
                tqdm_batch.update()
            tqdm_batch.close()

        accuracies = [acc_meters[t].avg for t in range(num_task_types)]
        return loss_meter.avg, accuracies

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


class CodeMatchingNetAgent(CodePrototypeNetAgent):

    def _create_model(self):
        super()._create_model()

        self.fce_f = ContextEncoder(
            self.config.model.d_model,
            num_layers=self.config.model.fce.n_encoder_layers,
        )
        self.fce_g = AttentionEncoder(
            self.config.model.d_model,
            unrolling_steps=self.config.model.fce.unrolling_steps,
        )
        self.fce_f = self.fce_f.to(self.device)
        self.fce_g = self.fce_g.to(self.device)

    def _all_parameters(self):
        all_parameters = [
            self.model.parameters(),
            self.fce_f.parameters(),
            self.fce_g.parameters(),
            [self.tau],
        ]
        if self.config.model.task_concat:
            all_parameters.append(self.concat_fusor.parameters())
        return chain(*all_parameters)

    def compute_loss(
            self,
            support_features,  # batch_size x n_ways x n_shots x d_model
            support_targets,   # batch_size x n_ways x n_shots
            query_features,    # batch_size x n_ways x n_queries x d_model
            query_targets,     # batch_size x n_ways x n_queries
        ):
        batch_size, n_ways, n_shots, d_model = support_features.size()
        n_queries = query_features.size(2)

        if self.config.model.fce.has_context:
            support_features = support_features.view(
                batch_size,
                n_ways * n_shots,
                d_model,
            )
            query_features = query_features.view(
                batch_size,
                n_ways * n_queries,
                d_model,
            )
            support_features = self.fce_f(support_features)
            query_features = self.fce_g(support_features, query_features)

        # dists : batch_size x n_ways * n_queries x n_ways * n_shots
        dists = self.tau * batch_euclidean_dist(query_features, support_features)

        # attentions : batch_size x n_ways * n_queries x n_ways * n_shots
        attentions = F.softmax(-dists, dim=2)

        # support_targets : batch_size x n_ways * n_shots
        support_targets = support_targets.view(
            batch_size,
            n_ways * n_shots,
        )
        # make into one-hotted
        support_targets_1hot = torch.zeros(batch_size, n_ways * n_shots, n_ways)
        # support_targets_1hot : batch_size x n_ways * n_shots x n_ways
        support_targets_1hot = support_targets_1hot.to(support_targets.device)
        support_targets_1hot.scatter_(2, support_targets.unsqueeze(2), 1)

        # probas : batch_size x n_ways * n_queries x n_ways
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
        if self.config.model.task_concat:
            out_dict['concat_fusor_state_dict'] = self.concat_fusor.state_dict()
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

                if self.config.model.task_concat:
                    concat_fusor_state_dict = checkpoint['concat_fusor_state_dict']
                    self.concat_fusor.load_state_dict(concat_fusor_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))

            return checkpoint

        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e


class CodeRelationNetAgent(CodePrototypeNetAgent):

    def _create_model(self):
        super()._create_model()

        self.relation = RelationNetwork(self.config.model.d_model)
        self.relation = self.relation.to(self.device)

    def _all_parameters(self):
        all_parameters = [
            self.model.parameters(),
            self.relation.parameters(),
            [self.tau],
        ]
        if self.config.model.task_concat:
            all_parameters.append(self.concat_fusor.parameters())
        return chain(*all_parameters)

    def compute_loss(
            self,
            support_features,  # batch_size x n_ways x n_shots x d_model
            support_targets,   # batch_size x n_ways x n_shots
            query_features,    # batch_size x n_ways x n_queries x d_model
            query_targets,     # batch_size x n_ways x n_queries
        ):
        batch_size, n_ways, n_query = query_targets.size()
        # query_targets : batch_size x n_ways x n_queries
        query_targets = query_targets.view(batch_size * n_ways * n_query)
        # scores : batch_size x n_ways * n_query * n_ways
        scores = self.relation(support_features, query_features)
        scores = scores.view(batch_size * n_ways * n_query, n_ways)

        # make one hot for targets
        labels = torch.zeros_like(scores)
        labels = labels.scatter_(1, query_targets.unsqueeze(1), 1)

        loss = F.mse_loss(scores, labels)

        acc = utils.get_accuracy(scores.view(batch_size, n_ways * n_query, n_ways),
                                 query_targets.view(batch_size, n_ways * n_query))
        # NOTE: scores are not logprobas but argmax works on it too
        return loss, acc, scores

    def save_metrics(self):
        out_dict = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'relation_state_dict': self.relation.state_dict(),
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
        if self.config.model.task_concat:
            out_dict['concat_fusor_state_dict'] = self.concat_fusor.state_dict()
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
                self.temp = list(checkpoint['temp'])
                self.current_val_metric = checkpoint['val_metric']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                relation_state_dict = checkpoint['relation_state_dict']
                self.model.load_state_dict(model_state_dict)
                self.relation.load_state_dict(relation_state_dict)
                self.tau.data = checkpoint['tau'].to(self.tau.device)

                if self.config.model.task_concat:
                    concat_fusor_state_dict = checkpoint['concat_fusor_state_dict']
                    self.concat_fusor.load_state_dict(concat_fusor_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))

            return checkpoint

        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e


class CodeSignaturesAgent(CodePrototypeNetAgent):

    def _create_model(self):
        super()._create_model()
        model_args = {
            'way': 2,
            'shot': self.config.dataset.train.n_shots,
        }
        model_args = DotMap(model_args)
        self.signature = DistSign(self.model, model_args)
        self.signature = self.signature.to(self.device)

    def _all_parameters(self):
        all_parameters = [
            # self.model.parameters(),
            self.signature.parameters(),
            [self.tau],
        ]
        if self.config.model.task_concat:
            all_parameters.append(self.concat_fusor.parameters())
        return chain(*all_parameters)

    def compute_loss(
            self,
            support_features,
            support_targets,
            query_features,
            query_targets,
        ):
        acc, loss, logprobas = self.signature(
            support_features, support_targets, query_features, query_targets)
        acc = np.array([acc])
        return loss, acc, logprobas


class CodeSupervisedAgent(BaseAgent):
    """
    Supervised baseline: finetune a model on a single task training
    points and make predictions on the query points. We only need to
    do this for the meta-test split.
    """

    def __init__(self, config):
        super().__init__(config)

        self.train_loss = []
        self.train_acc  = []
        self.test_acc   = []

    def _load_datasets(self):
        self.train_dataset = SupervisedExamSolutions(
            self.config.dataset.task_index,
            n_shots=self.config.dataset.train.n_shots,
            n_queries=self.config.dataset.test.n_queries,
            data_root=self.config.data_root,
            roberta_rubric=self.config.dataset.train.roberta_rubric,
            roberta_prompt=self.config.dataset.train.roberta_prompt,
            roberta_config=self.config.model.config,
            max_seq_len=self.config.dataset.max_seq_len,
            min_occ=self.config.dataset.min_occ,
            train=True,
            meta_train=True,
            hold_out_split=self.config.dataset.hold_out_split,
            hold_out_category=self.config.dataset.hold_out_category,
            enforce_binary=self.config.dataset.enforce_binary,
            pad_to_max_num_class=self.config.optim.batch_size > 1,
        )
        self.test_dataset = SupervisedExamSolutions(
            self.config.dataset.task_index,
            n_shots=self.config.dataset.train.n_shots,
            n_queries=self.config.dataset.test.n_queries,
            data_root=self.config.data_root,
            roberta_rubric=self.config.dataset.train.roberta_rubric,
            roberta_prompt=self.config.dataset.train.roberta_prompt,
            roberta_config=self.config.model.config,
            max_seq_len=self.config.dataset.max_seq_len,
            min_occ=self.config.dataset.min_occ,
            train=False,
            meta_train=True,  # always True
            hold_out_split=self.config.dataset.hold_out_split,
            hold_out_category=self.config.dataset.hold_out_category,
            enforce_binary=self.config.dataset.enforce_binary,
            pad_to_max_num_class=self.config.optim.batch_size > 1,
        )

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
        if self.config.model.name == 'transformer':
            vocab_size = self.train_dataset.vocab_size
            model = CodeTransformerEncoder(
                vocab_size,
                d_model=self.config.model.d_model,
                n_head=self.config.model.n_head,
                n_encoder_layers=self.config.model.n_encoder_layers,
                d_ff=self.config.model.d_ff,
                dropout=0.1,
                activation="relu",
                norm=True,
                pad_id=self.train_dataset.pad_index,
                is_tam=self.config.model.task_tam,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )
        elif self.config.model.name == 'roberta':
            model = RobertaModel.from_pretrained(
                self.config.model.config,
                is_tam=self.config.model.task_tam,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )
            utils.reset_model_for_training(model)

            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.pooler.parameters():
                    param.requires_grad = True
                # only allow some parameters to be finetuned
                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

        elif self.config.model.name == 'roberta_codesearch':
            model = RobertaForMaskedLM.from_pretrained(
                'roberta-base',
                is_tam=self.config.model.task_tam,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )
            # load the codesearch checkpoint
            checkpoint = torch.load(
                self.config.model.codesearch_checkpoint_path,
                map_location='cpu',
            )
            raw_state_dict = checkpoint['state_dict']
            state_dict = OrderedDict()
            for k, v in raw_state_dict.items():
                new_k = '.'.join(k.split('.')[1:])
                state_dict[new_k] = v
            model.load_state_dict(state_dict)
            model = model.roberta  # only keep roberta
            utils.reset_model_for_training(model)
            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False
                # only allow some parameters to be finetuned
                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

        elif self.config.model.name == 'roberta_scratch':
            config = RobertaConfig.from_pretrained(self.config.model.config)
            model = RobertaModel(
                config,
                is_tam=self.config.model.task_tam,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )
            utils.reset_model_for_training(model)

            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False
                # only allow some parameters to be finetuned
                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

        else:
            raise Exception(f'Model {self.config.model.name} not supported.')

        self.model = model.to(self.device)
        d_model = self.config.model.d_model
        bert_dim = 768

        if self.config.model.task_concat:
            # combine program embedding and rubric/question at the end of the forward pass
            concat_fusor = TaskEmbedding(d_model+bert_dim*2, d_model, hid_dim=d_model)
            self.concat_fusor = concat_fusor.to(self.device)

        classifier = nn.Linear(self.config.model.d_model, 1)
        self.classifier = classifier.to(self.device)

    def _all_parameters(self):
        all_parameters = [self.model.parameters()]
        if self.config.model.task_concat:
            all_parameters.append(self.concat_fusor.parameters())
        return chain(*all_parameters)

    def _create_optimizer(self):
        optimizer = torch.optim.AdamW(
            self._all_parameters(),
            lr=self.config.optim.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.01,
        )
        self.optim = optimizer
        self.config.optim.use_scheduler = False

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
        tokens = batch['tokens'].to(self.device)
        masks = batch['masks'].to(self.device)
        labels = batch['labels'].to(self.device)
        rubric_embs = batch['rubric_embs'].to(self.device)
        prompt_embs = batch['prompt_embs'].to(self.device)

        batch_size = tokens.size(0)
        if self.config.model.task_tam:
            # tam_features : ... x 2 x bert_dim
            tam_features = torch.cat([rubric_embs.unsqueeze(1), prompt_embs.unsqueeze(1)], dim=1)
            features = self.model(input_ids=tokens, attention_mask=masks, tam_embeds=tam_features)[0]
        elif self.config.model.task_adapter or self.config.model.task_tadam:
            # NOTE: we assume we don't use adapter/tadam/tam at the same time.
            task_features = torch.cat([rubric_embs, prompt_embs], dim=1)
            features = self.model(input_ids=tokens, attention_mask=masks,
                                  tadam_or_adapter_embeds=task_features)[0]
        else:
            features = self.model(input_ids=tokens, attention_mask=masks)[0]

        features = self.compute_masked_means(features, masks)

        if self.config.model.task_concat:
            # features: batch_size x (dim+bert_dim*2)
            features = torch.cat([features, rubric_embs, prompt_embs], dim=1)
            # features: batch_size x dim
            features = self.concat_fusor(features)

        logits = self.classifier(F.relu(features))
        probas = torch.sigmoid(logits)
        labels = labels.unsqueeze(1).float()
        loss = F.binary_cross_entropy(probas, labels)

        with torch.no_grad():
            preds = torch.round(probas)
            correct = preds.eq(labels).sum().item()
            acc = 100. / batch_size * correct

        return loss, acc, probas

    def train_one_epoch(self):
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

            if self.config.optim.use_scheduler:
                self.scheduler.step()

            with torch.no_grad():
                loss_meter.update(loss.item())
                acc_meter.update(acc)
                postfix = {
                    "Loss": loss_meter.avg,
                    "Acc": acc_meter.avg,
                }
                self.current_iteration += 1
                tqdm_batch.set_postfix(postfix)
            tqdm_batch.update()
        tqdm_batch.close()

        self.current_loss = loss_meter.avg
        self.train_loss.append(loss_meter.avg)
        print(f'Meta-Train Tasks: {acc_meter.avg}')
        self.train_acc.append(acc_meter.avg)

    def eval_split(self, name, loader):
        tqdm_batch = tqdm(total=len(loader), desc=f"[{name}]")
        self.model.eval()
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()

        with torch.no_grad():
            for batch in loader:
                loss, acc, _ = self.forward(batch)
                loss_meter.update(loss.item())
                acc_meter.update(acc)
                postfix = {
                    "Loss": loss_meter.avg,
                    "Acc": acc_meter.avg,
                }
                tqdm_batch.set_postfix(postfix)
                tqdm_batch.update()
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
        if self.config.model.task_concat:
            out_dict['concat_fusor_state_dict'] = self.concat_fusor.state_dict()
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
                # self.temp = list(checkpoint['temp'])
                self.current_val_metric = checkpoint['val_metric']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)

                if self.config.model.task_concat:
                    concat_fusor_state_dict = checkpoint['concat_fusor_state_dict']
                    self.concat_fusor.load_state_dict(concat_fusor_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))

            return checkpoint

        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e

class TextPrototypeNetAgent(BaseAgent):

    # START: Stuff from BaseCodeMetaAgent

    def __init__(self, config):
        super().__init__(config)

        self.train_loss = []
        self.train_acc  = []
        self.test_acc   = []
        self.temp       = []

    def _load_datasets(self):
        if not self.config.cuda:
            roberta_device = 'cpu'
        else:
            roberta_device = f'cuda:{self.config.gpu_device}'

        self.train_dataset = MetaDTSolutions(
            data_root=self.config.dataset.data_root,
            n_shots=self.config.dataset.train.n_shots,
            n_queries=self.config.dataset.test.n_queries,
            train=True,
            vocab=None,
            train_frac=self.config.dataset.train_frac,
            max_seq_len=self.config.dataset.max_seq_len,
            min_occ=self.config.dataset.min_occ,
            augment_by_rubric=self.config.dataset.train.augment_by_rubric,
            roberta_rubric=self.config.dataset.train.roberta_rubric,
            roberta_prompt=self.config.dataset.train.roberta_prompt,
            roberta_tokenize=self.config.dataset.roberta_tokenize,
            roberta_config=self.config.model.config,
            roberta_device=roberta_device,
            conservative=self.config.dataset.train.conservative,
            pad_to_max_num_class=self.config.optim.batch_size > 1,
            hold_out_split=self.config.dataset.hold_out_split,
            enforce_binary=self.config.dataset.enforce_binary,
            rubric_path=self.config.dataset.rubric_path,
            answers_path=self.config.dataset.answers_path,
            cache_path=self.config.dataset.cache_path,
            simple_binary=self.config.train.simple_binary,
            keep_all_in_train=self.config.dataset.train.keep_all_in_train,
        )
        self.test_dataset = MetaDTSolutions(
            data_root=self.config.dataset.data_root,
            n_shots=self.config.dataset.train.n_shots,
            n_queries=self.config.dataset.test.n_queries,
            train=False,
            vocab=self.train_dataset.vocab,
            train_frac=self.config.dataset.train_frac,
            max_seq_len=self.config.dataset.max_seq_len,
            min_occ=self.config.dataset.min_occ,
            augment_by_rubric=self.config.dataset.train.augment_by_rubric,
            roberta_rubric=self.config.dataset.train.roberta_rubric,
            roberta_prompt=self.config.dataset.train.roberta_prompt,
            roberta_tokenize=self.config.dataset.roberta_tokenize,
            roberta_config=self.config.model.config,
            roberta_device=roberta_device,
            conservative=self.config.dataset.train.conservative,
            pad_to_max_num_class=self.config.optim.batch_size > 1,
            hold_out_split=self.config.dataset.hold_out_split,
            enforce_binary=self.config.dataset.enforce_binary,
            rubric_path=self.config.dataset.rubric_path,
            answers_path=self.config.dataset.answers_path,
            cache_path=self.config.dataset.cache_path,
            simple_binary=self.config.train.simple_binary,
            larger_sample=self.config.dataset.larger_sample,
        )

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
        if self.config.model.name == 'transformer':
            vocab_size = self.train_dataset.vocab_size
            model = CodeTransformerEncoder(
                vocab_size,
                d_model=self.config.model.d_model,
                n_head=self.config.model.n_head,
                n_encoder_layers=self.config.model.n_encoder_layers,
                d_ff=self.config.model.d_ff,
                dropout=0.1,
                activation="relu",
                norm=True,
                pad_id=self.train_dataset.pad_index,
                is_tam=self.config.model.task_tam,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )

        elif self.config.model.name == 'roberta':
            model = RobertaModel.from_pretrained(
                self.config.model.config,
                is_tam=self.config.model.task_tam,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )
            # set everything to requires_grad = True
            utils.reset_model_for_training(model)

            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.pooler.parameters():
                    param.requires_grad = True
                # only allow some parameters to be finetuned
                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

        elif self.config.model.name == 'roberta_codesearch':
            model = RobertaForMaskedLM.from_pretrained(
                'roberta-base',
                is_tam=self.config.model.task_tam,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )
            # load the codesearch checkpoint
            checkpoint = torch.load(
                self.config.model.codesearch_checkpoint_path,
                map_location='cpu',
            )
            raw_state_dict = checkpoint['state_dict']
            state_dict = OrderedDict()
            for k, v in raw_state_dict.items():
                new_k = '.'.join(k.split('.')[1:])
                state_dict[new_k] = v
            model.load_state_dict(state_dict, strict=False)
            model = model.roberta  # only keep roberta
            utils.reset_model_for_training(model)
            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False
                # only allow some parameters to be finetuned
                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

        elif self.config.model.name == 'roberta_scratch':
            config = RobertaConfig.from_pretrained(self.config.model.config)
            model = RobertaModel(
                config,
                is_tadam=self.config.model.task_tadam,
                is_adapter=self.config.model.task_adapter,
            )
            # set everything to requires_grad = True
            utils.reset_model_for_training(model)

            if self.config.model.finetune:
                for param in model.parameters():
                    param.requires_grad = False
                # only allow some parameters to be finetuned
                for param in model.encoder.layer[-self.config.model.finetune_layers:].parameters():
                    param.requires_grad = True

        elif self.config.model.name == 'lstm':
            assert not self.config.model.task_tadam, "TADAM not support for LSTMs."
            assert not self.config.model.task_adapter, "Adapter not support for LSTMs."

            vocab_size = len(self.train_dataset.vocab['w2i'])
            model = CodeLSTMEncoder(
                vocab_size,
                d_model=self.config.model.d_model,
                n_encoder_layers=self.config.model.n_encoder_layers,
                dropout=0.1,
                is_tadam=self.config.model.task_tadam,
            )
        else:
            raise Exception(f'Model {self.config.model.name} not supported.')

        self.model = model.to(self.device)
        d_model = self.config.model.d_model
        bert_dim = 768

        if self.config.model.task_concat:
            # combine program embedding and rubric/question at the end of the forward pass
            concat_fusor = TaskEmbedding(d_model+bert_dim*2, d_model, hid_dim=d_model)
            self.concat_fusor = concat_fusor.to(self.device)

        tau = nn.Parameter(torch.ones(1)).to(self.device)
        tau = tau.detach().requires_grad_(True)
        self.tau = tau

    def _all_parameters(self):
        all_parameters = [self.model.parameters(), [self.tau]]
        if self.config.model.task_concat:
            all_parameters.append(self.concat_fusor.parameters())
        return chain(*all_parameters)

    def _create_optimizer(self):
        if self.config.model.name in ['roberta', 'roberta_mlm', 'roberta_scratch']:
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
        else:
            # this is the one used for Adam
            self.optim = torch.optim.AdamW(
                self._all_parameters(),
                lr=self.config.optim.learning_rate,
                betas=(0.9, 0.98),
                weight_decay=self.config.optim.weight_decay,
            )

            if self.config.optim.use_scheduler:

                def schedule(step_num):
                    d_model = self.config.model.d_model
                    warmup_steps = self.config.optim.warmup_steps
                    step_num += 1
                    lrate = d_model**(-0.5) * min(step_num**(-0.5), step_num * warmup_steps**(-1.5))
                    return lrate

                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, schedule)

    def train(self):
        for epoch in range(self.current_epoch, self.config.optim.num_epochs):
            print(f"Epoch: {epoch}") # to write to tee
            self.current_epoch = epoch
            self.write_to_file(epoch)
            self.write_to_file(self.train_one_epoch())

            if (self.config.validate and epoch % self.config.validate_freq == 0):
                self.write_to_file(self.eval_test())

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
        if self.config.model.task_concat:
            out_dict['concat_fusor_state_dict'] = self.concat_fusor.state_dict()
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
                self.temp = list(checkpoint['temp'])
                self.current_val_metric = checkpoint['val_metric']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)
                self.tau.data = checkpoint['tau'].to(self.tau.device)

                if self.config.model.task_concat:
                    concat_fusor_state_dict = checkpoint['concat_fusor_state_dict']
                    self.concat_fusor.load_state_dict(concat_fusor_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))

            return checkpoint

        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e

    # End: stuff from BaseCodeMetaAgent

    def compute_loss(
            self,
            support_features,
            support_targets,
            query_features,
            query_targets,
        ):
        batch_size, nway, nquery, dim = query_features.size()
        prototypes = torch.mean(support_features, dim=2)
        query_features_flat = query_features.view(batch_size, nway * nquery, dim)

        # batch-based euclidean dist between prototypes and query_features_flat
        # dists: batch_size x nway * nquery x nway
        dists = self.tau * batch_euclidean_dist(query_features_flat, prototypes)
        logprobas = F.log_softmax(-dists, dim=2).view(batch_size, nway, nquery, -1)

        loss = -logprobas.gather(3, query_targets.unsqueeze(3)).squeeze()
        loss = loss.view(-1).mean()

        acc = utils.get_accuracy(logprobas.view(batch_size, nway*nquery, -1),
                                 query_targets.view(batch_size, nway*nquery))
        return loss, acc, logprobas

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

    def forward(self, batch, n_shots, n_queries):
        # NOTE: n_shots, n_queries are unused

        support_toks = batch['support_toks'].to(self.device)
        support_lens = batch['support_lens'].to(self.device)
        support_masks = batch['support_masks'].to(self.device)
        support_labs = batch['support_labs'].to(self.device)
        query_toks  = batch['query_toks'].to(self.device)
        query_lens = batch['query_lens'].to(self.device)
        query_masks = batch['query_masks'].to(self.device)
        query_labs = batch['query_labs'].to(self.device)
        rubric_embs = batch['rubric_embs'].to(self.device)
        prompt_embs = batch['prompt_embs'].to(self.device)
        bert_dim = batch['rubric_embs'].size(-1)

        if self.config.override_n_shots != DotMap():  # NOTE: used in test time to vary supervision
            assert self.config.override_n_shots <= support_toks.size(2)

            if self.config.override_n_shots == 0:
                # separate procedure for zero-shot
                return self.zero_shot_forward(batch, n_shots, n_queries)

            # if > 0, we can just pretend like we have less
            support_toks = support_toks[:, :, :self.config.override_n_shots, :].contiguous()
            support_lens = support_lens[:, :, :self.config.override_n_shots].contiguous()
            support_masks = support_masks[:, :, :self.config.override_n_shots, :].contiguous()
            support_labs = support_labs[:, :, :self.config.override_n_shots].contiguous()

        batch_size = support_toks.size(0)
        n_ways = support_toks.size(1)
        n_support = support_toks.size(2)
        n_query = query_toks.size(2)
        seq_len = support_toks.size(-1)

        # support_toks: batch_size*n_ways*n_support x seq_len
        support_toks = support_toks.view(-1, seq_len)
        support_lens = support_lens.view(-1)
        support_masks = support_masks.view(-1, seq_len).long()
        query_toks = query_toks.view(-1, seq_len)
        query_lens = query_lens.view(-1)
        query_masks = query_masks.view(-1, seq_len).long()

        # rubric_embs: batch_size*n_ways x bert_dim
        rubric_embs = rubric_embs.view(-1, bert_dim)
        support_rubric_embs = rubric_embs.unsqueeze(1).repeat(1, n_support, 1)
        # support_rubric_embs: batch_size*n_ways*n_support x bert_dim
        support_rubric_embs = support_rubric_embs.view(-1, bert_dim)
        # query_rubric_embs: batch_size*n_ways*n_query x bert_dim
        query_rubric_embs = rubric_embs.unsqueeze(1).repeat(1, n_query, 1)
        query_rubric_embs = query_rubric_embs.view(-1, bert_dim)

        # prompt_embs: batch_size*n_ways x bert_dim
        prompt_embs = prompt_embs.view(-1, bert_dim)
        support_prompt_embs = prompt_embs.unsqueeze(1).repeat(1, n_support, 1)
        # support_rubric_embs: batch_size*n_ways*n_support x bert_dim
        support_prompt_embs = support_prompt_embs.view(-1, bert_dim)
        query_prompt_embs = prompt_embs.unsqueeze(1).repeat(1, n_query, 1)
        # query_rubric_embs: batch_size*n_ways*n_prompt x bert_dim
        query_prompt_embs = query_prompt_embs.view(-1, bert_dim)

        if self.config.model.name == 'lstm':
            # support_tam_features : ... x 2 x bert_dim
            # query_tam_features   : ... x 2 x bert_dim
            support_tam_features = torch.cat([support_rubric_embs.unsqueeze(1),
                                              support_prompt_embs.unsqueeze(1)], dim=1)
            query_tam_features = torch.cat([query_rubric_embs.unsqueeze(1),
                                            query_prompt_embs.unsqueeze(1)], dim=1)
            # support_features: batch_size*n_ways*n_support x dim
            # query_features: batch_size*n_ways*n_query x dim
            support_features = self.model(
                support_toks,
                support_lens,
                tam_embeds=support_tam_features,
            )
            query_features = self.model(
                query_toks,
                query_lens,
                tam_embeds=query_tam_features,
            )
        else:
            # support_features: batch_size*n_ways*n_support x T x dim
            # query_features: batch_size*n_ways*n_query x T x dim
            if self.config.model.task_tam:
                # support_tam_features : ... x 2 x bert_dim
                # query_tam_features   : ... x 2 x bert_dim
                support_tam_features = torch.cat([support_rubric_embs.unsqueeze(1),
                                                  support_prompt_embs.unsqueeze(1)], dim=1)
                query_tam_features = torch.cat([query_rubric_embs.unsqueeze(1),
                                                query_prompt_embs.unsqueeze(1)], dim=1)
                support_features = self.model(
                    input_ids=support_toks,
                    attention_mask=support_masks,
                    tam_embeds=support_tam_features,
                )[0]
                query_features = self.model(
                    input_ids=query_toks,
                    attention_mask=query_masks,
                    tam_embeds=query_tam_features,
                )[0]
            elif self.config.model.task_adapter or self.config.model.task_tadam:
                # NOTE: we assume we don't use adapter/tadam/tam at the same time.
                support_task_features = torch.cat([support_rubric_embs, support_prompt_embs], dim=1)
                query_task_features = torch.cat([query_rubric_embs, query_prompt_embs], dim=1)
                support_features = self.model(
                    input_ids=support_toks,
                    attention_mask=support_masks,
                    tadam_or_adapter_embeds=support_task_features,
                )[0]
                query_features = self.model(
                    input_ids=query_toks,
                    attention_mask=query_masks,
                    tadam_or_adapter_embeds=query_task_features,
                )[0]
            else:
                support_features = self.model(input_ids=support_toks, attention_mask=support_masks)[0]
                query_features = self.model(input_ids=query_toks, attention_mask=query_masks)[0]

            # support_features: batch_size*n_ways*n_support x dim
            # query_features: batch_size*n_ways*n_query x dim
            support_features = self.compute_masked_means(support_features, support_masks)
            query_features = self.compute_masked_means(query_features, query_masks)

        if self.config.model.task_concat:
            # support_features: batch_size*n_ways*n_query x (bert_dim*2+dim)
            support_features = torch.cat([support_features, support_rubric_embs, support_prompt_embs], dim=1)
            # query_features: batch_size*n_ways*n_query x (bert_dim*2+dim)
            query_features = torch.cat([query_features, query_rubric_embs, query_prompt_embs], dim=1)
            # support_features: batch_size*n_ways*n_support x dim
            # query_features: batch_size*n_ways*n_query x dim
            support_features = self.concat_fusor(support_features)
            query_features = self.concat_fusor(query_features)

        loss, top1, logprobas  = self.compute_loss(
            support_features.view(batch_size, n_ways, n_support, -1),
            support_labs.view(batch_size, n_ways, n_support),
            query_features.view(batch_size, n_ways, n_query, -1),
            query_labs.view(batch_size, n_ways, n_query),
        )
        return loss, top1, logprobas

    def train_one_epoch(self):
        tqdm_batch = tqdm(total=len(self.train_loader),
                          desc="[Epoch {}]".format(self.current_epoch))
        self.model.train()
        loss_meter = utils.AverageMeter()
        all_task_types = list(set(self.train_dataset.task_types))
        num_task_types = self.train_dataset.num_task_types
        acc_meters = [utils.AverageMeter() for _ in range(num_task_types)]

        for batch in self.train_loader:
            n_shots = self.config.dataset.train.n_shots
            n_queries = self.config.dataset.test.n_queries
            loss, acc, _ = self.forward(batch, n_shots, n_queries)
            task_type = batch['task_type'].cpu().numpy()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # I THINK: temperature scheduling (warmup and then cooldown)
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
            # I THINK: update tqdm with loss
            tqdm_batch.set_postfix(postfix)
            tqdm_batch.update()
        tqdm_batch.close()

        self.current_loss = loss_meter.avg
        self.train_loss.append(loss_meter.avg)
        accuracies = [acc_meters[t].avg for t in range(num_task_types)]
        print(f'Meta-Train Tasks: {accuracies}')
        self.train_acc.append(accuracies)
        self.temp.append(self.tau.item())
        print(f'Temperature: {self.tau.item()}')
        return f'Meta-Train Tasks: {accuracies}'

    def eval_split(self, name, loader):
        tqdm_batch = tqdm(total=len(loader), desc=f"[{name}]")
        print("LENGTH OF THE LOADER IS:", len(loader))
        self.model.eval()
        loss_meter = utils.AverageMeter()
        all_task_types = list(set(self.test_dataset.task_types))
        num_task_types = self.test_dataset.num_task_types
        acc_meters = [utils.AverageMeter() for _ in range(num_task_types)]

        with torch.no_grad():
            for batch in loader:

                n_shots = self.config.dataset.train.n_shots
                n_queries = self.config.dataset.test.n_queries
                loss, acc, _ = self.forward(batch, n_shots, n_queries)
                task_type = batch['task_type'].cpu().numpy()

                loss_meter.update(loss.item())
                postfix = {"Loss": loss_meter.avg}
                for t_, t in enumerate(all_task_types):
                    if sum(task_type == t) > 0:
                        acc_meters[t_].update(acc[task_type == t].mean())
                        postfix[f"Acc{t}"] = acc_meters[t_].avg
                tqdm_batch.update()
            tqdm_batch.close()

        accuracies = [acc_meters[t].avg for t in range(num_task_types)]
        return loss_meter.avg, accuracies

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
        return f'Meta-Val Tasks: {acc}'
