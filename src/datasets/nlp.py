import os
import copy
import torch
import itertools
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data.dataset import Dataset
from sklearn.datasets import fetch_20newsgroups
from transformers import RobertaTokenizer
from src.models.sentencebert import SentenceBERT


class BaseMetaNLPDataset(Dataset):

    def __init__(
            self,
            data_root,
            n_ways,
            n_shots,
            n_queries,
            roberta_device='cpu',
            smlmt_tasks_factor=0,
            train=True,
            fix_seed=1337,
        ):
        super(BaseMetaNLPDataset, self).__init__()

        print("RUNNING INIT FOR BASEMETANLPDATASET")
        self.data_root = data_root
        self.fix_seed = fix_seed
        self.rs = np.random.RandomState(fix_seed)
        self.load_data(train)

        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.max_seq_len = 512
        self.n_tasks = len(list(itertools.combinations(range(self.num_cats), n_ways)))
        self.n_smlmt = int(smlmt_tasks_factor * self.n_tasks)
        self.roberta_device = roberta_device
        self.train = train

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_index = self.tokenizer.pad_token_id
        self.mask_index = self.tokenizer.mask_token_id

        if not os.path.isdir(self.cache_dir): os.makedirs(self.cache_dir)

        self.names_dict = self.embed_names(cache_dir=self.cache_dir)
        print(self.names_dict)
        token_seqs, token_masks, token_labs, token_names = self.process_data(cache_dir=self.cache_dir)
        self.tasks, self.task_types = self.build_meta_tasks(token_seqs, token_masks, token_labs, token_names,
                                                            n_smlmt=self.n_smlmt)

    def process_data(self, cache_dir):
        split = 'train' if self.train else 'test'
        cache_file = os.path.join(cache_dir, f'tokenized_{split}.pth.tar')

        if os.path.isfile(cache_file):
            tokenized = torch.load(cache_file)
            token_seqs = tokenized['inputs']
            token_masks = tokenized['masks']
            token_labs = tokenized['labels']
            token_names = tokenized['names']
        else:
            token_seqs, token_masks, token_labs = [], [], []
            token_names = []  # store strings

            for i in tqdm(range(len(self.raw_texts))):
                tokenizer_outputs = self.tokenizer(
                    self.raw_texts[i],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_seq_len,
                    pad_to_max_length=True,
                    return_tensors='pt',
                )
                tokens = tokenizer_outputs['input_ids']
                masks = tokenizer_outputs['attention_mask']
                label_i = self.raw_labels[i]

                token_seqs.append(tokens)
                token_masks.append(masks)
                token_labs.append(label_i)
                token_names.append(self.raw_names[label_i])

            token_seqs = torch.cat(token_seqs, dim=0)       # N x 512
            token_masks = torch.cat(token_masks, dim=0)     # N x 512
            token_labs = torch.LongTensor(token_labs)       # N
            token_names = np.array(token_names)             # N

            torch.save({'inputs': token_seqs, 'masks': token_masks,
                        'labels': token_labs, 'names': token_names}, cache_file)

        return token_seqs, token_masks, token_labs, token_names

    def embed_names(self, cache_dir):
        split = 'train' if self.train else 'test'
        cache_file = os.path.join(cache_dir, f'sideinfo_{split}.pth.tar')

        if os.path.exists(cache_file):
            cache = torch.load(cache_file)
        else:
            print('embedding side information...')
            cache = {'None': torch.zeros(768)}
            names = copy.deepcopy(self.raw_names)
            bert = SentenceBERT(
                version='bert-base-nli-stsb-mean-tokens',
                device=self.roberta_device,
            )
            embs = bert(names, batch_size=32, show_progress_bar=True)
            embs = embs.detach().cpu()

            for i in range(len(names)):
                cache[names[i]] = embs[i]

            print('saving to cache...')
            torch.save(cache, cache_file)

        return cache

    def build_meta_tasks(self, inputs, masks, labels, names, n_smlmt):
        unique_labels = list(set(labels.numpy().tolist()))
        smlmt_choices = self.prep_smlmt_task(inputs)

        # figure out the combo of unique tasks to build
        task_labels = list(itertools.combinations(unique_labels, self.n_ways))
        task_labels = np.array(task_labels)
        n_tasks = len(task_labels)

        tasks = []
        task_types = []
        print(f'building {n_tasks} meta tasks...')
        for i in tqdm(range(n_tasks)):
            task_cats = task_labels[i]
            task = self.build_meta_task(inputs, masks, labels, names, task_cats)
            tasks.append(task)
            task_types.append(0)

        print(f'building {n_smlmt} smlmt tasks...')
        for _ in tqdm(range(n_smlmt)):
            smlmt_cats = self.rs.choice(smlmt_choices, self.n_ways)
            task = self.build_smlmt_task(inputs, masks, smlmt_cats)
            tasks.append(task)
            task_types.append(1)

        return tasks, task_types

    def build_meta_task(self, inputs, masks, labels, names, task_cats):
        support_inputs, support_masks, support_labels = [], [], []
        query_inputs, query_masks, query_labels = [], [], []
        side_info = []

        for c, cat in enumerate(task_cats):
            inputs_cat = inputs[labels == cat]
            masks_cat  = masks[labels == cat]
            n_ex_cat = inputs_cat.size(0)

            names_cat = names[np.where(labels.numpy() == cat)[0]]
            assert len(set(names_cat)) == 1
            name_cat = names_cat[0]

            name_emb = self.names_dict[name_cat]
            side_info.append(name_emb)

            support_indices = self.rs.choice(np.arange(n_ex_cat), self.n_shots, replace=False)
            query_indices = np.setxor1d(np.arange(n_ex_cat), support_indices)
            query_indices = self.rs.choice(query_indices, self.n_queries, replace=False)

            support_indices = torch.LongTensor(support_indices)
            query_indices = torch.LongTensor(query_indices)

            support_inputs_cat = inputs_cat[support_indices]
            support_masks_cat = masks_cat[support_indices]
            support_labels_cat = torch.ones(self.n_shots) * c

            query_inputs_cat = inputs_cat[query_indices]
            query_masks_cat = masks_cat[query_indices]
            query_labels_cat = torch.ones(self.n_queries) * c

            support_inputs.append(support_inputs_cat)
            support_masks.append(support_masks_cat)
            support_labels.append(support_labels_cat)

            query_inputs.append(query_inputs_cat)
            query_masks.append(query_masks_cat)
            query_labels.append(query_labels_cat)

        support_inputs = torch.stack(support_inputs)  # n_ways x n_shots x 512
        support_masks = torch.stack(support_masks)    # n_ways x n_shots x 512
        support_labels = torch.stack(support_labels)  # n_ways x n_shots

        query_inputs = torch.stack(query_inputs)  # n_ways x n_queries x 512
        query_masks = torch.stack(query_masks)    # n_ways x n_queries x 512
        query_labels = torch.stack(query_labels)  # n_ways x n_queries

        support_lens = torch.sum(support_masks, dim=2).long()   # n_ways x n_shots
        query_lens = torch.sum(query_masks, dim=2).long()       # n_ways x n_queries

        side_info = torch.stack(side_info)  # n_ways x 768
        side_info = torch.mean(side_info, dim=0)

        task_dict = dict(
            support_toks=support_inputs,
            support_masks=support_masks,
            support_lens=support_lens,
            support_labs=support_labels.long(),
            query_toks=query_inputs,
            query_masks=query_masks,
            query_lens=query_lens,
            query_labs=query_labels.long(),
            side_info=side_info,
        )

        return task_dict

    def prep_smlmt_task(self, inputs):
        inputs = inputs.numpy()
        inputs_unique = []

        for i in range(len(inputs)):
            inputs_i = np.unique(inputs[i])
            inputs_unique.append(inputs_i)

        inputs_unique = np.concatenate(inputs_unique)
        freqs = Counter(inputs_unique)

        valid_chars = []
        for ch, fr in freqs.items():
            if fr >= (self.n_shots + self.n_queries):
                valid_chars.append(ch)

        return np.array(valid_chars)

    def build_smlmt_task(self, inputs, masks, smlmt_cats):
        support_inputs, support_masks, support_labels = [], [], []
        query_inputs, query_masks, query_labels = [], [], []

        for c, cat in enumerate(smlmt_cats):
            # check which inputs have this
            has_char = (torch.sum(inputs == cat, dim=1) > 0)
            inputs_cat = inputs[has_char].clone()
            masks_cat = masks[has_char]
            n_ex_cat = inputs_cat.size(0)

            # replace special cat with mask
            inputs_cat[inputs_cat == cat] = self.mask_index

            support_indices = self.rs.choice(np.arange(n_ex_cat), self.n_shots, replace=False)
            query_indices = np.setxor1d(np.arange(n_ex_cat), support_indices)
            query_indices = self.rs.choice(query_indices, self.n_queries, replace=False)

            support_indices = torch.LongTensor(support_indices)
            query_indices = torch.LongTensor(query_indices)

            support_inputs_cat = inputs_cat[support_indices]
            support_masks_cat = masks_cat[support_indices]
            support_labels_cat = torch.ones(self.n_shots) * c

            query_inputs_cat = inputs_cat[query_indices]
            query_masks_cat = masks_cat[query_indices]
            query_labels_cat = torch.ones(self.n_queries) * c

            support_inputs.append(support_inputs_cat)
            support_masks.append(support_masks_cat)
            support_labels.append(support_labels_cat)

            query_inputs.append(query_inputs_cat)
            query_masks.append(query_masks_cat)
            query_labels.append(query_labels_cat)

        support_inputs = torch.stack(support_inputs)  # n_ways x n_shots x 512
        support_masks = torch.stack(support_masks)    # n_ways x n_shots x 512
        support_labels = torch.stack(support_labels)  # n_ways x n_shots

        query_inputs = torch.stack(query_inputs)  # n_ways x n_queries x 512
        query_masks = torch.stack(query_masks)    # n_ways x n_queries x 512
        query_labels = torch.stack(query_labels)  # n_ways x n_queries

        support_lens = torch.sum(support_masks, dim=2).long()  # n_ways x n_shots
        query_lens = torch.sum(query_masks, dim=2).long()      # n_ways x n_queries

        task_dict = dict(
            support_toks=support_inputs,
            support_masks=support_masks,
            support_labs=support_labels.long(),
            support_lens=support_lens,
            query_toks=query_inputs,
            query_masks=query_masks,
            query_labs=query_labels.long(),
            query_lens=query_lens,
            side_info=self.names_dict['None'],
        )

        return task_dict

    def __getitem__(self, index):
        task = self.tasks[index]
        task_type = self.task_types[index]
        task['task_type'] = task_type
        return task

    def __len__(self):
        return self.n_tasks + self.n_smlmt


class BaseSupNLPDataset(BaseMetaNLPDataset):

    def __init__(
            self,
            task_index,
            data_root,
            n_ways,
            n_shots,
            n_queries,
            roberta_device='cpu',
            train=True,
            fix_seed=1337,
        ):
        super().__init__(data_root, n_ways, n_shots, n_queries,
                         roberta_device=roberta_device,
                         train=train, fix_seed=fix_seed)
        self.task_index = task_index

    def __getitem__(self, index):
        task = self.tasks[self.task_index]

        if self.train:
            data_dict = dict(
                tokens=task['support_toks'],
                lengths=task['support_lens'],
                labels=task['support_labs'],
                masks=task['support_masks'],
                side_info=task['side_info'],
            )
        else:
            data_dict = dict(
                tokens=task['query_toks'],
                lengths=task['query_lens'],
                labels=task['query_labs'],
                masks=task['query_masks'],
                side_info=task['side_info'],
            )
        return data_dict

    def __len__(self):
        task = self.tasks[self.task_index]
        if self.train:
            return len(task['support_toks'])
        else:
            return len(task['query_toks'])


class MetaNewsGroup(BaseMetaNLPDataset):

    def load_data(self, train):
        newsgroup = fetch_20newsgroups(subset='all', data_home=os.path.join(self.data_root, 'newsgroup'))

        # split this into classes
        data = np.array(newsgroup.data)
        target = np.array(newsgroup.target)
        target_names = newsgroup.target_names

        test_names = self.rs.choice(target_names, 5, replace=False)
        train_names = np.array(list(set(target_names) - set(test_names)))

        if train:
            cur_target_names = train_names
        else:
            cur_target_names = test_names

        target_indices = [target_names.index(name) for name in cur_target_names]
        target_indices = np.array(target_indices)
        target_indices = np.in1d(target, target_indices)

        self.raw_texts = data[target_indices].tolist()
        self.raw_labels = target[target_indices].tolist()

        # remap labels to be contiguous
        label_map = dict(zip(list(set(self.raw_labels)), range(len(cur_target_names))))
        self.raw_labels = [label_map[l] for l in self.raw_labels]

        self.raw_names = cur_target_names
        self.cache_dir = os.path.join(self.data_root, 'newsgroup', 'cache')
        self.num_cats = len(cur_target_names)


class SupNewsGroup(BaseSupNLPDataset):

    def load_data(self, train):
        newsgroup = fetch_20newsgroups(subset='all', data_home=os.path.join(self.data_root, 'newsgroup'))

        # split this into classes
        data = np.array(newsgroup.data)
        target = np.array(newsgroup.target)
        target_names = newsgroup.target_names

        test_names = self.rs.choice(target_names, 5, replace=False)
        train_names = np.array(list(set(target_names) - set(test_names)))

        if train:
            cur_target_names = train_names
        else:
            cur_target_names = test_names

        target_indices = [target_names.index(name) for name in cur_target_names]
        target_indices = np.array(target_indices)
        target_indices = np.in1d(target, target_indices)

        self.raw_texts = data[target_indices].tolist()
        self.raw_labels = target[target_indices].tolist()

        # remap labels to be contiguous
        label_map = dict(zip(list(set(self.raw_labels)), range(len(cur_target_names))))
        self.raw_labels = [label_map[l] for l in self.raw_labels]

        self.raw_names = cur_target_names
        self.cache_dir = os.path.join(self.data_root, 'newsgroup', 'cache')
        self.num_cats = len(cur_target_names)
