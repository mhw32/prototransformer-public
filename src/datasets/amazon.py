import os
import re
import copy
import string
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset
from collections import Counter, defaultdict

from src.models.sentencebert import SentenceBERT
from transformers import RobertaTokenizer


class BaseFewShotAmazonSentiment(Dataset):

    def load_data(self, data_root, split='train'):
        train_domains, test_domains = get_domains(
            data_root,
            'workspace.filtered.list',
            'workspace.target.list')
        if split == 'train':
            data = get_train_data(data_root, train_domains)
        elif split == 'val':
            data, _ = get_test_data(data_root, test_domains)
        elif split == 'test':
            _, data = get_test_data(data_root, test_domains)
        else:
            raise Exception(f'Split {split} not supported.')
        return data

    def tokenize_data(self, tokenizer, data, keys, max_seq_len=512):
        for task_key in keys:
            task_neg_data = data[task_key]['neg']['data']
            task_pos_data = data[task_key]['pos']['data']

            neg_token_seqs, neg_token_masks = [], []
            pos_token_seqs, pos_token_masks = [], []

            for i in range(len(task_neg_data)):
                output_i = tokenizer(
                    ' '.join(task_neg_data[i]),
                    truncation=True,
                    padding='max_length',
                    max_length=max_seq_len,
                    pad_to_max_length=True,
                    return_tensors='pt',
                )
                neg_token_seqs.append(output_i['input_ids'])
                neg_token_masks.append(output_i['attention_mask'])

            for i in range(len(task_pos_data)):
                output_i = tokenizer(
                    ' '.join(task_pos_data[i]),
                    truncation=True,
                    padding='max_length',
                    max_length=max_seq_len,
                    pad_to_max_length=True,
                    return_tensors='pt',
                )
                pos_token_seqs.append(output_i['input_ids'])
                pos_token_masks.append(output_i['attention_mask'])

            data[task_key]['neg']['tokens'] = neg_token_seqs
            data[task_key]['neg']['masks'] = neg_token_masks
            data[task_key]['pos']['tokens'] = pos_token_seqs
            data[task_key]['pos']['masks'] = pos_token_masks

    def embed_names(self, keys, roberta_device='cpu'):
        cache = {'none': torch.zeros(768)}  # for SMLMT
        names = [key.split('.')[0] for key in keys]
        bert = SentenceBERT(
            version='bert-base-nli-stsb-mean-tokens',
            device=roberta_device,
        )
        embs = bert(names, batch_size=32)
        embs = embs.detach().cpu()

        for i in range(len(names)):
            cache[names[i]] = embs[i]

        return cache


class FewShotAmazonSentiment(BaseFewShotAmazonSentiment):
    """Used for meta-training + quick evaluation."""

    def __init__(
            self,
            data_root,
            n_shots=5,
            n_queries=27,
            train=True,
            roberta_device='cpu',
            smlmt_tasks_factor=0,
            fix_seed=42,
        ):
        super().__init__()

        self.data_root = data_root
        self.cache_dir = os.path.join(self.data_root, 'cache')
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.fix_seed = fix_seed
        self.rs = np.random.RandomState(fix_seed)
        self.roberta_device = roberta_device

        self.n_shots = n_shots
        self.n_queries = n_queries
        self.max_seq_len = 512

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_index = self.tokenizer.pad_token_id
        self.mask_index = self.tokenizer.mask_token_id
        self.mask_token = self.tokenizer.mask_token

        # load raw data
        split = 'train' if train else 'val'
        print('loading data...')
        self.data = self.load_data(self.data_root, split=split)
        self.keys = list(self.data.keys())
        self.n_tasks = len(self.data)
        self.n_smlmt = int(smlmt_tasks_factor * self.n_tasks)
        print(f"n_smlmt is: {n_smlmt} because it's {smlmt_tasks_factor} (factor) * {self.n_tasks} (n_tasks)")

        if self.n_smlmt > 0:
            smlmt_cache = os.path.join(self.cache_dir, f'smlmt_mapping_{split}.pkl')
            if os.path.isfile(smlmt_cache):
                with open(smlmt_cache, 'rb') as fp:
                    smlmt_mapping = pickle.load(fp)
            else:
                print('creating smlmt tasks...')
                smlmt_mapping = self.prep_smlmt_task(self.data, self.keys)
                with open(smlmt_cache, 'wb') as fp:
                    pickle.dump(dict(smlmt_mapping), fp)

        for m in range(self.n_smlmt):
            smlmt_key = f'none.t{m}'
            smlmt_data = self.build_smlmt_task(smlmt_mapping, smlmt_key)
            self.data[smlmt_key] = smlmt_data
            self.keys = list(self.data.keys())

        # embed side information
        print('embedding side info...')
        self.names_dict = self.embed_names(self.keys, self.roberta_device)

        # add roberta encodings to self.data
        print('tokenizing data...')
        self.tokenize_data(self.tokenizer, self.data, self.keys,
                           max_seq_len=self.max_seq_len)

    def prep_smlmt_task(self, data, keys):
        all_text = []
        for key in keys:
            pos_text = data[key]['pos']['data']
            neg_text = data[key]['neg']['data']
            all_text += pos_text
            all_text += neg_text

        unique_text = []
        for i in range(len(all_text)):
            text_i = np.unique(all_text[i])
            unique_text.append(text_i)

        unique_text = np.concatenate(unique_text)
        freqs = Counter(unique_text)

        valid_words = []
        for word, fr in freqs.items():
            if fr >= (self.n_shots + self.n_queries):
                valid_words.append(word)

        # these are the tokens with enough
        # labels to choose from!
        smlmt_cats = np.array(valid_words)

        # now we need to map each of these cats to
        # the indices of sentences that contain them
        smlmt_mapping = defaultdict(lambda: [])
        pbar = tqdm(total=len(all_text))
        for text in all_text:
            tokens = set(text)
            for word in smlmt_cats:
                if word in tokens:
                    smlmt_mapping[word].append(text)
            pbar.update()
        pbar.close()

        # maps valid category to all sequences containing it
        return smlmt_mapping

    def build_smlmt_task(self, smlmt_mapping, smlmt_key):
        smlmt_words = list(smlmt_mapping.keys())
        words = self.rs.choice(smlmt_words, 2, replace=False)

        data = {
            'pos': {
                'filename': smlmt_key,
                'is_smlmt': True,
                'data': None,
                'target': None,
            },
            'neg': {
                'filename': smlmt_key,
                'is_smlmt': True,
                'data': None,
                'target': None,
            }
        }

        for i, word in enumerate(words):
            data_i = self.rs.choice(
                smlmt_mapping[word], size=self.n_shots + self.n_queries, replace=False)
            masked_data_i = []
            for text in data_i:
                # perform the masking of ALL instances
                text = np.array(text)
                text[text == word] = self.mask_token
                text = text.tolist()
                masked_data_i.append(text)

            if i == 0:
                targets_i = [0 for _ in range(len(masked_data_i))]
                data['neg']['data'] = masked_data_i
                data['neg']['target'] = targets_i
            else:
                targets_i = [1 for _ in range(len(masked_data_i))]
                data['pos']['data'] = masked_data_i
                data['pos']['target'] = targets_i

        return data

    def __getitem__(self, index):
        task_key = self.keys[index]
        data = self.data[task_key]
        filename = data['pos']['filename']

        task_type = 0
        if 'is_smlmt' in data['pos']:
            if data['pos']['is_smlmt']:
                task_type = 1

        num_pos = len(data['pos']['data'])
        num_neg = len(data['neg']['data'])

        # randomly choose a support set and query set
        pos_indices = self.rs.choice(
            np.arange(num_pos), self.n_shots + self.n_queries,
            replace=True if num_pos < (self.n_shots + self.n_queries) else False).tolist()
        neg_indices = self.rs.choice(
            np.arange(num_neg), self.n_shots + self.n_queries,
            replace=True if num_neg < (self.n_shots + self.n_queries) else False).tolist()

        pos_tokens = torch.stack([data['pos']['tokens'][ix] for ix in pos_indices])   # n_shots+n_queries x 512
        pos_masks = torch.stack([data['pos']['masks'][ix] for ix in pos_indices])     # n_shots+n_queries x 512
        pos_labels = torch.LongTensor([data['pos']['target'][ix] for ix in pos_indices])

        neg_tokens = torch.stack([data['neg']['tokens'][ix] for ix in neg_indices])   # n_shots+n_queries x 512
        neg_masks = torch.stack([data['neg']['masks'][ix] for ix in neg_indices])     # n_shots+n_queries x 512
        neg_labels =  torch.LongTensor([data['neg']['target'][ix] for ix in neg_indices])

        tokens = torch.stack([neg_tokens, pos_tokens]).squeeze(2)  # 2 x (n_shots+n_queries) x 512
        masks = torch.stack([neg_masks, pos_masks]).squeeze(2)     # 2 x (n_shots+n_queries) x 512
        labels = torch.stack([neg_labels, pos_labels])             # 2 x (n_shots+n_queries)
        lengths = torch.sum(masks, dim=2)                          # 2 x (n_shots+n_queries)

        filename = filename.split('.')[0]

        task_dict = dict(
            support_toks=tokens[:, :self.n_shots].long(),
            support_masks=masks[:, :self.n_shots].long(),
            support_labs=labels[:, :self.n_shots].long(),
            support_lens=lengths[:, :self.n_shots].long(),
            # --
            query_toks=tokens[:, -self.n_queries:].long(),
            query_masks=masks[:, -self.n_queries:].long(),
            query_labs=labels[:, -self.n_queries:].long(),
            query_lens=lengths[:, -self.n_queries:].long(),
            # --
            side_info=self.names_dict[filename].float(),  # 512
            task_type=task_type,
        )
        return task_dict

    def __len__(self):
        return self.n_tasks + self.n_smlmt


class EvalFewShotAmazonSentiment(BaseFewShotAmazonSentiment):
    """Use the first n_shots as support set and compute accuracy
    for the rest of the examples in the "test_task_index"-th task.
    """
    def __init__(
            self,
            data_root,
            test_task_index,
            n_shots=5,
            roberta_device='cpu',
            fix_seed=42,
        ):
        super().__init__()

        self.data_root = data_root
        self.cache_dir = os.path.join(self.data_root, 'cache')
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.fix_seed = fix_seed
        self.rs = np.random.RandomState(fix_seed)
        self.roberta_device = roberta_device

        self.n_shots = n_shots
        self.max_seq_len = 512

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_index = self.tokenizer.pad_token_id
        self.mask_index = self.tokenizer.mask_token_id

        # load raw data
        data = self.load_data(self.data_root, split='test')  # NOTE: test dset
        keys = list(data.keys())
        key = keys[test_task_index]
        data = {key: data[key]}

        # embed side information
        self.names_dict = self.embed_names([key], self.roberta_device)
        print(self.names_dict)

        # add roberta encodings to self.data
        self.tokenize_data(self.tokenizer, data, [key], max_seq_len=self.max_seq_len)
        self.data = data
        self.key = key

        self.support_toks, self.support_masks, self.support_labs, self.support_lens = \
            self.get_support_set()

        # concatenate all pos and neg data
        data[key]['all'] = {'tokens': [], 'target': [], 'filename': None}
        data[key]['all']['filename'] = data[key]['pos']['filename']
        data[key]['all']['tokens'] = data[key]['neg']['tokens'] + data[key]['pos']['tokens']
        data[key]['all']['masks'] = data[key]['neg']['masks'] + data[key]['pos']['masks']
        data[key]['all']['target'] = data[key]['neg']['target'] + data[key]['pos']['target']
        self.size = len(data[key]['all']['tokens'])

    def get_support_set(self):
        data = self.data[self.key]
        pos_support_toks = torch.stack(data['pos']['tokens'][:self.n_shots])
        pos_support_masks = torch.stack(data['pos']['masks'][:self.n_shots])
        pos_support_labs = torch.LongTensor(data['pos']['target'][:self.n_shots])
        neg_support_toks = torch.stack(data['neg']['tokens'][:self.n_shots])
        neg_support_masks = torch.stack(data['neg']['masks'][:self.n_shots])
        neg_support_labs = torch.LongTensor(data['neg']['target'][:self.n_shots])

        # use the SAME support set regardless
        support_toks = torch.stack([neg_support_toks, pos_support_toks]).squeeze(2)
        support_masks = torch.stack([neg_support_masks, pos_support_masks]).squeeze(2)
        support_labs = torch.stack([neg_support_labs, pos_support_labs])
        support_lens = torch.sum(support_masks, dim=2)

        return support_toks, support_masks, support_labs, support_lens

    def __getitem__(self, index):
        data = self.data[self.key]
        filename = data['all']['filename']

        query_toks = data['all']['tokens'][index].squeeze()
        query_masks = data['all']['masks'][index].squeeze()
        query_labs = data['all']['target'][index]
        query_lens = torch.sum(query_masks)

        filename = filename.split('.')[0]

        task_dict = dict(
            query_toks=query_toks.long(),    # 512
            query_masks=query_masks.long(),  # 512
            query_labs=query_labs,    # 1
            query_lens=query_lens,    # 1
            # --
            side_info=self.names_dict[filename].float(),  # 512
        )
        return task_dict

    def __len__(self):
        return self.size


def get_train_data(data_path, domains):
    # https://github.com/zhongyuchen/few-shot-text-classification
    train_data = _get_data(data_path, domains, 'train')
    return train_data


def get_test_data(data_path, domains):
    # https://github.com/zhongyuchen/few-shot-text-classification

    # get dev, test data
    support_data = _get_data(data_path, domains, 'train')
    dev_data = _get_data(data_path, domains, 'dev')
    test_data = _get_data(data_path, domains, 'test')

    # support -> dev, test
    dev_data = _combine_data(support_data, dev_data)
    test_data = _combine_data(support_data, test_data)
    return dev_data, test_data


def get_domains(data_path, filtered_name, target_name):
    # https://github.com/zhongyuchen/few-shot-text-classification
    all_domains = _parse_list(data_path, filtered_name)
    test_domains = _parse_list(data_path, target_name)
    train_domains = all_domains - test_domains
    return sorted(list(train_domains)), sorted(list(test_domains))


def _parse_list(data_path, list_name):
    domain = set()
    with open(os.path.join(data_path, list_name), 'r', encoding='utf-8') as f:
        for line in f:
            domain.add(line.strip('\n'))
    return domain


def _get_data(data_path, domains, usage):
    # usage in ['train', 'dev', 'test']
    data = {}
    for domain in domains:
        for t in ['t2', 't4', 't5']:
            filename = '.'.join([domain, t, usage])
            neg, pos = _parse_data(data_path, filename)
            neg = _process_data(neg)
            pos = _process_data(pos)
            data[filename] = {'neg': neg, 'pos': pos}
    return data


def _process_data(data_dict):
    for i in range(len(data_dict['data'])):
        text = data_dict['data'][i]
        # ignore string.punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        # string.whitespace -> space
        text = re.sub('[%s]' % re.escape(string.whitespace), ' ', text)
        # lower case
        text = text.lower()
        # split by whitespace
        text = text.split()
        # replace
        data_dict['data'][i] = text
    return data_dict


def _parse_data(data_path, filename):
    neg = {
        'filename': filename,
        'data': [],
        'target': []
    }
    pos = {
        'filename': filename,
        'data': [],
        'target': []
    }
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            if line[-2:] == '-1':
                neg['data'].append(line[:-2])
                neg['target'].append(0)
            else:
                pos['data'].append(line[:-1])
                pos['target'].append(1)
    return neg, pos


def _combine_data(support_data, data):
    # support -> dev, test
    for key in data:
        key_split = key.split('.')[0:-1] + ['train']
        support_key = '.'.join(key_split)
        for value in data[key]:
            data[key][value]['support_data'] = \
                copy.deepcopy(support_data[support_key][value]['data'])
            data[key][value]['support_target'] = \
                copy.deepcopy(support_data[support_key][value]['target'])
    return data
