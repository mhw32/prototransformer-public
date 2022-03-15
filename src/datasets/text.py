"""Compilation of datasets for few-shot text classification.

Few-shot Text Classification with Distributional Signatures
Yujia Bao, Menghua Wu, Shiyu Chang and Regina Barzilay.

https://arxiv.org/pdf/1908.06039.pdf

@inproceedings{
    bao2020fewshot,
    title={Few-shot Text Classification with Distributional Signatures},
    author={Yujia Bao and Menghua Wu and Shiyu Chang and Regina Barzilay},
    booktitle={International Conference on Learning Representations},
    year={2020}
}
"""

import os
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset
from collections import Counter, defaultdict
from transformers import RobertaTokenizer


class BaseFewShotTextDataset(Dataset):

    def __init__(
            self,
            data_root,
            n_ways=5,
            n_shots=5,
            n_queries=25,
            split='train',
            roberta_device='cpu',
            fix_seed=42,
        ):
        super().__init__()

        self.data_root = data_root
        self.cache_dir = os.path.realpath(os.path.join(self.data_root, '../cache'))
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.split = split
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.roberta_device = roberta_device
        self.rs = np.random.RandomState(fix_seed)
        self.fix_seed = fix_seed

        self.max_seq_len = 512
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_index = self.tokenizer.pad_token_id
        self.mask_index = self.tokenizer.mask_token_id
        self.mask_token = self.tokenizer.mask_token

        print('loading data...')
        data, self.classes = self.load_data()
        # NOTE: no side information since we don't have anything special
        # NOTE: no smlmt for simplicitly
        self.tokens, self.masks, self.labels = self.process_data(data)

    def update_n_shots(self, new_shots):
        self.n_shots = new_shots
        print("UPDATED SHOT VALUE: ", self.n_shots)

    def update_n_ways(self, new_ways):
        self.n_ways = new_ways
        print("UPDATED WAYS: ", self.n_ways)

    def make_classes(self):
        raise NotImplementedError

    def load_data(self):
        train_classes, val_classes, test_classes = self.make_classes()

        if self.split == 'train':
            classes = train_classes
        elif self.split == 'val':
            classes = val_classes
        elif self.split == 'test':
            classes = test_classes
        else:
            raise Exception(f'split {self.split} not supported.')

        all_data = _load_json(self.data_root)

        # partition data with classes!
        data = []
        for example in all_data:
            if example['label'] in classes:
                data.append(example)

        return data, classes

    def process_data(self, data):
        texts = [row['text'] for row in data]
        labels = [row['label'] for row in data]

        tokens, masks = [], []
        for text in texts:
            outputs = self.tokenizer(
                ' '.join(text),
                truncation=True,
                padding='max_length',
                max_length=self.max_seq_len,
                pad_to_max_length=True,
                return_tensors='pt',
            )
            tokens.append(outputs['input_ids'])
            masks.append(outputs['attention_mask'])

        labels = np.array(labels)
        return tokens, masks, labels

    def prep_smlmt_task(self, data):
        all_text = [row['text'] for row in data]

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

    def build_smlmt_task(self, smlmt_mapping, data):
        smlmt_words = list(smlmt_mapping.keys())
        words = self.rs.choice(smlmt_words, self.n_ways, replace=False)
        data = []

        for i, word in enumerate(words):
            data_i =  {}
            toks_i = smlmt_mapping[word][:100]  # at most 100
            for text in toks_i:
                # perform the masking of ALL instances
                text = np.array(text)
                text[text == word] = self.mask_token
                text = text.tolist()
                data_i['text'] = text
                data_i['label'] = i
            data.append(data_i)

        return data

    def __getitem__(self, index):
        categories = self.rs.choice(self.classes, size=self.n_ways, replace=False)
        task_tokens = []
        task_masks  = []
        task_labels = []
        for c in range(len(categories)):
            category = categories[c]
            indices = np.where(self.labels == category)[0]
            should_replace = True if len(indices) < (self.n_shots+self.n_queries) else False
            indices = self.rs.choice(indices, size=self.n_shots+self.n_queries, replace=should_replace)
            # task_tokens_i : (n_shots+n_queries) x 512
            task_tokens_i = torch.stack([self.tokens[ix] for ix in indices])
            # task_masks_i : (n_shots+n_queries) x 512
            task_masks_i = torch.stack([self.masks[ix] for ix in indices])
            # task_labels_i : (n_shots+n_queries)
            task_labels_i = torch.zeros(self.n_shots+self.n_queries).long() + c
            task_tokens.append(task_tokens_i)
            task_masks.append(task_masks_i)
            task_labels.append(task_labels_i)
        # task_tokens : n_ways x (n_shots+n_queries) x 512
        task_tokens = torch.stack(task_tokens)
        # task_masks : n_ways x (n_shots+n_queries) x 512
        task_masks = torch.stack(task_masks)
        # task_labels : n_ways x (n_shots+n_queries)
        task_labels = torch.stack(task_labels)
        # task_lengths : n_ways x (n_shots+n_queries)
        task_lengths = torch.sum(task_masks, dim=2)

        task_dict = dict(
            support_toks=task_tokens[:, :self.n_shots].long(),
            support_masks=task_masks[:, :self.n_shots].long(),
            support_labs=task_labels[:, :self.n_shots].long(),
            support_lens=task_lengths[:, :self.n_shots].long(),
            # --
            query_toks=task_tokens[:, -self.n_queries:].long(),
            query_masks=task_masks[:, -self.n_queries:].long(),
            query_labs=task_labels[:, -self.n_queries:].long(),
            query_lens=task_lengths[:, -self.n_queries:].long(),
            # --
            task_type=0,
        )
        return task_dict

    def num_episodes(self):
        if self.split == 'train':
            return 100
        elif self.split == 'val':
            return 100
        elif self.split == 'test':
            return 1000
        else:
            raise Exception(f'Split {self.split} not supported.')

    def __len__(self):  # number of episodes
        return self.num_episodes()


class FewShot20News(BaseFewShotTextDataset):
    LABEL_DICT = {
        'talk.politics.mideast': 0,
        'sci.space': 1,
        'misc.forsale': 2,
        'talk.politics.misc': 3,
        'comp.graphics': 4,
        'sci.crypt': 5,
        'comp.windows.x': 6,
        'comp.os.ms-windows.misc': 7,
        'talk.politics.guns': 8,
        'talk.religion.misc': 9,
        'rec.autos': 10,
        'sci.med': 11,
        'comp.sys.mac.hardware': 12,
        'sci.electronics': 13,
        'rec.sport.hockey': 14,
        'alt.atheism': 15,
        'rec.motorcycles': 16,
        'comp.sys.ibm.pc.hardware': 17,
        'rec.sport.baseball': 18,
        'soc.religion.christian': 19,
    }

    def make_classes(self):
        train_classes = []
        for key in self.LABEL_DICT.keys():
            if key[:key.find('.')] in ['sci', 'rec']:
                train_classes.append(self.LABEL_DICT[key])

        val_classes = []
        for key in self.LABEL_DICT.keys():
            if key[:key.find('.')] in ['comp']:
                val_classes.append(self.LABEL_DICT[key])

        test_classes = []
        for key in self.LABEL_DICT.keys():
            if key[:key.find('.')] not in ['comp', 'sci', 'rec']:
                test_classes.append(self.LABEL_DICT[key])

        return train_classes, val_classes, test_classes


class FewShotAmazon(BaseFewShotTextDataset):
    LABEL_DICT = {
        'Amazon_Instant_Video': 0,
        'Apps_for_Android': 1,
        'Automotive': 2,
        'Baby': 3,
        'Beauty': 4,
        'Books': 5,
        'CDs_and_Vinyl': 6,
        'Cell_Phones_and_Accessories': 7,
        'Clothing_Shoes_and_Jewelry': 8,
        'Digital_Music': 9,
        'Electronics': 10,
        'Grocery_and_Gourmet_Food': 11,
        'Health_and_Personal_Care': 12,
        'Home_and_Kitchen': 13,
        'Kindle_Store': 14,
        'Movies_and_TV': 15,
        'Musical_Instruments': 16,
        'Office_Products': 17,
        'Patio_Lawn_and_Garden': 18,
        'Pet_Supplies': 19,
        'Sports_and_Outdoors': 20,
        'Tools_and_Home_Improvement': 21,
        'Toys_and_Games': 22,
        'Video_Games': 23
    }

    def make_classes(self):
        train_classes = [2, 3, 4, 7, 11, 12, 13, 18, 19, 20]
        val_classes = [1, 22, 23, 6, 9]
        test_classes = [0, 5, 14, 15, 8, 10, 16, 17, 21]

        return train_classes, val_classes, test_classes


class FewShotHuffPost(BaseFewShotTextDataset):

    def make_classes(self):
        train_classes = list(range(20))
        val_classes = list(range(20,25))
        test_classes = list(range(25,41))
        return train_classes, val_classes, test_classes


class FewShotRCV1(BaseFewShotTextDataset):

    def make_classes(self):
        train_classes = [1, 2, 12, 15, 18, 20, 22, 25, 27, 32, 33, 34, 38, 39,
                         40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                         54, 55, 56, 57, 58, 59, 60, 61, 66]
        val_classes = [5, 24, 26, 28, 29, 31, 35, 23, 67, 36]
        test_classes = [0, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 21, 30, 37,
                        62, 63, 64, 65, 68, 69, 70]
        return train_classes, val_classes, test_classes


class FewShotReuters(BaseFewShotTextDataset):

    def make_classes(self):
        train_classes = list(range(15))
        val_classes = list(range(15,20))
        test_classes = list(range(20,31))
        return train_classes, val_classes, test_classes


class FewShotFewRel(BaseFewShotTextDataset):

    def make_classes(self):
        # head=WORK_OF_ART validation/test split
        train_classes = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                        22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                        39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                        59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                        76, 77, 78]
        val_classes = [7, 9, 17, 18, 20]
        test_classes = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]
        return train_classes, val_classes, test_classes


def _load_json(path, max_seq_len=512):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)
            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1
            item = {'label': int(row['label']),
                    'text': row['text'][:max_seq_len]}
            text_len.append(len(row['text']))
            keys = ['head', 'tail', 'ebd_id']
            for k in keys:
                if k in row:
                    item[k] = row[k]
            data.append(item)
    return data
