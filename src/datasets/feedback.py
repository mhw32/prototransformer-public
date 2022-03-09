import os, io, re, json, copy, random, hashlib
import torch, numpy as np, pandas as pd
from tqdm import tqdm
import tokenize as pythonlang
from transformers import RobertaTokenizer
from collections import defaultdict, Counter
from torch.utils.data.dataset import Dataset

from src.models.sentencebert import SentenceBERT
from src.utils.utils import OrderedCounter, string_concat
from src.utils.python_utils import PYTHON_KEYWORDS, camel_to_snake

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
MASK_TOKEN = '<mask>'
ENTER_SCOPE_TOKEN = '<scope>'
EXIT_SCOPE_TOKEN = '</scope>'
NEWLINE_TOKEN = '<newline>'
LEEWAY = 5

class MetaDTSolutions(Dataset):
    """
    Dataset of student solutions to the DT dataset of physics problems
    """
    def __init__(
            self,
            n_shots,
            n_queries,
            data_root=None,                  # root of the dataset
            answers_path=None,               # path to exam files from root
            rubric_path=None,                # path to rubrics from root
            cache_path=None,                 # where to cache RoBerta embeddings of rubric names if roberta_rubric = True
            vocab=None,
            augment_by_rubric=True,          # if True, randomly shuffle programs with identical rubrics in training
            roberta_rubric=True,             # if True, map rubric text to RoBERTa encoding
            roberta_prompt=True,             # if True, map question text to RoBERTa encoding
            roberta_tokenize=False,          # if True, use RoBERTa tokenizer to tokenize
            roberta_device='cpu',            # used for cache-ing rubric / prompt embeddings
            roberta_config='microsoft/codebert-base',  # which pretrained RoBERTa to use for tokenizing
            max_seq_len=1000,                # maximum # of tokens
            min_occ=1,                       # minimum number of occurences to keep a token in vocab (only used if not RoBERTa)
            conservative=True,               # if not conservative, keep more tasks (even unbalanced onces)
            train=True,
            train_frac=0.9,
            hold_out_split=True,             # if True, we hold out questions
            enforce_binary=False,            # if True, all tasks are binary prediction problems
            fix_seed=True,
            pad_to_max_num_class=False,      # if True, all tasks are padded to the maximum # of classes
            min_task_answers=50,
            simple_binary=False,             # If True, we make each rubric its own binary task instead of just the plurality class
        ):
        super(MetaDTSolutions, self).__init__()

        # I THINK: making cache directories
        if cache_path:
            cache_path = os.path.join(data_root, cache_path)

            if not os.path.isdir(cache_path):
                os.makedirs(cache_path)

        if not train and not roberta_tokenize:
            assert vocab is not None

        if roberta_tokenize:
            # if we use roberta, turn off a bunch of other features
            vocab = None

        if enforce_binary:
            pad_to_max_num_class = False

        self.train = train
        self.train_frac = train_frac
        self.conservative = conservative
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.min_occ = min_occ
        self.answers_path = os.path.join(data_root, answers_path)
        self.rubric_path = os.path.join(data_root, rubric_path)
        self.fix_seed = fix_seed
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.rs = np.random.RandomState(fix_seed)
        self.hold_out_split = hold_out_split
        self.enforce_binary = enforce_binary
        self.roberta_tokenize = roberta_tokenize
        self.pad_to_max_num_class = pad_to_max_num_class
        self.roberta_device = roberta_device
        self.min_task_answers = min_task_answers
        self.simple_binary = simple_binary

        print('loading exams...')
        answers, indices, labels, tasks, \
        questions, task_splits, task_classes, task_stats, \
        rubrics, prompts, equivalences = self.load_exam_data()
        print(len(answers))
        print("answers", len(answers))
        print("indices", len(indices))
        print("labels", len(labels))
        print("tasks", len(tasks))
        print("questions", len(questions))
        print("task_splits", len(task_splits))
        print("task_classes", len(task_classes))
        print("task_stats", len(task_stats))
        print("rubrics", len(rubrics))
        print("prompts", len(prompts))
        print("equivalences", len(equivalences))

        # NOTE: loading assignment data has been masked in public repo
        answers, indices, labels, tasks, questions, task_splits, \
        task_classes, task_stats, init_task_types, rubrics, prompts, equivalences = \
            self.combine_data_sources(
                [answers], [indices], [labels], [tasks],
                [questions], [task_splits], [task_classes], [task_stats],
                [rubrics], [prompts], [equivalences])
        self.task_classes = task_classes
        self.task_stats = task_stats

        if hold_out_split:
            tasks_in_split = np.array([k for k, v in task_splits.items() if v != train])
            split_indices = np.in1d(tasks, tasks_in_split)
            split_indices = np.where(split_indices)[0]
        else:
            # meta tasks train test split randomly based on task.
            split_indices = self.train_test_split_by_question(
                questions, train=train, train_frac=train_frac)

        # indices : remaining indices in the current split
        indices = [indices[i] for i in split_indices]
        labels = [labels[i] for i in split_indices]
        tasks = [tasks[i] for i in split_indices]
        init_task_types = [init_task_types[t] for t in np.unique(tasks)]
        rubrics = [rubrics[t] for t in np.unique(tasks)]
        prompts = [prompts[t] for t in np.unique(tasks)]

        # I THINK: generates roberta labels only once
        if roberta_rubric:
            print(cache_path)
            rubric_bert_cache = self.roberta_embed_rubrics(
                rubrics, cache_path, key='human_label')
            prompt_bert_cache = self.roberta_embed_prompts(
                prompts, cache_path)
        else:
            rubric_bert_cache = None
            prompt_bert_cache = None

        # remap indices to be contiguous and just grab relevant answers.
        unique_indices = np.unique(indices).tolist()
        index_mapping = dict(zip(unique_indices, range(len(unique_indices))))
        indices = [index_mapping[index] for index in indices]
        answers = [answers[i] for i in unique_indices]

        # this is to find equivalent "answers" semantically
        equivalences = remap_equivalences(equivalences, index_mapping)
        equivalences = [equivalences[i] for i in range(len(unique_indices))]

        # Use R
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_config)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_index = self.tokenizer.pad_token_id

        print('processing entries with vocab...')
        token_seqs, token_lens = self.process_answers(
            answers,
            roberta_tokenize=roberta_tokenize)

        print('constructing meta info...')
        indices_by_task, labels_by_task = self.prepare_meta(indices, tasks, labels)

        num_tasks = len(indices_by_task)

        assert num_tasks == len(init_task_types)
        task_types = init_task_types

        print('done')
        self.answers = answers
        self.rubrics = rubrics
        self.prompts = prompts
        self.rubric_bert_cache = rubric_bert_cache
        self.prompt_bert_cache = prompt_bert_cache
        self.roberta_rubric = roberta_rubric
        self.roberta_prompt = roberta_prompt
        self.roberta_config = roberta_config
        if train:
            self.equivalences = equivalences
        else:
            # we should not be able to use this is validation
            self.equivalences = None
        self.token_seqs = token_seqs
        self.token_lens = token_lens
        self.max_index = max(indices)
        self.indices_by_task = indices_by_task
        self.labels_by_task = labels_by_task
        self.task_ids = [int(x) for x in sorted(list(indices_by_task.keys()))]
        self.augment_by_rubric = augment_by_rubric
        self.task_types = task_types
        self.num_task_types = len(set(task_types))
        self.task_classes = task_classes
        self.max_num_classes = max(self.task_classes.values())

    def load_exam_data(self):
        df = pd.read_csv(self.answers_path)
        key_df = pd.read_csv(self.rubric_path)

        key_rubrics = list(key_df['rubric'])
        key_rubrics = [json.loads(r) for r in key_rubrics]
        key_rubrics = dict(zip(list(key_df['id']), key_rubrics))

        answers = np.asarray(df['answer'])
        rubrics = np.asarray(df['gradeData'])
        prompts = np.asarray(df['questionText'])

        # id isn't work bc its unique
        tasks = np.asarray(df["questionId"])
        # to delete
        ta = tasks[0]

        ids = np.asarray(df['id']) # TODO: Not sure what this is

        if self.hold_out_split:
            # reserve some questions for test split
            questions = list(set(tasks))
            num_test_questions = max(int((1 - self.train_frac) * len(questions)), 1)
            self.rs.shuffle(questions)
            questions = list(questions)
            test_questions = questions[-num_test_questions:]
            is_test = np.in1d(tasks, test_questions)

        print('parsing rubrics...')
        print(f"First {10} rubrics are: {rubrics[:10]}")
        # replace rubrics with string format with key_rubrics
        # I THINK: rubrics is originally a list of JSON strings, which each corrspond for a certain question to
        # a dictionary of rubric_item_id : 0/1? And then we replace rubric_item_id with a JSON string
        # of the dictionary for that rubric (containing more info about it)
        new_rubrics = []
        for task, rubric, task_id in zip(tasks, rubrics, ids):
            rubric = json.loads(rubric)

            if task in key_rubrics:
                new_rubric = {}
                rubric_map = key_rubrics[task]
                for key in rubric:
                    if key in rubric_map:
                        new_key = {
                            'name': int(task),
                            'id': int(task_id),
                            'human_label': rubric_map[key],
                            'abbrev_label': key,
                        }
                        new_key = json.dumps(new_key)
                        new_rubric[new_key] = rubric[key]
                    else:
                        new_key = {
                            'name': int(task),
                            'id': int(task_id),
                            'human_label': key,
                            'abbrev_label': key,
                        }
                        new_key = json.dumps(new_key)
                        new_rubric[new_key] = rubric[key]
                new_rubrics.append(new_rubric)
            else:
                new_rubric = {}
                for key in rubric:
                    new_key = {
                        'name': int(task),
                        'id': int(task_id),
                        'human_label': key,
                        'abbrev_label': key,
                    }
                    new_key = json.dumps(new_key)
                    new_rubric[new_key] = rubric[key]
                new_rubrics.append(new_rubric)

        # replace rubric with new_rubric
        rubrics = np.asarray(new_rubrics)

        # build equivalence maps from index to a set of indices
        # that are answers with the exact same output
        equivalence_maps = build_many_equivalences(tasks, rubrics) # TODO: Not sure why we do this

        # map tasks to integers
        unique_tasks = sorted(set(tasks))
        task_mapping = dict(zip(unique_tasks, range(len(unique_tasks))))
        tasks = np.array([task_mapping[t] for t in tasks])

        # we use indices to track answers
        # v0: we worked directly on program strings but this quickly
        #     got expensive as we duplicate answers excessively.
        # v1: we work on indices and look up answers. Main advantage
        #     is that we only have to compile answers once.
        indices = np.arange(len(answers))

        print('building tasks...')
        print(f"confirming that tasks still has length {len(tasks)}")
        # a task is now defined by several labels, we can duplicate each to new tasks
        indices, labels, tasks, questions, task_splits, \
        task_classes, task_stats, rubric_maps, prompt_maps = \
            construct_tasks_by_rubric(
                indices, rubrics, prompts, tasks, is_test)

        # We remove any tasks where there are fewer than "self.n_shots + self.n_queries" examples
        print(f"after constructing by rubric")
        self.print_debug(tasks, labels, indices)
        if self.conservative:
            print('removing trivial classes...')
            indices, labels, tasks, questions, task_splits, \
            task_classes, task_stats, rubric_maps, prompt_maps = \
                remove_small_classes(
                    indices, labels, tasks, questions, task_splits, task_classes,
                    task_stats, rubric_maps, prompt_maps,
                    min_freq=self.n_shots + self.n_queries) # TODO: Not 100% sure why this is shots + queries...it seems like we only need >=max(shots, queries)
                # since this will only be one or the other

        print(f"after removing small tasks is")
        self.print_debug(tasks, labels, indices)
        if self.enforce_binary:
            print('collapsing tasks to binary...')
            if self.simple_binary:
                indices, labels, tasks, questions, task_splits, \
                task_classes, task_stats, rubric_maps, prompt_maps = \
                    make_binary_tasks(
                        indices, labels, tasks, questions, task_splits, task_classes,
                        task_stats, rubric_maps, prompt_maps)
            else: # I think we want to always take this path, but left the option open
                indices, labels, tasks, questions, task_splits, \
                task_classes, task_stats, rubric_maps, prompt_maps = \
                    make_binary_tasks_liberally(
                        indices, labels, tasks, questions, task_splits, task_classes,
                        task_stats, rubric_maps, prompt_maps)

        print(f"right before returning")
        self.print_debug(tasks, labels, indices)
        return answers, indices, labels, tasks, questions, task_splits, \
               task_classes, task_stats, rubric_maps, prompt_maps, equivalence_maps

    # TODO: delete
    def print_debug(self, tasks, labels, indices, length=10):
        print(f"There are {len(indices)} indices")
        print(f"There are {len(tasks)} tasks")
        labels = np.array(labels)
        tasks = np.array(tasks)
        task_labels = labels[tasks == tasks[0]]
        print(f"task labels are: {task_labels}")

    def build_attention_masks(self, lengths):
        batch_size = len(lengths)
        mask = np.zeros((batch_size, self.max_seq_len))
        for i in range(batch_size):
            mask[i, :lengths[i]] = 1
        return torch.from_numpy(mask).long()

    def process_answers(
            self,
            answers,
            roberta_tokenize,
        ):
        """
        Use Roberta to tokenize
        """
        token_seqs, token_lengths = [], []

        for i in tqdm(range(len(answers))):
            answer = answers[i]

            assert self.max_seq_len == 512
            tokenizer_outputs = self.tokenizer(
                answer,
                truncation=True,
                padding='max_length',
                max_length=self.max_seq_len,
                return_length=True,
                pad_to_max_length=True)

            tokens = tokenizer_outputs['input_ids']
            token_length = tokenizer_outputs['length']

            token_seqs.append(tokens)
            token_lengths.append(token_length)

        return token_seqs, token_lengths

    def combine_data_sources(
            self,
            programs_list,
            indices_list,
            labels_list,
            tasks_list,
            questions_list,
            task_splits_list,
            task_classes_list,
            task_stats_list,
            rubrics_list,
            prompts_list,
            equivalences_list,
        ):
        programs = []
        indices = []
        labels = []
        tasks = []
        questions = []
        task_splits = {}
        task_classes = {}
        task_stats = []
        task_types = []
        rubrics = {}
        prompts = {}
        equivalences = {}

        max_index = 0
        max_task = 0
        max_question = 0
        n = len(programs_list)
        for i in range(n):
            programs.extend(programs_list[i])
            indices_i = indices_list[i]
            indices_i = [d + max_index for d in indices_i]
            indices.extend(indices_i)
            labels.extend(labels_list[i])
            tasks_i = tasks_list[i]
            tasks_i = [t + max_task for t in tasks_i]
            tasks.extend(tasks_i)
            questions_i = questions_list[i]
            questions_i = [q + max_question for q in questions_i]
            questions.extend(questions_i)
            for t, rubs in rubrics_list[i].items():
                assert t + max_task not in rubrics
                rubrics[t + max_task] = rubs
            for t, pmts in prompts_list[i].items():
                assert t + max_task not in prompts
                prompts[t + max_task] = pmts
            for t, splits in task_splits_list[i].items():
                assert t + max_task not in task_splits
                task_splits[t + max_task] = splits
            for t, classes in task_classes_list[i].items():
                assert t + max_task not in task_classes
                task_classes[t + max_task] = classes
            task_stats.extend(task_stats_list[i])
            task_types.extend([i for _ in range(len(task_stats_list[i]))])
            for ix, eqs in equivalences_list[i].items():
                assert ix + max_index not in equivalences
                equivalences[ix + max_index] = eqs
            # increment max tasks
            max_task = int(max(tasks_i) + 1)
            max_index = int(max(indices_i) + 1)
            max_question = int(max(questions_i) + 1)

        return programs, indices, labels, tasks, questions, task_splits, \
               task_classes, task_stats, task_types, rubrics, prompts, equivalences

    def roberta_embed_rubrics(self, rubric_list, cache_path, key='human_label'):
        """
        rubric_list maps a task to a rubric name and list of value names
        """
        print('Computing rubric BERT embeddings...')

        edited_cache = False
        cache_file = os.path.join(
            cache_path,
            f'sentence_bert_rubric_cache_{self.n_shots}shots_{self.n_queries}queries.pth.tar',
        )
        if os.path.exists(cache_file):
            roberta_cache = torch.load(cache_file)
        else:
            roberta_cache = {}

        if 'None' not in roberta_cache:
            roberta_cache['None'] = torch.zeros(768)

        texts = []
        for task in range(len(rubric_list)):
            rubric = rubric_list[task]
            if isinstance(rubric, list):
                r = rubric[0]  # they are all the same
                r = json.loads(r)[key]
                texts.append(r)
            elif isinstance(rubric, str):
                r = json.loads(rubric)[key]
                texts.append(r)
            else:
                raise Exception('Unsupported rubric type.')
        texts = list(set(texts))

        texts_to_run = []  # these are the things we need to run

        for text in texts:
            text = str(text)
            if text in roberta_cache:
                embs = roberta_cache[text]
            elif text == 'None':  # replace this with just zeros
                embs = torch.zeros(768)
                roberta_cache[text] = embs
                edited_cache = True
            else:
                texts_to_run.append(text)
                edited_cache = True

        if len(texts_to_run) > 0:
            assert edited_cache
            bert = SentenceBERT(
                version='bert-base-nli-stsb-mean-tokens',
                device=self.roberta_device,
            )
            embs = bert(texts_to_run, batch_size=32, show_progress_bar=True)
            embs = embs.detach().cpu()

            for i in range(len(texts_to_run)):
                roberta_cache[texts_to_run[i]] = embs[i]

        if edited_cache:
            print('Made edits to cache... saving...')
            torch.save(roberta_cache, cache_file)

        return roberta_cache

    def roberta_embed_prompts(self, prompt_list, cache_path):
        print('Computing prompt BERT embeddings...')

        edited_cache = False
        cache_file = os.path.join(
            cache_path,
            f'sentence_bert_prompt_cache_{self.n_shots}shots_{self.n_queries}queries.pth.tar',
        )
        if os.path.exists(cache_file):
            roberta_cache = torch.load(cache_file)
        else:
            roberta_cache = {}

        if 'None' not in roberta_cache:
            roberta_cache['None'] = torch.zeros(768)

        texts_to_run = []  # these are the things we need to run
        hashs_to_run = []

        for prompt in prompt_list:
            prompt = str(prompt)
            prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            if prompt_hash in roberta_cache:
                embs = roberta_cache[prompt_hash]
            elif prompt == 'None':
                embs = torch.zeros(768)
                roberta_cache[prompt_hash] = embs
                edited_cache = True
            else:
                texts_to_run.append(prompt)
                hashs_to_run.append(prompt_hash)
                edited_cache = True

        if len(texts_to_run) > 0:
            assert edited_cache
            assert len(texts_to_run) == len(hashs_to_run)
            bert = SentenceBERT(
                version='bert-base-nli-stsb-mean-tokens',
                device=self.roberta_device,
            )
            embs = bert(texts_to_run, batch_size=32, show_progress_bar=True)
            embs = embs.detach().cpu()

            for i in range(len(texts_to_run)):
                roberta_cache[hashs_to_run[i]] = embs[i]

        if edited_cache:
            print('Made edits to cache... saving...')
            torch.save(roberta_cache, cache_file)

        return roberta_cache

    def train_test_split_by_question(self, questions, train=True, train_frac=0.9):
        """
        Split tasks into train and test splits but do so preserving that all the
        tasks from one question are put together. Otherwise, it is cheating a bit
        since the same programs where seen in training.
        """
        unique_questions = np.sort(np.unique(questions))
        num_questions = len(unique_questions)

        num_train = int(num_questions * train_frac)
        # important to fix the random seed here we get same split every call
        train_questions = self.rs.choice(unique_questions, num_train, replace=False)
        train_questions = np.sort(train_questions)

        if train:
            split_indices = np.in1d(questions, train_questions)
        else:
            split_indices = ~np.in1d(questions, train_questions)

        split_indices = np.where(split_indices)[0]
        return split_indices

    def prepare_meta(self, indices, tasks, labels):
        def create_item():
            return []

        indices_by_task = defaultdict(create_item)
        labels_by_task  = defaultdict(create_item)

        for i in range(len(indices)):
            indices_by_task[tasks[i]].append(indices[i])
            labels_by_task[tasks[i]].append(labels[i])

        return indices_by_task, labels_by_task

    def __len__(self):
        return len(self.task_ids)

    def __getitem__(self, index):
        task = int(self.task_ids[index])
        task_type = self.task_types[index]

        # --- handle grabbing the rubric embedding ---
        rubric_name = self.rubrics[index]
        if isinstance(rubric_name, list):
            rubric_name = list(set([json.loads(x)['human_label'] for x in rubric_name]))[0]
        elif isinstance(rubric_name, str):
            rubric_name = json.loads(rubric_name)['human_label']
        else:
            raise Exception('type not supported.')

        prompt_text = self.prompts[index]

        if self.roberta_rubric:
            rubric_bert_name = self.rubric_bert_cache[rubric_name]
        else:
            rubric_bert_name = torch.zeros(768)

        if self.roberta_prompt:
            prompt_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()
            prompt_bert_name = self.prompt_bert_cache[prompt_hash]
        else:
            prompt_bert_name = torch.zeros(768)

        # --- done ---

        support_toks, support_lens, support_labs = [], [], []
        query_toks, query_lens, query_labs = [], [], []

        indices = self.indices_by_task[task]
        toks = [self.token_seqs[i] for i in indices]
        lens = [self.token_lens[i] for i in indices]
        labs = np.array(self.labels_by_task[task])

        num_classes = self.task_classes[task]

        rubric_embs = []
        prompt_embs = []
        for cls in range(num_classes):
            valid_indices = np.where(labs == cls)[0]
            sample_with_replace = True if len(valid_indices) < self.n_shots else False
            support_ids_sampled = self.rs.choice(valid_indices, self.n_shots, sample_with_replace)
            support_toks_cls = []
            support_lens_cls = []
            for s in support_ids_sampled:
                if self.train:
                    toks_s = toks[s]
                    lens_s = lens[s]

                    # --- augmentations to make more training tasks ---
                    # don't shuffle entries if this is a cloze/execution/smlmt task.
                    if self.augment_by_rubric:
                        toks_s, new_s = shuffle_entries_augmentation(
                            indices[s],
                            self.equivalences,
                            self.token_seqs,
                            p=0.5,
                        )
                        lens_s = self.token_lens[new_s]
                    # --- done ---
                    support_toks_cls.append(toks_s)
                    support_lens_cls.append(lens_s)
                else:
                    # do nothing. we used to replace unknown tokens with random ones
                    # but i think thats a bad idea
                    support_toks_cls.append(toks[s])
                    support_lens_cls.append(lens[s])
            support_toks.append(support_toks_cls)
            support_lens.append(support_lens_cls)
            support_labs.append([labs[s] for s in support_ids_sampled])

            # do the same for query set
            query_ids = np.setxor1d(valid_indices, support_ids_sampled)
            if len(query_ids) == 0:
                query_ids = valid_indices  # no choice but to resample
            query_replace = False if len(query_ids) > self.n_queries else True
            query_ids = self.rs.choice(query_ids, self.n_queries, query_replace)
            query_toks_cls = []
            query_lens_cls = []
            for s in query_ids:
                query_toks_cls.append(toks[s])
                query_lens_cls.append(lens[s])
            query_toks.append(query_toks_cls)
            query_lens.append(query_lens_cls)
            query_labs.append([labs[s] for s in query_ids])

            rubric_embs.append(rubric_bert_name)
            prompt_embs.append(prompt_bert_name)

        support_toks = np.array(support_toks)
        support_lens = np.array(support_lens)
        support_labs = np.array(support_labs)

        query_toks = np.array(query_toks)
        query_lens = np.array(query_lens)
        query_labs = np.array(query_labs)

        rubric_embs = torch.stack(rubric_embs)
        prompt_embs = torch.stack(prompt_embs)

        if self.pad_to_max_num_class:
            # we want everything to be of the same number of classes
            # so that can organize them as a minibatch. Each sentence
            # is going to be ['<s>', '</s>'] (length 2).
            num_class = support_toks.shape[0]
            if num_class < self.max_num_classes:
                support_fill_toks = np.zeros((self.max_num_classes - num_class,
                                              support_toks.shape[1],
                                              support_toks.shape[2]))
                support_fill_toks[:, :, 0] = self.w2i[SOS_TOKEN]
                support_fill_toks[:, :, 1] = self.w2i[EOS_TOKEN]
                support_fill_toks[:, :, 2:] = self.w2i[PAD_TOKEN]
                support_fill_lens = np.zeros((self.max_num_classes - num_class,
                                              support_lens.shape[1])) + 2
                # filler label is always 0.
                support_fill_labs = np.zeros((self.max_num_classes - num_class,
                                              support_labs.shape[1]))

                query_fill_toks = np.zeros((self.max_num_classes - num_class,
                                            query_toks.shape[1],
                                            query_toks.shape[2]))
                query_fill_toks[:, :, 0] = self.w2i[SOS_TOKEN]
                query_fill_toks[:, :, 1] = self.w2i[EOS_TOKEN]
                query_fill_toks[:, :, 2:] = self.w2i[PAD_TOKEN]
                query_fill_lens = np.zeros((self.max_num_classes - num_class,
                                            query_lens.shape[1])) + 2
                query_fill_labs = np.zeros((self.max_num_classes - num_class,
                                            query_labs.shape[1]))

                support_toks = np.concatenate([support_toks, support_fill_toks], axis=0)
                support_lens = np.concatenate([support_lens, support_fill_lens], axis=0)
                support_labs = np.concatenate([support_labs, support_fill_labs], axis=0)

                query_toks = np.concatenate([query_toks, query_fill_toks], axis=0)
                query_lens = np.concatenate([query_lens, query_fill_lens], axis=0)
                query_labs = np.concatenate([query_labs, query_fill_labs], axis=0)

        support_toks = torch.from_numpy(support_toks).long()
        support_lens = torch.from_numpy(support_lens).long()
        support_labs = torch.from_numpy(support_labs).long()
        support_masks = self.build_attention_masks(support_lens.view(-1))
        support_masks = support_masks.view(num_classes, self.n_shots, -1)

        query_toks = torch.from_numpy(query_toks).long()
        query_lens = torch.from_numpy(query_lens).long()
        query_labs = torch.from_numpy(query_labs).long()
        query_masks = self.build_attention_masks(query_lens.view(-1))
        query_masks = query_masks.view(num_classes, self.n_queries, -1)

        output_dict = dict(
            task=task,
            support_toks=support_toks,
            support_lens=support_lens,
            support_masks=support_masks,
            support_labs=support_labs,
            query_toks=query_toks,
            query_lens=query_lens,
            query_masks=query_masks,
            query_labs=query_labs,
            task_type=task_type,
            rubric_embs=rubric_embs,
            prompt_embs=prompt_embs,
        )
        return output_dict

class MetaExamSolutions(Dataset):
    """
    Dataset of student solutions to programming exams for meta training.
    """
    def __init__(
            self,
            n_shots,
            n_queries,
            data_root=None,                  # root of the dataset
            exam_path=None,                  # path to exam files from root
            exam_rubric=None,                # path to rubrics from root
            exam_prompt=None,                # path to questions/prompts from root
            cache_path=None,                 # where to cache RoBerta embeddings of rubric names if roberta_rubric = True
            vocab=None,
            obfuscate_names=True,            # if True, replace variable names with <VAR:XX> and function names with <FUNC:XX>
            augment_by_names=True,           # if True, randomly swap variable names in training
            augment_by_rubric=True,          # if True, randomly shuffle programs with identical rubrics in training
            roberta_rubric=True,             # if True, map rubric text to RoBERTa encoding
            roberta_prompt=True,             # if True, map question text to RoBERTa encoding
            roberta_tokenize=False,          # if True, use RoBERTa tokenizer to tokenize
            roberta_device='cpu',            # used for cache-ing rubric / prompt embeddings
            roberta_config='microsoft/codebert-base',  # which pretrained RoBERTa to use for tokenizing
            cloze_tasks_factor=0,            # if >0, create tasks for masked token modeling
            execution_tasks_factor=0,        # if >0, create tasks for predicting execution output
            smlmt_tasks_factor=0,            # if >0, create tasks from SMLMT (https://arxiv.org/pdf/2009.08445)
            max_num_var=100,                 # if obfuscate_names = True, cap out at XX variable names
            max_num_func=10,                 # if obfuscate_names = True, cap out at XX function names
            max_seq_len=1000,                # maximum # of tokens of student code
            min_occ=1,                       # minimum number of occurences to keep a token in vocab (only used if not RoBERTa)
            conservative=True,               # if not conservative, keep more tasks (even unbalanced onces)
            train=True,
            train_frac=0.9,
            hold_out_split=True,             # if True, use unseen exam as test set; otherwise use unseen (exam, rubric) tuples
            hold_out_category='exam',        # exam | question
            enforce_binary=False,            # if True, all tasks are binary prediction problems
            fix_seed=True,
            pad_to_max_num_class=False,      # if True, all tasks are padded to the maximum # of classes
        ):
        super(MetaExamSolutions, self).__init__()
        cache_path = os.path.join(data_root, cache_path)

        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)

        if not train and not roberta_tokenize:
            assert vocab is not None

        if not train:
            augment_by_names = False

        if roberta_tokenize:
            # if we use roberta, turn off a bunch of other features
            obfuscate_names = False
            augment_by_names = False
            vocab = None

        if augment_by_names:
            obfuscate_names = True

        if enforce_binary:
            pad_to_max_num_class = False

        self.train = train
        self.train_frac = train_frac
        self.conservative = conservative
        self.vocab = vocab
        self.max_num_var = max_num_var
        self.max_num_func = max_num_func
        self.max_seq_len = max_seq_len
        self.min_occ = min_occ
        self.exam_path = os.path.join(data_root, exam_path)
        self.exam_rubric = os.path.join(data_root, exam_rubric)
        self.exam_prompt = os.path.join(data_root, exam_prompt)
        self.fix_seed = fix_seed
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.rs = np.random.RandomState(fix_seed)
        self.hold_out_split = hold_out_split
        self.hold_out_category = hold_out_category
        self.enforce_binary = enforce_binary
        self.roberta_tokenize = roberta_tokenize
        self.obfuscate_names = obfuscate_names
        self.pad_to_max_num_class = pad_to_max_num_class
        self.roberta_device = roberta_device

        print('loading exams...')
        programs, traces, indices, labels, tasks, \
        questions, task_splits, task_classes, task_stats, \
        rubrics, prompts, equivalences = self.load_exam_data()

        # NOTE: loading assignment data has been masked in public repo
        programs, traces, indices, labels, tasks, questions, task_splits, \
        task_classes, task_stats, init_task_types, rubrics, prompts, equivalences = \
            self.combine_data_sources(
                [programs], [traces], [indices], [labels], [tasks],
                [questions], [task_splits], [task_classes], [task_stats],
                [rubrics], [prompts], [equivalences])
        self.task_classes = task_classes
        self.task_stats = task_stats

        if hold_out_split:
            tasks_in_split = np.array([k for k, v in task_splits.items() if v != train])
            split_indices = np.in1d(tasks, tasks_in_split)
            split_indices = np.where(split_indices)[0]
        else:
            # meta tasks train test split randomly based on task.
            split_indices = self.train_test_split_by_question(
                questions, train=train, train_frac=train_frac)

        # indices : remaining indices in the current split
        indices = [indices[i] for i in split_indices]
        labels = [labels[i] for i in split_indices]
        tasks = [tasks[i] for i in split_indices]
        init_task_types = [init_task_types[t] for t in np.unique(tasks)]
        rubrics = [rubrics[t] for t in np.unique(tasks)]
        prompts = [prompts[t] for t in np.unique(tasks)]

        if roberta_rubric:
            rubric_bert_cache = self.roberta_embed_rubrics(
                rubrics, cache_path, key='human_label')
            prompt_bert_cache = self.roberta_embed_prompts(
                prompts, cache_path)
        else:
            rubric_bert_cache = None
            prompt_bert_cache = None

        # remap indices to be contiguous and just grab relevant programs.
        unique_indices = np.unique(indices).tolist()
        index_mapping = dict(zip(unique_indices, range(len(unique_indices))))
        indices = [index_mapping[index] for index in indices]
        programs = [programs[i] for i in unique_indices]
        traces = [traces[i] for i in unique_indices]

        # this is to find equivalent "programs" semantically
        equivalences = remap_equivalences(equivalences, index_mapping)
        equivalences = [equivalences[i] for i in range(len(unique_indices))]

        if obfuscate_names:
            print('minifying variable and function names...')
            minifier_maps = self.process_variables(traces, max_num_var, max_num_func)
        else:
            minifier_maps = [None for _ in range(len(traces))]

        if not roberta_tokenize:
            # get the train or test split for programs and traces
            if train and self.vocab is None:
                print('creating vocab...')
                self.vocab = self.create_vocab(
                    programs,
                    traces,
                    minifier_maps,
                    min_occ=min_occ,
                    max_num_var=max_num_var,
                    max_num_func=max_num_func,
                    obfuscate_names=obfuscate_names,
                )
                print(f'built vocab with {len(self.vocab["w2i"])} tokens...')

            self.w2i, self.i2w = self.vocab['w2i'], self.vocab['i2w']
            self.pad_index = self.w2i[PAD_TOKEN]
            self.vocab_size = len(self.w2i)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(roberta_config)
            self.vocab_size = self.tokenizer.vocab_size
            self.pad_index = self.tokenizer.pad_token_id

        if obfuscate_names:
            # save some stuff that will be used for augmentations in __getitem__
            func_tokens = [f'<FUNC:{i}>' for i in range(max_num_func)] + ['<FUNC:unseen>']
            var_tokens = [f'<VAR:{i}>' for i in range(max_num_var)] + ['<VAR:unseen>']
            self.func_tokens = sorted(list(set([self.w2i[tok] for tok in func_tokens])))
            self.var_tokens = sorted(list(set([self.w2i[tok] for tok in var_tokens])))
        else:
            self.func_tokens, self.var_tokens = None, None

        print('processing entries with vocab...')
        token_seqs, token_lens = self.process_programs(
            programs,
            traces,
            minifier_maps,
            obfuscate_names=obfuscate_names,
            roberta_tokenize=roberta_tokenize)

        print('constructing meta info...')
        indices_by_task, labels_by_task = self.prepare_meta(indices, tasks, labels)

        num_tasks = len(indices_by_task)
        num_cloze_tasks = int(cloze_tasks_factor * num_tasks)
        num_execution_tasks = int(execution_tasks_factor * num_tasks)
        num_smlmt_tasks = int(smlmt_tasks_factor * num_tasks)

        assert num_tasks == len(init_task_types)
        task_types = init_task_types + \
            [(max(init_task_types) + 1) for _ in range(num_cloze_tasks)] + \
            [(max(init_task_types) + 2) for _ in range(num_execution_tasks)] + \
            [(max(init_task_types) + 3) for _ in range(num_smlmt_tasks)]

        # default parameters if we aren't using cloze and execution tasks
        self.cloze_task_type = np.inf
        self.execution_task_type = np.inf
        self.smlmt_task_type = np.inf

        if num_cloze_tasks > 0:
            """
            Auxiliary Task #1:

            Cloze tasks mask out 2 different tokens across a set of programs
            and ask the model to predict which of the two tokens belongs in
            each spot. We ignore variable and function names.
            """
            cloze_mappings = self.init_cloze_task(token_seqs)
            print('constructing cloze tasks...')
            cloze_indices_by_task, cloze_labels_by_task, \
            _cloze_masks_by_task, cloze_rubrics_by_task = \
                self.prepare_cloze(
                    indices, cloze_mappings,
                    num_tasks=num_cloze_tasks,
                    n_way=2,
                    n_shots=n_shots,
                    n_queries=n_queries)
            cloze_masks_by_task = {}
            max_task_id = max(indices_by_task.keys()) + 1
            for t in range(len(cloze_indices_by_task)):
                indices_by_task[max_task_id + t] = cloze_indices_by_task[t]
                labels_by_task[max_task_id + t] = cloze_labels_by_task[t]
                cloze_masks_by_task[max_task_id + t] = _cloze_masks_by_task[t]
                rubrics.append(cloze_rubrics_by_task[t])
                prompts.append('None')
                task_classes[max_task_id + t] = 2
            self.cloze_task_type = max(init_task_types) + 1
            self.cloze_masks_by_task = cloze_masks_by_task

        if num_execution_tasks > 0:
            """
            Auxiliary Task #2:

            Execution tasks ask us to predict the output of running the program.
            We design a task by randomly choosing two different possible outputs.
            These include error messages and outputs.
            """
            # update the number of tasks
            print('constructing execution tasks...')
            executions, execution_vocab = self.init_execution_task(programs)
            exec_indices_by_task, exec_labels_by_task, exec_rubrics_by_task = \
                self.prepare_execution_task(
                    indices, executions, execution_vocab,
                    num_tasks=num_execution_tasks,
                    n_way=2,
                    n_shots=n_shots,
                    n_queries=n_queries)
            max_task_id = max(indices_by_task.keys()) + 1
            for t in range(len(exec_indices_by_task)):
                indices_by_task[max_task_id + t] = exec_indices_by_task[t]
                labels_by_task[max_task_id + t] = exec_labels_by_task[t]
                rubrics.append(exec_rubrics_by_task[t])
                prompts.append('None')
                task_classes[max_task_id + t] = 2
            ii = 2 if num_cloze_tasks > 0 else 1
            self.execution_task_type = max(init_task_types) + ii
            self.execution_vocab = execution_vocab

        if num_smlmt_tasks > 0:
            """
            Auxiliary Task #3:

            Cloze tasks mask out 2 different tokens across a set of programs
            and ask the model to predict which of the two tokens belongs in
            each spot. We treat all tokens equally (don't do anything custom
            to programming code: https://arxiv.org/pdf/2009.08445.pdf).
            """
            smlmt_mappings = self.init_smlmt_task(token_seqs)
            print('constructing smlmt tasks...')
            smlmt_indices_by_task, smlmt_labels_by_task, \
            _smlmt_masks_by_task, smlmt_rubrics_by_task = \
                self.prepare_smlmt(
                    indices, smlmt_mappings,
                    num_tasks=num_smlmt_tasks,
                    n_way=2,
                    n_shots=n_shots,
                    n_queries=n_queries)
            smlmt_masks_by_task = {}
            max_task_id = max(indices_by_task.keys()) + 1
            for t in range(len(smlmt_indices_by_task)):
                indices_by_task[max_task_id + t] = smlmt_indices_by_task[t]
                labels_by_task[max_task_id + t] = smlmt_labels_by_task[t]
                smlmt_masks_by_task[max_task_id + t] = _smlmt_masks_by_task[t]
                rubrics.append(smlmt_rubrics_by_task[t])
                prompts.append('None')
                task_classes[max_task_id + t] = 2
            self.smlmt_task_type = max(init_task_types) + 1
            self.smlmt_masks_by_task = smlmt_masks_by_task

        print('done')
        self.programs = programs
        self.traces = traces
        self.rubrics = rubrics
        self.prompts = prompts
        self.rubric_bert_cache = rubric_bert_cache
        self.prompt_bert_cache = prompt_bert_cache
        self.roberta_rubric = roberta_rubric
        self.roberta_prompt = roberta_prompt
        self.roberta_config = roberta_config
        if train:
            self.equivalences = equivalences
        else:
            # we should not be able to use this is validation
            self.equivalences = None
        self.token_seqs = token_seqs
        self.token_lens = token_lens
        self.max_index = max(indices)
        self.indices_by_task = indices_by_task
        self.labels_by_task = labels_by_task
        self.minifier_maps = minifier_maps
        self.task_ids = [int(x) for x in sorted(list(indices_by_task.keys()))]
        self.augment_by_names = augment_by_names
        self.obfuscate_names = obfuscate_names
        self.augment_by_rubric = augment_by_rubric
        self.cloze_tasks_factor = cloze_tasks_factor
        self.execution_tasks_factor = execution_tasks_factor
        self.smlmt_tasks_factor = smlmt_tasks_factor
        self.task_types = task_types
        self.num_task_types = len(set(task_types))
        self.task_classes = task_classes
        self.max_num_classes = max(self.task_classes.values())

    def load_exam_data(self):
        df = pd.read_csv(self.exam_path)
        key_df = pd.read_csv(self.exam_rubric)
        question_df = pd.read_csv(self.exam_prompt)

        key_rubrics = list(key_df['rubric'])
        key_rubrics = [json.loads(r) for r in key_rubrics]
        key_rubrics = dict(zip(list(key_df['id']), key_rubrics))

        # merge with questions dataframe
        df = pd.merge(df, question_df, how='left', on=['examId', 'questionId'])

        programs = np.asarray(df['answer'])
        rubrics = np.asarray(df['gradeData'])
        prompts = np.asarray(df['questionText'])
        scores = np.asarray(df['score'])

        courses = list(np.unique(df['examId']))

        # id isn't work bc its unique
        tasks = string_concat(
            np.asarray(df['examId']),
            np.asarray(df['questionId']).astype(str))
        ids = np.asarray(df['id'])

        if self.hold_out_category == 'exam':
            # reserve some exams for test split
            num_test_courses = max(int((1 - self.train_frac) * len(courses)), 1)
            test_courses = courses[-num_test_courses:]
            is_test = np.asarray(df['examId'].isin(test_courses))
        elif self.hold_out_category == 'question':
            # reserve some questions for test split
            questions = list(set(tasks))
            num_test_questions = max(int((1 - self.train_frac) * len(questions)), 1)
            self.rs.shuffle(questions)
            questions = list(questions)
            test_questions = questions[-num_test_questions:]
            is_test = np.in1d(tasks, test_questions)
        else:
            raise Exception(f'Category {self.hold_out_category} not supported.')

        # remove missing tasks
        indices = ~np.array([isinstance(t, float) for t in tasks])
        programs, rubrics, prompts, scores, tasks, ids, is_test = (
            programs[indices], rubrics[indices], prompts[indices], scores[indices],
            tasks[indices], ids[indices], is_test[indices])
        # remove missing rubrics
        indices = ~np.array([isinstance(r, float) for r in rubrics])
        programs, rubrics, prompts, scores, tasks, ids, is_test = (
            programs[indices], rubrics[indices], prompts[indices], scores[indices],
            tasks[indices], ids[indices], is_test[indices])

        # compile programs and remove those with bad compilations
        traces, indices = [], []
        print('compiling programs...')
        for i in tqdm(range(len(programs))):
            trace_i = compile_program(programs[i])
            if trace_i is not None:
                traces.append(trace_i)
                indices.append(i)
        traces = np.array(traces)

        # subset the other objects with indices
        indices = np.array(indices)
        programs, rubrics, prompts, scores, tasks, ids, is_test = (
            programs[indices], rubrics[indices], prompts[indices], scores[indices],
            tasks[indices], ids[indices], is_test[indices],
        )

        print('parsing rubrics...')
        # replace rubrics with string format with key_rubrics
        new_rubrics = []
        for task, rubric, task_id in zip(tasks, rubrics, ids):
            rubric = json.loads(rubric)

            if task in key_rubrics:
                new_rubric = {}
                rubric_map = key_rubrics[task]
                for key in rubric:
                    if key in rubric_map:
                        new_key = {
                            'name': task,
                            'id': task_id,
                            'human_label': rubric_map[key],
                            'abbrev_label': key,
                        }
                        new_key = json.dumps(new_key)
                        new_rubric[new_key] = rubric[key]
                    else:
                        new_key = {
                            'name': task,
                            'id': task_id,
                            'human_label': key,
                            'abbrev_label': key,
                        }
                        new_key = json.dumps(new_key)
                        new_rubric[new_key] = rubric[key]
                new_rubrics.append(new_rubric)
            else:
                new_rubric = {}
                for key in rubric:
                    new_key = {
                        'name': task,
                        'id': task_id,
                        'human_label': key,
                        'abbrev_label': key,
                    }
                    new_key = json.dumps(new_key)
                    new_rubric[new_key] = rubric[key]
                new_rubrics.append(new_rubric)

        # replace rubric with new_rubric
        rubrics = np.asarray(new_rubrics)

        # remove tasks where everyone got the same score
        # remove tasks with less than 50 responses. These tasks
        # are probably not useful for learning

        # if not conservative, don't remove anything and
        # keep trivial tasks with one label.
        if self.conservative:
            bad_tasks = []
            for ta in np.unique(tasks):
                if len(np.unique(scores[tasks == ta])) == 1:
                    bad_tasks.append(ta)
                elif len(scores[tasks == ta]) < 50:
                    bad_tasks.append(ta)
            bad_tasks = np.array(bad_tasks)
            indices = ~np.in1d(tasks, bad_tasks)
            programs, traces, rubrics, prompts, scores, tasks, is_test, = (
                programs[indices], traces[indices], rubrics[indices],
                prompts[indices], scores[indices], tasks[indices],
                is_test[indices],
            )

        # build equivalence maps from index to a set of indices
        # these are programs with the exact same output
        equivalence_maps = build_many_equivalences(tasks, rubrics)

        # map tasks to integers
        unique_tasks = sorted(set(tasks))
        task_mapping = dict(zip(unique_tasks, range(len(unique_tasks))))
        tasks = np.array([task_mapping[t] for t in tasks])

        # we use indices to track programs
        # v0: we worked directly on program strings but this quickly
        #     got expensive as we duplicate programs excessively.
        # v1: we work on indices and look up programs. Main advantage
        #     is that we only have to compile programs once.
        indices = np.arange(len(programs))

        print('building tasks...')
        # a task is now defined by several labels, we can duplicate each to new tasks
        indices, labels, tasks, questions, task_splits, \
        task_classes, task_stats, rubric_maps, prompt_maps = \
            construct_tasks_by_rubric(
                indices, rubrics, prompts, tasks, is_test)

        if self.conservative:
            print('removing trivial classes...')
            indices, labels, tasks, questions, task_splits, \
            task_classes, task_stats, rubric_maps, prompt_maps = \
                remove_small_classes(
                    indices, labels, tasks, questions, task_splits, task_classes,
                    task_stats, rubric_maps, prompt_maps,
                    min_freq=self.n_shots + self.n_queries)

        if self.enforce_binary:
            print('collapsing tasks to binary...')
            if self.conservative:
                indices, labels, tasks, questions, task_splits, \
                task_classes, task_stats, rubric_maps, prompt_maps = \
                    make_binary_tasks(
                        indices, labels, tasks, questions, task_splits, task_classes,
                        task_stats, rubric_maps, prompt_maps)
            else:
                indices, labels, tasks, questions, task_splits, \
                task_classes, task_stats, rubric_maps, prompt_maps = \
                    make_binary_tasks_liberally(
                        indices, labels, tasks, questions, task_splits, task_classes,
                        task_stats, rubric_maps, prompt_maps)

        return programs, traces, indices, labels, tasks, questions, task_splits, \
               task_classes, task_stats, rubric_maps, prompt_maps, equivalence_maps

    def create_vocab(
            self,
            programs,
            traces,
            minifier_maps,
            min_occ=1,
            max_num_var=100,
            max_num_func=10,
            obfuscate_names=False,
        ):
        w2c = OrderedCounter()
        w2i, i2w = dict(), dict()

        special_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, MASK_TOKEN]

        if obfuscate_names:
            # no need to do this if we are not obfuscating
            func_tokens = [f'<FUNC:{i}>' for i in range(max_num_func)] + ['<FUNC:unseen>']
            var_tokens = [f'<VAR:{i}>' for i in range(max_num_var)] + ['<VAR:unseen>']
            special_tokens = special_tokens + func_tokens + var_tokens

        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        pbar = tqdm(total=len(programs))
        for program, trace, minifier_map in zip(programs, traces, minifier_maps):
            program = clean_program(program)
            tokens = self.tokenize_program(
                program,
                trace,
                minifier_map=minifier_map,
                obfuscate_names=obfuscate_names,
            )
            w2c.update(tokens)
            pbar.update()
        pbar.close()

        for w, c in w2c.items():
            if c >= min_occ:
                if w not in w2i:  # make sure it's not there already
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)
        vocab = dict(w2i=w2i, i2w=i2w, w2c=w2c)
        return vocab

    def tokenize_program(self, program, trace, minifier_map=None, obfuscate_names=False):
        tokens = []
        for obj in trace:
            if obj.type == 5:
                tokens.append(ENTER_SCOPE_TOKEN)
            elif obj.type == 6:
                tokens.append(EXIT_SCOPE_TOKEN)
            elif obj.type == 4:
                tokens.append(NEWLINE_TOKEN)
            elif obj.type == 1 and not obfuscate_names:
                # enter here if variable name AND we are not obfuscating
                token = obj.string
                token = camel_to_snake(token)
                parts = token.split('_')
                if len(parts) > 1:
                    seps = ['_' for _ in range(len(parts))]
                    comps = parts + seps
                    comps[0::2] = parts
                    comps[1::2] = seps
                    comps = comps[:-1]  # remove last _
                    tokens.extend(comps)
                else:
                    tokens.append(token)
            # these tokens are not ones we want to keep
            elif obj.type not in [3, 55, 57, 62, 60, 0]:
                token = obj.string
                if obfuscate_names and (token in minifier_map):
                    token = minifier_map[token]
                tokens.append(token)
        return tokens

    def build_attention_masks(self, lengths):
        batch_size = len(lengths)
        mask = np.zeros((batch_size, self.max_seq_len))
        for i in range(batch_size):
            mask[i, :lengths[i]] = 1
        return torch.from_numpy(mask).long()

    def process_programs(
            self,
            programs,
            traces,
            minifier_maps,
            obfuscate_names=False,
            roberta_tokenize=False,
        ):
        token_seqs, token_lengths = [], []

        for i in tqdm(range(len(programs))):
            program, trace, minifier_map = programs[i], traces[i], minifier_maps[i]
            program = clean_program(program)

            if roberta_tokenize:
                assert self.max_seq_len == 512
                tokenizer_outputs = self.tokenizer(
                    program,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_seq_len,
                    return_length=True,
                    pad_to_max_length=True)

                tokens = tokenizer_outputs['input_ids']
                token_length = tokenizer_outputs['length']
            else:
                tokens = self.tokenize_program(
                    program,
                    trace,
                    minifier_map=minifier_map,
                    obfuscate_names=obfuscate_names)

                tokens = [SOS_TOKEN] + tokens[:self.max_seq_len - 2] + [EOS_TOKEN]
                token_length = len(tokens)

                tokens.extend([PAD_TOKEN] * (self.max_seq_len - token_length))

                new_tokens = []
                for token in tokens:
                    if token in self.w2i:
                        new_token = self.w2i[token]
                    else:
                        new_token = self.w2i[UNK_TOKEN]
                    new_tokens.append(new_token)
                tokens = new_tokens

            token_seqs.append(tokens)
            token_lengths.append(token_length)

        return token_seqs, token_lengths

    def combine_data_sources(
            self,
            programs_list,
            traces_list,
            indices_list,
            labels_list,
            tasks_list,
            questions_list,
            task_splits_list,
            task_classes_list,
            task_stats_list,
            rubrics_list,
            prompts_list,
            equivalences_list,
        ):
        programs = []
        traces = []
        indices = []
        labels = []
        tasks = []
        questions = []
        task_splits = {}
        task_classes = {}
        task_stats = []
        task_types = []
        rubrics = {}
        prompts = {}
        equivalences = {}

        max_index = 0
        max_task = 0
        max_question = 0
        n = len(programs_list)
        for i in range(n):
            programs.extend(programs_list[i])
            traces.extend(traces_list[i])
            indices_i = indices_list[i]
            indices_i = [d + max_index for d in indices_i]
            indices.extend(indices_i)
            labels.extend(labels_list[i])
            tasks_i = tasks_list[i]
            tasks_i = [t + max_task for t in tasks_i]
            tasks.extend(tasks_i)
            questions_i = questions_list[i]
            questions_i = [q + max_question for q in questions_i]
            questions.extend(questions_i)
            for t, rubs in rubrics_list[i].items():
                assert t + max_task not in rubrics
                rubrics[t + max_task] = rubs
            for t, pmts in prompts_list[i].items():
                assert t + max_task not in prompts
                prompts[t + max_task] = pmts
            for t, splits in task_splits_list[i].items():
                assert t + max_task not in task_splits
                task_splits[t + max_task] = splits
            for t, classes in task_classes_list[i].items():
                assert t + max_task not in task_classes
                task_classes[t + max_task] = classes
            task_stats.extend(task_stats_list[i])
            task_types.extend([i for _ in range(len(task_stats_list[i]))])
            for ix, eqs in equivalences_list[i].items():
                assert ix + max_index not in equivalences
                equivalences[ix + max_index] = eqs
            # increment max tasks
            max_task = int(max(tasks_i) + 1)
            max_index = int(max(indices_i) + 1)
            max_question = int(max(questions_i) + 1)

        return programs, traces, indices, labels, tasks, questions, task_splits, \
               task_classes, task_stats, task_types, rubrics, prompts, equivalences

    def roberta_embed_rubrics(self, rubric_list, cache_path, key='human_label'):
        """
        rubric_list maps a task to a rubric name and list of value names
        """
        print('Computing rubric BERT embeddings...')

        edited_cache = False
        cache_file = os.path.join(
            cache_path,
            f'sentence_bert_rubric_cache_{self.n_shots}shots_{self.n_queries}queries.pth.tar',
        )
        if os.path.exists(cache_file):
            roberta_cache = torch.load(cache_file)
        else:
            roberta_cache = {}

        if 'None' not in roberta_cache:
            roberta_cache['None'] = torch.zeros(768)

        texts = []
        for task in range(len(rubric_list)):
            rubric = rubric_list[task]
            if isinstance(rubric, list):
                r = rubric[0]  # they are all the same
                r = json.loads(r)[key]
                texts.append(r)
            elif isinstance(rubric, str):
                r = json.loads(rubric)[key]
                texts.append(r)
            else:
                raise Exception('Unsupported rubric type.')
        texts = list(set(texts))

        texts_to_run = []  # these are the things we need to run

        for text in texts:
            text = str(text)
            if text in roberta_cache:
                embs = roberta_cache[text]
            elif text == 'None':  # replace this with just zeros
                embs = torch.zeros(768)
                roberta_cache[text] = embs
                edited_cache = True
            else:
                texts_to_run.append(text)
                edited_cache = True

        if len(texts_to_run) > 0:
            assert edited_cache
            bert = SentenceBERT(
                version='bert-base-nli-stsb-mean-tokens',
                device=self.roberta_device,
            )
            embs = bert(texts_to_run, batch_size=32, show_progress_bar=True)
            embs = embs.detach().cpu()

            for i in range(len(texts_to_run)):
                roberta_cache[texts_to_run[i]] = embs[i]

        if edited_cache:
            print('Made edits to cache... saving...')
            torch.save(roberta_cache, cache_file)

        return roberta_cache

    def roberta_embed_prompts(self, prompt_list, cache_path):
        print('Computing prompt BERT embeddings...')

        edited_cache = False
        cache_file = os.path.join(
            cache_path,
            f'sentence_bert_prompt_cache_{self.n_shots}shots_{self.n_queries}queries.pth.tar',
        )
        if os.path.exists(cache_file):
            roberta_cache = torch.load(cache_file)
        else:
            roberta_cache = {}

        if 'None' not in roberta_cache:
            roberta_cache['None'] = torch.zeros(768)

        texts_to_run = []  # these are the things we need to run
        hashs_to_run = []

        for prompt in prompt_list:
            prompt = str(prompt)
            prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            if prompt_hash in roberta_cache:
                embs = roberta_cache[prompt_hash]
            elif prompt == 'None':
                embs = torch.zeros(768)
                roberta_cache[prompt_hash] = embs
                edited_cache = True
            else:
                texts_to_run.append(prompt)
                hashs_to_run.append(prompt_hash)
                edited_cache = True

        if len(texts_to_run) > 0:
            assert edited_cache
            assert len(texts_to_run) == len(hashs_to_run)
            bert = SentenceBERT(
                version='bert-base-nli-stsb-mean-tokens',
                device=self.roberta_device,
            )
            embs = bert(texts_to_run, batch_size=32, show_progress_bar=True)
            embs = embs.detach().cpu()

            for i in range(len(texts_to_run)):
                roberta_cache[hashs_to_run[i]] = embs[i]

        if edited_cache:
            print('Made edits to cache... saving...')
            torch.save(roberta_cache, cache_file)

        return roberta_cache

    def train_test_split_by_question(self, questions, train=True, train_frac=0.9):
        """
        Split tasks into train and test splits but do so preserving that all the
        tasks from one question are put together. Otherwise, it is cheating a bit
        since the same programs where seen in training.
        """
        unique_questions = np.sort(np.unique(questions))
        num_questions = len(unique_questions)

        num_train = int(num_questions * train_frac)
        # important to fix the random seed here we get same split every call
        train_questions = self.rs.choice(unique_questions, num_train, replace=False)
        train_questions = np.sort(train_questions)

        if train:
            split_indices = np.in1d(questions, train_questions)
        else:
            split_indices = ~np.in1d(questions, train_questions)

        split_indices = np.where(split_indices)[0]
        return split_indices

    def process_variables(self, traces, max_num_var=100, max_num_func=10):
        """
        For each program, find which tokens can be considered variables.
        replace variables with <VAR:NUM> or <FUNC:NUM> so that we can
        minimize extra variables. Also ignore special python keywords.
        Also, if there are more than <max_num_var>, we just cap with
        <FUNC:NEW> or <VAR:NEW>.
        """
        bad_compilation = 0
        num_programs = len(traces)

        # each mapping goes from raw variable string to a new name
        variable_maps = [dict() for _ in range(num_programs)]

        for i in tqdm(range(num_programs)):
            trace = traces[i]
            num_variables = 0
            num_functions = 0

            if trace is None or len(trace) == 0:
                bad_compilation += 1
                continue

            for k, obj in enumerate(trace):
                token = obj.string
                is_reserved = token in PYTHON_KEYWORDS
                # parsing will consider python reserved keywords
                # "NAME" objects as well. We need to avoid this as
                # best we can.
                is_var = (obj.type == 1) and not is_reserved

                if is_var:
                    if token in variable_maps[i]:
                        # if we've already seen this, nothing to do
                        continue

                    is_func = False
                    if k > 0:
                        if trace[k-1].string == 'def':
                            is_func = True
                    if is_func:
                        if num_functions >= max_num_func:
                            variable_maps[i][token] = '<FUNC:unseen>'
                        else:
                            variable_maps[i][token] = f'<FUNC:{num_functions}>'
                            num_functions += 1

                    else:
                        if num_variables >= max_num_var:
                            variable_maps[i][token] = '<VAR:unseen>'
                        else:
                            variable_maps[i][token] = f'<VAR:{num_variables}>'
                            num_variables += 1

        return variable_maps

    def extract_variables(self, traces, vocab):
        # for each program, find which tokens can be considered variables
        # learn a map from program index -> vocab list. Also make a big list
        # of merged vocabulary tokens.
        bad_compilation = 0
        num_programs = len(traces)
        variables = [[] for _ in range(num_programs)]
        all_variables = []

        for i in tqdm(range(num_programs)):
            trace = traces[i]

            if trace is None or len(trace) == 0:
                bad_compilation += 1
                continue

            for obj in trace:
                token = obj.string
                is_var = obj.type == 1

                if is_var:
                    token_index = vocab['w2i'].get(token, vocab['w2i'][UNK_TOKEN])
                    variables[i].append(token_index)
                    all_variables.append(token_index)

            variables[i] = sorted(list(set(variables[i])))

        all_variables = sorted(list(set(all_variables)))
        print(f'\t{bad_compilation} programs unable to compile.')
        return variables, all_variables

    def prepare_meta(self, indices, tasks, labels):
        indices_by_task = defaultdict(lambda: [])
        labels_by_task  = defaultdict(lambda: [])

        for i in range(len(indices)):
            indices_by_task[tasks[i]].append(indices[i])
            labels_by_task[tasks[i]].append(labels[i])

        return indices_by_task, labels_by_task

    # --- task augmentation: cloze ---

    def init_cloze_task(self, token_seqs):
        """
        See https://arxiv.org/pdf/2009.08445.pdf. We create meta
        learning tasks by choosing N random words and finding all
        sentences that contain those words. To do this, we will
        need to map unique tokens to all sentences that contain it.
        """
        token_id_to_indices = defaultdict(lambda: [])
        for i, token_seq in enumerate(token_seqs):
            for token in token_seq:
                token_id_to_indices[token].append(i)
        token_id_to_indices = dict(token_id_to_indices)
        return token_id_to_indices

    def prepare_cloze(self, indices, cloze_mappings,
                      num_tasks, n_way, n_shots, n_queries):
        indices_by_task = defaultdict(lambda: [])
        masks_by_task = defaultdict(lambda: [])
        labels_by_task = defaultdict(lambda: [])
        rubrics_by_task = defaultdict(lambda: [])

        if self.roberta_tokenize:
            reserved_tokens = [
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.unk_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.mask_token_id,
            ]
        else:
            # we do not want to mask this trivial tokens
            reserved_tokens = [
                self.w2i[SOS_TOKEN],
                self.w2i[EOS_TOKEN],
                self.w2i[PAD_TOKEN],
                self.w2i[UNK_TOKEN],
                self.w2i[MASK_TOKEN],
            ]
        if self.obfuscate_names:
            # we do not want to maks function or variable
            # tokens because we are swapping them a lot
            reserved_tokens += self.func_tokens
            reserved_tokens += self.var_tokens

        vocab = sorted(list(set(list(cloze_mappings.keys())) -
                            set(reserved_tokens)))
        # pick only from the vocab that has enough members
        valid_vocab = []
        for key in vocab:
            if len(cloze_mappings[key]) >= (n_shots + n_queries):
                valid_vocab.append(key)
        vocab = valid_vocab

        for t in tqdm(range(num_tasks)):
            task_tokens = self.rs.choice(vocab, size=n_way, replace=False)
            task_tokens = task_tokens.tolist()
            task_indices = []
            task_labels = []
            for l, token in enumerate(task_tokens):
                indices = cloze_mappings[token]
                indices = self.rs.choice(
                    indices,
                    size=n_shots+n_queries,
                    replace=False,
                )
                labels = [l for _ in range(len(indices))]
                task_labels.extend(labels)
                task_indices.extend(list(indices))
            indices_by_task[t] = task_indices
            labels_by_task[t] = task_labels
            masks_by_task[t] = task_tokens
            task_rubric = json.dumps({'name': 'cloze', 'human_label': 'None'})
            rubrics_by_task[t] = task_rubric

        return indices_by_task, labels_by_task, masks_by_task, rubrics_by_task

    # --- task augmentation: smlmt ---

    def init_smlmt_task(self, token_seqs):
        return self.init_cloze_task(token_seqs)

    def prepare_smlmt(self, indices, smlmt_mappings,
                      num_tasks, n_way, n_shots, n_queries):
        indices_by_task = defaultdict(lambda: [])
        masks_by_task = defaultdict(lambda: [])
        labels_by_task = defaultdict(lambda: [])
        rubrics_by_task = defaultdict(lambda: [])

        vocab = sorted(list(smlmt_mappings.keys()))

        # pick only from the vocab that has enough members
        valid_vocab = []
        for key in vocab:
            if len(smlmt_mappings[key]) >= (n_shots + n_queries):
                valid_vocab.append(key)
        vocab = valid_vocab

        for t in tqdm(range(num_tasks)):
            task_tokens = self.rs.choice(vocab, size=n_way, replace=False)
            task_tokens = task_tokens.tolist()
            task_indices = []
            task_labels = []
            for l, token in enumerate(task_tokens):
                indices = smlmt_mappings[token]
                indices = self.rs.choice(
                    indices,
                    size=n_shots+n_queries,
                    replace=False,
                )
                labels = [l for _ in range(len(indices))]
                task_labels.extend(labels)
                task_indices.extend(list(indices))
            indices_by_task[t] = task_indices
            labels_by_task[t] = task_labels
            masks_by_task[t] = task_tokens
            task_rubric = json.dumps({'name': 'smlmt', 'human_label': 'None'})
            rubrics_by_task[t] = task_rubric

        return indices_by_task, labels_by_task, masks_by_task, rubrics_by_task

    # --- task augmentation: execution ---

    def init_execution_task(self, programs):
        """
        Run all the programs and either get error output or output.
        """
        def execute_program(program):
            try:
                compile(program, '<string>', 'exec')
                output = 'success'
            except Exception as e:
                output = str(e.msg)
                if 'Did you mean print' in output:
                    output = 'print syntax error'
                elif 'invalid character in identifier' in output:
                    output = 'invalid token'
            return output

        outputs = []
        for program in programs:
            output = execute_program(program)
            outputs.append(output)

        output_vocab = sorted(list(set(outputs)))
        return outputs, output_vocab

    def prepare_execution_task(self, indices, executions, execution_vocab,
                               num_tasks, n_way, n_shots, n_queries):
        indices_by_task = defaultdict(lambda: [])
        labels_by_task = defaultdict(lambda: [])
        rubrics_by_task = defaultdict(lambda: [])

        executions = [execution_vocab.index(exe) for exe in executions]
        executions = np.array(executions)

        exe_counts = Counter(executions)
        exe_classes = []
        for i in range(len(execution_vocab)):
            if exe_counts[i] >= (n_shots + n_queries):
                exe_classes.append(i)
        exe_classes = np.array(exe_classes)

        for t in tqdm(range(num_tasks)):
            task_classes = self.rs.choice(exe_classes, size=n_way, replace=False)
            task_indices = []
            task_labels = []
            for l, task_class in enumerate(task_classes):
                indices = np.where(executions == task_class)[0]
                indices = self.rs.choice(
                    indices,
                    size=n_shots+n_queries,
                    replace=False,
                )
                labels = [l for _ in range(len(indices))]
                task_labels.extend(labels)
                task_indices.extend(list(indices))
            indices_by_task[t] = task_indices
            labels_by_task[t] = task_labels
            task_rubric = json.dumps({'name': 'cloze', 'human_label': 'None'})
            rubrics_by_task[t] = task_rubric

        return indices_by_task, labels_by_task, rubrics_by_task

    def __len__(self):
        return len(self.task_ids)

    def __getitem__(self, index):
        task = int(self.task_ids[index])
        task_type = self.task_types[index]

        # --- handle grabbing the rubric embedding ---
        rubric_name = self.rubrics[index]
        if isinstance(rubric_name, list):
            rubric_name = list(set([json.loads(x)['human_label'] for x in rubric_name]))[0]
        elif isinstance(rubric_name, str):
            rubric_name = json.loads(rubric_name)['human_label']
        else:
            raise Exception('type not supported.')

        prompt_text = self.prompts[index]

        if self.roberta_rubric:
            rubric_bert_name = self.rubric_bert_cache[rubric_name]
        else:
            rubric_bert_name = torch.zeros(768)

        if self.roberta_prompt:
            prompt_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()
            prompt_bert_name = self.prompt_bert_cache[prompt_hash]
        else:
            prompt_bert_name = torch.zeros(768)

        # --- done ---

        support_toks, support_lens, support_labs = [], [], []
        query_toks, query_lens, query_labs = [], [], []

        indices = self.indices_by_task[task]
        toks = [self.token_seqs[i] for i in indices]
        lens = [self.token_lens[i] for i in indices]
        labs = np.array(self.labels_by_task[task])

        # --- handle cloze tasks ---
        if self.cloze_tasks_factor > 0 and task_type == self.cloze_task_type:
            cloze_masks = self.cloze_masks_by_task[task]
            if self.roberta_tokenize:
                mask_index = self.tokenizer.mask_token_id
            else:
                # in each program replaced the token with mask
                mask_index = self.w2i[MASK_TOKEN]
            new_toks = []
            for tok, lab in zip(toks, labs):
                new_tok = copy.deepcopy(np.array(tok))
                new_tok[new_tok == cloze_masks[lab]] = mask_index
                new_tok = new_tok.tolist()
                new_toks.append(new_tok)
            toks = new_toks

        # --- handle smlmt tasks ---
        if self.smlmt_tasks_factor > 0 and task_type == self.smlmt_task_type:
            smlmt_masks = self.smlmt_masks_by_task[task]
            if self.roberta_tokenize:
                mask_index = self.tokenizer.mask_token_id
            else:
                # in each program replaced the token with mask
                mask_index = self.w2i[MASK_TOKEN]
            new_toks = []
            for tok, lab in zip(toks, labs):
                new_tok = copy.deepcopy(np.array(tok))
                new_tok[new_tok == smlmt_masks[lab]] = mask_index
                new_tok = new_tok.tolist()
                new_toks.append(new_tok)
            toks = new_toks
        # --- done ---

        num_classes = self.task_classes[task]

        rubric_embs = []
        prompt_embs = []
        for cls in range(num_classes):
            valid_indices = np.where(labs == cls)[0]
            sample_with_replace = True if len(valid_indices) < self.n_shots else False
            support_ids_sampled = self.rs.choice(valid_indices, self.n_shots, sample_with_replace)
            support_toks_cls = []
            support_lens_cls = []
            for s in support_ids_sampled:
                if self.train:
                    toks_s = toks[s]
                    lens_s = lens[s]

                    # --- augmentations to make more training tasks ---
                    # don't shuffle entries if this is a cloze/execution/smlmt task.
                    is_aux = task_type in [
                        self.cloze_task_type,
                        self.execution_task_type,
                        self.smlmt_task_type,
                    ]
                    is_smlmt = task_type == self.smlmt_task_type
                    if self.augment_by_rubric and not is_aux:
                        toks_s, new_s = shuffle_entries_augmentation(
                            indices[s],
                            self.equivalences,
                            self.token_seqs,
                            p=0.5,
                        )
                        lens_s = self.token_lens[new_s]
                    # can't augment names if SMLMT task bc we might
                    # be try to predict those. We can with cloze task
                    # because we avoid names.
                    if self.augment_by_names and not is_smlmt:
                        toks_s = shuffle_names_augmentation(
                            toks_s,
                            self.vocab,
                            self.func_tokens,
                            self.var_tokens,
                            p=0.5,
                        )
                    # --- done ---
                    support_toks_cls.append(toks_s)
                    support_lens_cls.append(lens_s)
                else:
                    # do nothing. we used to replace unknown tokens with random ones
                    # but i think thats a bad idea
                    support_toks_cls.append(toks[s])
                    support_lens_cls.append(lens[s])
            support_toks.append(support_toks_cls)
            support_lens.append(support_lens_cls)
            support_labs.append([labs[s] for s in support_ids_sampled])

            # do the same for query set
            query_ids = np.setxor1d(valid_indices, support_ids_sampled)
            if len(query_ids) == 0:
                query_ids = valid_indices  # no choice but to resample
            query_replace = False if len(query_ids) > self.n_queries else True
            query_ids = self.rs.choice(query_ids, self.n_queries, query_replace)
            query_toks_cls = []
            query_lens_cls = []
            for s in query_ids:
                if self.train and self.augment_by_names:
                    toks_s = toks[s]
                    lens_s = lens[s]

                    # don't shuffle entries if this is a cloze/execution/smlmt task.
                    is_aux = task_type in [
                        self.cloze_task_type,
                        self.execution_task_type,
                        self.smlmt_task_type,
                    ]
                    is_smlmt = task_type == self.smlmt_task_type
                    if self.augment_by_rubric and not is_aux:
                        toks_s, new_s = shuffle_entries_augmentation(
                            indices[s],
                            self.equivalences,
                            self.token_seqs,
                            p=0.5,
                        )
                        lens_s = self.token_lens[new_s]
                    if self.augment_by_names and not is_smlmt:
                        toks_s = shuffle_names_augmentation(
                            toks_s,
                            self.vocab,
                            self.func_tokens,
                            self.var_tokens,
                            p=0.5,
                        )
                    query_toks_cls.append(toks_s)
                    query_lens_cls.append(len(toks_s))
                else:  # do nothing
                    query_toks_cls.append(toks[s])
                    query_lens_cls.append(lens[s])
            query_toks.append(query_toks_cls)
            query_lens.append(query_lens_cls)
            query_labs.append([labs[s] for s in query_ids])

            rubric_embs.append(rubric_bert_name)
            prompt_embs.append(prompt_bert_name)

        support_toks = np.array(support_toks)
        support_lens = np.array(support_lens)
        support_labs = np.array(support_labs)

        query_toks = np.array(query_toks)
        query_lens = np.array(query_lens)
        query_labs = np.array(query_labs)

        rubric_embs = torch.stack(rubric_embs)
        prompt_embs = torch.stack(prompt_embs)

        if self.pad_to_max_num_class:
            # we want everything to be of the same number of classes
            # so that can organize them as a minibatch. Each sentence
            # is going to be ['<s>', '</s>'] (length 2).
            num_class = support_toks.shape[0]
            if num_class < self.max_num_classes:
                support_fill_toks = np.zeros((self.max_num_classes - num_class,
                                              support_toks.shape[1],
                                              support_toks.shape[2]))
                support_fill_toks[:, :, 0] = self.w2i[SOS_TOKEN]
                support_fill_toks[:, :, 1] = self.w2i[EOS_TOKEN]
                support_fill_toks[:, :, 2:] = self.w2i[PAD_TOKEN]
                support_fill_lens = np.zeros((self.max_num_classes - num_class,
                                              support_lens.shape[1])) + 2
                # filler label is always 0.
                support_fill_labs = np.zeros((self.max_num_classes - num_class,
                                              support_labs.shape[1]))

                query_fill_toks = np.zeros((self.max_num_classes - num_class,
                                            query_toks.shape[1],
                                            query_toks.shape[2]))
                query_fill_toks[:, :, 0] = self.w2i[SOS_TOKEN]
                query_fill_toks[:, :, 1] = self.w2i[EOS_TOKEN]
                query_fill_toks[:, :, 2:] = self.w2i[PAD_TOKEN]
                query_fill_lens = np.zeros((self.max_num_classes - num_class,
                                            query_lens.shape[1])) + 2
                query_fill_labs = np.zeros((self.max_num_classes - num_class,
                                            query_labs.shape[1]))

                support_toks = np.concatenate([support_toks, support_fill_toks], axis=0)
                support_lens = np.concatenate([support_lens, support_fill_lens], axis=0)
                support_labs = np.concatenate([support_labs, support_fill_labs], axis=0)

                query_toks = np.concatenate([query_toks, query_fill_toks], axis=0)
                query_lens = np.concatenate([query_lens, query_fill_lens], axis=0)
                query_labs = np.concatenate([query_labs, query_fill_labs], axis=0)

        support_toks = torch.from_numpy(support_toks).long()
        support_lens = torch.from_numpy(support_lens).long()
        support_labs = torch.from_numpy(support_labs).long()
        support_masks = self.build_attention_masks(support_lens.view(-1))
        support_masks = support_masks.view(num_classes, self.n_shots, -1)

        query_toks = torch.from_numpy(query_toks).long()
        query_lens = torch.from_numpy(query_lens).long()
        query_labs = torch.from_numpy(query_labs).long()
        query_masks = self.build_attention_masks(query_lens.view(-1))
        query_masks = query_masks.view(num_classes, self.n_queries, -1)

        output_dict = dict(
            task=task,
            support_toks=support_toks,
            support_lens=support_lens,
            support_masks=support_masks,
            support_labs=support_labs,
            query_toks=query_toks,
            query_lens=query_lens,
            query_masks=query_masks,
            query_labs=query_labs,
            task_type=task_type,
            rubric_embs=rubric_embs,
            prompt_embs=prompt_embs,
        )
        return output_dict


class SupervisedExamSolutions(MetaExamSolutions):

    def __init__(
            self,
            task_index,
            n_shots,
            n_queries,
            data_root=None,                  # root of the dataset
            exam_path=None,
            exam_rubric=None,
            cache_path=None,                 # where to cache RoBerta embeddings of rubric names if roberta_rubric = True
            roberta_rubric=True,             # if True, map rubric text to RoBERTa encoding
            roberta_prompt=True,
            roberta_config='microsoft/codebert-base',  # which pretrained RoBERTa to use for tokenizing
            max_seq_len=1000,
            min_occ=1,
            train=True,
            meta_train=True,                 # choose from meta-train set
            hold_out_split=True,             # if True, use unseen exam as test set; otherwise use unseen (exam, rubric) tuples
            hold_out_category='exam',
            enforce_binary=False,            # if True, all tasks are binary prediction problems
            fix_seed=True,
            pad_to_max_num_class=False,      # if True, all tasks are padded to the maximum # of classes
        ):
        super().__init__(
            n_shots,
            n_queries,
            data_root=data_root,
            exam_path=exam_path,
            exam_rubric=exam_rubric,
            cache_path=cache_path,
            vocab=None,
            obfuscate_names=False,
            augment_by_names=False,
            augment_by_rubric=False,
            roberta_rubric=roberta_rubric,
            roberta_prompt=roberta_prompt,
            roberta_tokenize=True,
            roberta_config=roberta_config,
            cloze_tasks_factor=0,
            execution_tasks_factor=0,
            smlmt_tasks_factor=0,
            max_num_var=100,
            max_num_func=10,
            max_seq_len=max_seq_len,
            min_occ=min_occ,
            train=meta_train,  # always true!
            train_frac=0.9,
            hold_out_split=hold_out_split,
            hold_out_category=hold_out_category,
            enforce_binary=enforce_binary,
            fix_seed=fix_seed,
            pad_to_max_num_class=pad_to_max_num_class,
        )
        self.task_index = task_index
        self.data = self.prep_data(train=train)

    def prep_data(self, train=False):
        """
        NOTE: modeled after MetaExamSolutions(...).__getitem__()
        """
        task = int(self.task_ids[self.task_index])

        rubric_name = self.rubrics[self.task_index]
        if isinstance(rubric_name, list):
            rubric_name = list(set([json.loads(x)['human_label'] for x in rubric_name]))[0]
        elif isinstance(rubric_name, str):
            rubric_name = json.loads(rubric_name)['human_label']
        else:
            raise Exception('type not supported.')
        prompt_name = self.prompts[self.task_index]

        if self.roberta_rubric:
            rubric_bert_name = self.rubric_bert_cache[rubric_name]
        else:
            rubric_bert_name = torch.zeros(768)

        if self.roberta_prompt:
            prompt_hash = hashlib.sha256(prompt_name.encode('utf-8')).hexdigest()
            prompt_bert_name = self.prompt_bert_cache[prompt_hash]
        else:
            prompt_bert_name = torch.zeros(768)

        indices = self.indices_by_task[task]
        toks = [self.token_seqs[i] for i in indices]
        lens = [self.token_lens[i] for i in indices]
        labs = np.array(self.labels_by_task[task])

        num_classes = self.task_classes[task]
        train_indices, test_indices = [], []

        for cls in range(num_classes):
            valid_indices = np.where(labs == cls)[0]
            train_ids = self.rs.choice(valid_indices, self.n_shots, False)
            test_ids = np.setxor1d(valid_indices, train_ids)
            test_ids = self.rs.choice(test_ids, self.n_queries, False)
            train_ids = sorted(train_ids.tolist())
            test_ids = sorted(test_ids.tolist())
            train_indices.extend(train_ids)
            test_indices.extend(test_ids)

        if train:
            train_toks = [toks[i] for i in train_indices]
            train_lens = [lens[i] for i in train_indices]
            train_labs = [labs[i] for i in train_indices]
            train_masks = self.build_attention_masks(train_lens)
            data_dict = dict(
                tokens=train_toks,
                lengths=train_lens,
                labels=train_labs,
                masks=train_masks,
                rubric_bert_name=rubric_bert_name,
                prompt_bert_name=prompt_bert_name,
            )
        else:
            test_toks = [toks[i] for i in test_indices]
            test_lens = [lens[i] for i in test_indices]
            test_labs = [labs[i] for i in test_indices]
            test_masks = self.build_attention_masks(test_lens)
            data_dict = dict(
                tokens=test_toks,
                lengths=test_lens,
                labels=test_labs,
                masks=test_masks,
                rubric_bert_name=rubric_bert_name,
                prompt_bert_name=prompt_bert_name,
            )
        return data_dict

    def __getitem__(self, index):
        output_dict = dict(
            tokens=torch.LongTensor(self.data['tokens'][index]),
            lengths=self.data['lengths'][index],
            labels=self.data['labels'][index],
            masks=self.data['masks'][index].long(),
            rubric_embs=self.data['rubric_bert_name'].float(),
            prompt_embs=self.data['prompt_bert_name'].float(),
        )
        return output_dict

    def __len__(self):
        return len(self.data['tokens'])

# I THINK: returns a dictionary where for every answer, you get the indices of all the answers with the same exact rubric items
def build_many_equivalences(tasks, rubrics):
    utasks = list(set(tasks))

    def simplify_rubric(rubric):
        simple = {}
        for k, v in rubric.items():
            k = json.loads(k)
            simple[k['abbrev_label']] = v
        return simple

    def get_all_keys(rubrics):
        keys = []
        for r in rubrics:
            keys += list(r.keys())
        return list(set(keys))

    equivalences = {}
    for task in utasks:
        task_indices = tasks == task
        task_indices = np.where(task_indices)[0]
        task_rubrics = rubrics[task_indices]
        task_rubrics = [simplify_rubric(r) for r in task_rubrics]
        task_keys = get_all_keys(task_rubrics)
        task_rubrics = [[r.get(key, False) for key in task_keys] for r in task_rubrics]
        task_rubrics = np.array(task_rubrics)

        task_equivalences = build_equivalences(task_indices, task_rubrics)
        equivalences.update(task_equivalences)

    return equivalences


def build_equivalences(indices, rubrics):
    equivalences = {}
    for i, index in enumerate(indices):
        rows = np.where((rubrics == rubrics[i]).all(axis=1))[0]
        equivalences[index] = list(indices[rows])
    return equivalences


def make_rubrics_float(rubrics):
    num_col = rubrics.shape[1]
    new_rubrics = []
    for c in range(num_col):
        rubrics_c = rubrics[:, c]
        rubrics_c = [0 if str(x) == 'nan' else x for x in rubrics_c]
        keys_c = list(set(rubrics_c))
        maps_c = dict(zip(keys_c, range(len(keys_c))))
        new_rubrics_c = [maps_c[x] for x in rubrics_c]
        new_rubrics_c = np.array(new_rubrics_c)
        new_rubrics.append(new_rubrics_c)
    new_rubrics = np.array(new_rubrics).T
    return new_rubrics


def compile_program(program):
    program = clean_program(program)
    try:
        bprogram = bytes(program, 'utf-8')
        compilation = pythonlang.tokenize(io.BytesIO(bprogram).readline)
        compilation = [obj for obj in compilation]
    except:
        compilation = None

    return compilation


def clean_program(program):
    # strip, remove repeated newlines
    program = program.strip()
    program = re.sub('\n+', '\n', program)
    return program


# I THINK: Drops rubrics with generalDeduction or comments
# and then replaces the rubric with human-readable labels.
# It also returns rubric_map, a dictionary of human-readable rubric itens to all the JSON strings they replaced
def clean_exam_rubrics(rubrics, key='human_label'):
    new_rubrics = []
    rubric_map = defaultdict(lambda: [])
    for rubric in rubrics:
        if 'generalDeduction' in rubric:
            continue
        if 'comments' in rubric:
            continue
        new_rubric = {}
        for r in rubric:
            j = json.loads(r)
            j = j['human_label']
            new_rubric[j] = rubric[r]
            rubric_map[j].append(r)
        new_rubrics.append(new_rubric)
    return new_rubrics, rubric_map

# I THINK: Gets a list of all the keys in rubrics
def get_exam_rubric_keys(rubrics):
    all_keys = set()
    for rubric in rubrics:
        keys = list(rubric.keys())
        all_keys.update(keys)
    return sorted(list(all_keys))


def construct_tasks_by_rubric(indices, rubrics, prompts, tasks, is_test):
    new_indices, new_labels, new_tasks, new_questions = [], [], [], []
    task_cnt = 0
    task_to_split_mapping = {}
    task_to_class_mapping = {}
    task_to_stats_mapping = {}
    task_label_to_rubric = {}
    task_to_prompt = {}

    for ta in np.unique(tasks):
        task_indices = indices[tasks == ta]
        raw_task_rubrics = rubrics[tasks == ta]
        task_rubrics, task_rmap = clean_exam_rubrics(raw_task_rubrics)
        rubric_keys = get_exam_rubric_keys(task_rubrics)

        # Get the prompt for the task (question text)
        task_prompts = prompts[tasks == ta]
        assert len(set(task_prompts)) == 1
        task_prompt = task_prompts[0]

        # Get if the task is a holdout for test
        task_is_test = is_test[tasks == ta]
        task_is_test = set(task_is_test)
        assert len(task_is_test) == 1
        task_is_test = list(task_is_test)[0]

        # each rubric key will be a new task
        for k in rubric_keys: # For each potential rubric item for this task
            new_task_labels = np.array([r.get(k, False) for r in task_rubrics]) # Check if each answer has the rubric item
            unique_labels = np.unique(new_task_labels)
            new_task_classes = len(unique_labels)
            new_task_stats = dict(Counter(new_task_labels))
            new_task = (np.zeros_like(new_task_labels) + task_cnt).astype(int)
            new_q = np.zeros_like(new_task) + ta
            task_label_to_rubric[task_cnt] = list(task_rmap[k]) # TODO: For a rubric item (which are our new tasks)
            # this is going to be a map from the name of that rubric item to all the JSON strings it had.
            # This is going to be same for every rubric item in the same original task. Idk why this is useful

            new_indices.append(task_indices)
            new_labels.append(new_task_labels)
            new_tasks.append(new_task)
            new_questions.append(new_q)

            task_to_split_mapping[task_cnt] = task_is_test # If should be excluded from training
            task_to_class_mapping[task_cnt] = new_task_classes
            task_to_stats_mapping[task_cnt] = new_task_stats
            task_to_prompt[task_cnt] = task_prompt

            task_cnt += 1

    # Combine the new items to get a single list
    new_indices = np.concatenate(new_indices)
    new_labels = np.concatenate(new_labels)
    new_tasks = np.concatenate(new_tasks)
    new_questions = np.concatenate(new_questions)

    return new_indices, new_labels, new_tasks, new_questions, \
           task_to_split_mapping, task_to_class_mapping, \
           task_to_stats_mapping, task_label_to_rubric, task_to_prompt


def remove_small_classes(indices, labels, tasks, questions, task_splits,
                         task_classes, task_stats, rubric_maps, prompt_maps,
                         min_freq=1):
    """
    Small classes will be removed as there are not enough examples in them.

    Small means any rubric category with <min_freq
    """
    new_indices = []
    new_labels = []
    new_tasks = []
    new_questions = []
    new_task_splits = {}
    new_task_classes = {}
    new_task_stats = {}
    new_rubric_maps = {}
    new_prompt_maps = {}

    task_ix = 0
    for ta in np.unique(tasks):
        task_indices = indices[tasks == ta]
        task_questions = questions[tasks == ta]
        task_labels = labels[tasks == ta]
        task_freqs = Counter(task_labels).most_common()
        task_freqs = task_freqs[::-1]
        task_freqs = sorted(task_freqs, key=lambda x: x[0])

        task_cnts = np.array([t[1] for t in task_freqs])
        num_above = np.sum(task_cnts >= min_freq)
        keep_ix = np.where(task_cnts >= min_freq)[0]
        task_freqs = [task_freqs[ix] for ix in keep_ix]

        if num_above < 2:
            # trivial task now... delete
            continue

        # calc old to new index conversion
        old_to_new = {}
        for new, (old, _) in enumerate(task_freqs):
            old_to_new[old] = new

        slicer = np.where(np.in1d(task_labels, list(old_to_new.keys())))[0]
        sub_task_labels = task_labels[slicer]
        new_task_labels = np.array([old_to_new[l] for l in sub_task_labels])

        new_task_indices = task_indices[slicer]
        new_task_questions = task_questions[slicer]

        # build new stats dict for task
        ta_stats = {}
        for l, f in task_freqs:
            ta_stats[l] = f

        new_tasks.extend([task_ix for _ in range(len(new_task_labels))])
        new_labels.extend(list(new_task_labels))
        new_indices.extend(list(new_task_indices))
        new_questions.extend(list(new_task_questions))
        new_task_splits[task_ix] = task_splits[ta]
        new_task_classes[task_ix] = num_above
        new_task_stats[task_ix] = ta_stats
        new_rubric_maps[task_ix] = rubric_maps[ta]
        new_prompt_maps[task_ix] = prompt_maps[ta]

        task_ix += 1

    return new_indices, new_labels, new_tasks, new_questions, \
           new_task_splits, new_task_classes, new_task_stats, \
           new_rubric_maps, new_prompt_maps


def make_binary_tasks(indices, labels, tasks, questions, task_splits,
                      task_classes, task_stats, rubric_maps, prompt_maps,
                      min_freq=1):
    """
    Combine small classes to make binary tasks.
    """
    new_indices = []
    new_labels = []
    new_tasks = []
    new_questions = []
    new_splits = {}
    new_classes = {}
    new_stats = {}
    new_rubric_maps = {}
    new_prompt_maps = {}

    indices = np.array(indices)
    questions = np.array(questions)
    labels = np.array(labels)
    tasks = np.array(tasks)

    for ta in np.unique(tasks):
        task_indices = indices[tasks == ta]
        task_questions = questions[tasks == ta]
        task_labels = labels[tasks == ta]

        if task_classes[ta] == 2:
            # do nothing...
            new_task_indices = list(task_indices)
            new_task_questions = list(task_questions)
            new_task_labels = list(task_labels)
            new_task_tasks = list(tasks[tasks == ta])
            new_task_splits = task_splits[ta]
            new_task_classes = 2
            new_task_stats = task_stats[ta]
            new_task_rubric_maps = rubric_maps[ta]
            new_task_prompt_maps = prompt_maps[ta]
        else:
            # we can get away with a really easy policy: take the
            # most common class and then sum all of the other ones
            # as a second class.
            task_freqs = Counter(task_labels).most_common()
            most_common_label = task_freqs[0][0]

            new_task_indices = list(task_indices)
            new_task_questions = list(task_questions)
            new_task_labels = np.ones_like(task_labels)
            new_task_labels[task_labels == most_common_label] = 0
            new_task_tasks = list(tasks[tasks == ta])
            new_task_splits = task_splits[ta]
            new_task_classes = 2
            new_task_stats = {
                0: sum(new_task_labels == 0),
                1: sum(new_task_labels == 1),
            }
            new_task_labels = list(new_task_labels)
            new_task_rubric_maps = rubric_maps[ta]
            new_task_prompt_maps = prompt_maps[ta]

        new_tasks.extend(new_task_tasks)
        new_labels.extend(new_task_labels)
        new_indices.extend(new_task_indices)
        new_questions.extend(new_task_questions)
        new_splits[ta] = new_task_splits
        new_classes[ta] = new_task_classes
        new_stats[ta] = new_task_stats
        new_rubric_maps[ta] = new_task_rubric_maps
        new_prompt_maps[ta] = new_task_prompt_maps

    return new_indices, new_labels, new_tasks, new_questions, \
           new_splits, new_classes, new_stats, new_rubric_maps, \
           new_prompt_maps


def make_binary_tasks_liberally(indices, labels, tasks, questions, task_splits,
                                task_classes, task_stats, rubric_maps, prompt_maps,
                                min_freq=1):
    """
    Combine small classes to make binary tasks. But do this exhaustively
    for options.
    """
    new_indices = []
    new_labels = []
    new_tasks = []
    new_questions = []
    new_splits = {}
    new_classes = {}
    new_stats = {}
    new_rubric_maps = {}
    new_prompt_maps = {}

    indices = np.array(indices)
    questions = np.array(questions)
    labels = np.array(labels)
    tasks = np.array(tasks)

    task_count = 0

    for ta in np.unique(tasks):
        # print(f"TASK is {ta}")
        task_indices = indices[tasks == ta]
        task_questions = questions[tasks == ta]
        task_labels = labels[tasks == ta]
        # print(task_labels)
        # print(len(task_indices))

        uniq_task_labels = np.unique(task_labels)
        num_labels = uniq_task_labels.shape[0]
        # print(f"THERE are {num_labels} possibilities")

        for i in range(num_labels):
            new_task_indices = list(task_indices)
            new_task_questions = list(task_questions)
            num_entries = len(new_task_indices)
            # all 1's but set some to 0
            new_task_labels = np.ones_like(task_labels)
            new_task_labels[task_labels == uniq_task_labels[i]] = 0

            new_task_tasks = list([task_count for _ in range(num_entries)])
            new_task_splits = task_splits[ta]
            new_task_classes = len(np.unique(new_task_labels))
            new_task_stats = {}
            for _l in np.unique(new_task_labels):
                new_task_stats[i] = sum(new_task_labels == _l)
            new_task_labels = list(new_task_labels)

            new_tasks.extend(new_task_tasks)
            new_labels.extend(new_task_labels)
            new_indices.extend(new_task_indices)
            new_questions.extend(new_task_questions)
            new_splits[task_count] = new_task_splits
            new_classes[task_count] = new_task_classes
            new_stats[task_count] = new_task_stats
            new_rubric_maps[task_count] = rubric_maps[ta]
            new_prompt_maps[task_count] = prompt_maps[ta]

            task_count += 1  # increment count

    return new_indices, new_labels, new_tasks, new_questions, \
           new_splits, new_classes, new_stats, new_rubric_maps, \
           new_prompt_maps


def merge_small_classes(labels, tasks, task_classes, task_stats, rubric_maps):
    """
    Small classes will merged together into a big class
    """
    new_labels = copy.deepcopy(labels)
    new_task_classes = copy.deepcopy(task_classes)
    new_task_stats = copy.deepcopy(task_stats)

    for ta in np.unique(tasks):
        if task_classes[ta] == 2:
            # nothing we can do for binary tasks
            continue

        task_labels = labels[tasks == ta]
        task_freqs = Counter(task_labels).most_common()
        task_freqs = task_freqs[::-1]

        new_classes = []
        group_labels = []
        group_freqs = 0
        goal_freq = task_freqs[-1][1]  # largest class!

        for label, freq in task_freqs:
            if (group_freqs + freq) < goal_freq:
                group_freqs += freq
                group_labels.append(label)
            elif (group_freqs + freq) <= (goal_freq + LEEWAY):
                group_freqs += freq
                group_labels.append(label)
                new_classes.append(copy.deepcopy(group_labels))
                group_labels = []
                group_freqs = 0
            else:
                new_classes.append(copy.deepcopy(group_labels))
                group_labels = [label]
                group_freqs = freq

        if group_freqs > 0 and group_freqs <= goal_freq:
            new_classes.append(copy.deepcopy(group_labels))

        # ok now we need to make new task_labels from the groups
        old_to_new = {}
        for i, grp in enumerate(new_classes):
            for lab in grp:
                old_to_new[lab] = i

        new_task_labels = np.array([old_to_new[l] for l in task_labels])
        new_labels[tasks == ta] = new_task_labels
        new_task_classes[ta] = len(new_classes)
        new_task_stats[ta] = dict(Counter(new_task_labels))

    return new_labels, new_task_classes, new_task_stats


def prune_trivial_classes(indices, labels, tasks, questions, task_classes, task_stats, rubric_maps):
    trivial_tasks = []
    for i in range(len(task_stats)):
        if len(task_stats[i]) < 2:
            trivial_tasks.append(i)

    new_indices = []
    new_labels = []
    new_tasks = []
    new_questions = []
    new_task_classes = {}
    new_task_stats = []
    new_rubric_maps = {}

    unique_tasks = np.unique(tasks)
    new_task_i = 0

    for task_i in unique_tasks:
        if task_i not in trivial_tasks:
            keep_indices = tasks == task_i
            new_indices.append(indices[keep_indices])
            new_labels.append(labels[keep_indices])
            new_tasks.append(np.zeros(sum(task_stats[task_i].values())) + new_task_i)
            new_questions.append(questions[keep_indices])
            new_task_classes[new_task_i] = task_classes[task_i]
            new_task_stats.append(task_stats[task_i])
            new_rubric_maps[new_task_i] = rubric_maps[task_i]
            new_task_i += 1

    new_indices = np.concatenate(new_indices)
    new_labels = np.concatenate(new_labels)
    new_tasks = np.concatenate(new_tasks).astype(int)
    new_questions = np.concatenate(new_questions)

    return new_indices, new_labels, new_tasks, new_questions, \
           new_task_classes, new_task_stats, new_rubric_maps


def shuffle_names_augmentation(token_seqs, vocab, func_tokens, var_tokens, p=0.5):
    """
    We want to shuffle around the <FUNC:int> and <VAR:int> variables
    so the model doesn't start memorizing this.

    @param token_seqs: max_seq_len
    """
    seq_funcs = sorted(list(set(token_seqs).intersection(set(func_tokens))))
    seq_vars = sorted(list(set(token_seqs).intersection(set(var_tokens))))

    new_funcs = list(np.random.choice(func_tokens, size=len(seq_funcs), replace=False))
    new_vars = list(np.random.choice(var_tokens, size=len(seq_vars), replace=False))

    func_map = dict(zip(seq_funcs, new_funcs))
    var_map = dict(zip(seq_vars, new_vars))

    new_token_seqs = []
    for i in range(len(token_seqs)):
        token_i = token_seqs[i]
        if token_i in seq_funcs:
            if random.random() < p:
                new_token_i = func_map[token_i]
            else:
                new_token_i = token_i
        elif token_i in seq_vars:
            if random.random() < p:
                new_token_i = var_map[token_i]
            else:
                new_token_i = token_i
        else:
            new_token_i = token_i
        new_token_seqs.append(new_token_i)

    return new_token_seqs


def shuffle_entries_augmentation(index, equivalence_maps, all_token_seqs, p=0.5):
    if random.random() < p:
        indices = equivalence_maps[index]
        if len(indices) > 0:
            index = np.random.choice(indices)

    return all_token_seqs[index], index


def remap_equivalences(equivalences, index_mapping):
    new_equivalences = {}
    for old_index, new_index in index_mapping.items():
        old_eq = equivalences[old_index]
        # we need to map everything in this
        new_eq = []
        for ix in old_eq:
            # throw away ones not in our split
            if ix in index_mapping:
                new_eq.append(index_mapping[ix])
        new_equivalences[new_index] = new_eq
    return new_equivalences
