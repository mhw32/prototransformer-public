"""
Adapted from https://github.com/YujiaBao/Distributional-Signatures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import \
    pack_padded_sequence, pad_packed_sequence
from src.models.monkeypatch import RobertaModel
from src.utils import utils


class BASE(nn.Module):
    '''
        BASE model
    '''
    def __init__(self, args):
        super(BASE, self).__init__()
        self.args = args

        # cached tensor for speed
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def _compute_l2(self, XS, XQ):
        '''
            Compute the pairwise l2 distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim
            @return dist: query_size x support_size
        '''
        diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
        dist = torch.norm(diff, dim=2)

        return dist

    def _compute_cos(self, XS, XQ):
        '''
            Compute the pairwise cos distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim
            @return dist: query_size support_size
        '''
        dot = torch.matmul(
                XS.unsqueeze(0).unsqueeze(-2),
                XQ.unsqueeze(1).unsqueeze(-1)
                )
        dot = dot.squeeze(-1).squeeze(-1)

        scale = (torch.norm(XS, dim=1).unsqueeze(0) *
                 torch.norm(XQ, dim=1).unsqueeze(1))

        scale = torch.max(scale,
                          torch.ones_like(scale) * 1e-8)

        dist = 1 - dot/scale

        return dist

    def reidx_y(self, YS, YQ):
        '''
            Map the labels into 0,..., way
            @param YS: batch_size
            @param YQ: batch_size
            @return YS_new: batch_size
            @return YQ_new: batch_size
        '''
        unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

        if len(unique1) != len(unique2):
            raise ValueError(
                'Support set classes are different from the query set')

        if len(unique1) != self.args.way:
            raise ValueError(
                'Support set classes are different from the number of ways')

        if int(torch.sum(unique1 - unique2).item()) != 0:
            raise ValueError(
                'Support set classes are different from the query set classes')

        Y_new = torch.arange(start=0, end=self.args.way, dtype=unique1.dtype,
                device=unique1.device)

        return Y_new[inv_S], Y_new[inv_Q]

    def _init_mlp(self, in_d, hidden_ds, drop_rate):
        modules = []

        for d in hidden_ds[:-1]:
            modules.extend([
                nn.Dropout(drop_rate),
                nn.Linear(in_d, d),
                nn.ReLU()])
            in_d = d

        modules.extend([
            nn.Dropout(drop_rate),
            nn.Linear(in_d, hidden_ds[-1])])

        return nn.Sequential(*modules)

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size
            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    @staticmethod
    def compute_acc(pred, true):
        '''
            Compute the accuracy.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        return torch.mean((torch.argmax(pred, dim=1) == true).float()).item()


class R2D2(BASE):
    '''
        META-LEARNING WITH DIFFERENTIABLE CLOSED-FORM SOLVERS
    '''
    def __init__(self, ebd_dim, args):
        super(R2D2, self).__init__(args)
        self.ebd_dim = ebd_dim

        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        # lambda and alpha is learned in the log space

        # cached tensor for speed
        self.I_support = nn.Parameter(
            torch.eye(self.args.shot * self.args.way, dtype=torch.float),
            requires_grad=False)
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def _compute_w(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way
            @return W: ebd_dim * way
        '''

        W = XS.t() @ torch.inverse(
                XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot

        return W

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size
            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def forward(self, XS, YS, XQ, YQ):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size
            @return acc
            @return loss
        '''

        YS, YQ = self.reidx_y(YS, YQ)

        YS_onehot = self._label2onehot(YS)

        W = self._compute_w(XS, YS_onehot)

        pred = (10.0 ** self.alpha) * XQ @ W + self.beta

        loss = F.cross_entropy(pred, YQ)

        acc = BASE.compute_acc(pred, YQ)

        logprobas = F.log_softmax(pred, dim=-1)

        return acc, loss, logprobas


class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional,
            dropout):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout)

    def _sort_tensor(self, input, lengths):
        '''
        pack_padded_sequence  requires the length of seq be in descending order
        to work.
        Returns the sorted tensor, the sorted seq length, and the
        indices for inverting the order.
        Input:
                input: batch_size, seq_len, *
                lengths: batch_size
        Output:
                sorted_tensor: batch_size-num_zero, seq_len, *
                sorted_len:    batch_size-num_zero
                sorted_order:  batch_size
                num_zero
        '''
        sorted_lengths, sorted_order = lengths.sort(0, descending=True)
        sorted_input = input[sorted_order]
        _, invert_order  = sorted_order.sort(0, descending=False)

        # Calculate the num. of sequences that have len 0
        nonzero_idx = sorted_lengths.nonzero()
        num_nonzero = nonzero_idx.size()[0]
        num_zero = sorted_lengths.size()[0] - num_nonzero

        # temporarily remove seq with len zero
        sorted_input = sorted_input[:num_nonzero]
        sorted_lengths = sorted_lengths[:num_nonzero]

        return sorted_input, sorted_lengths, invert_order, num_zero

    def _unsort_tensor(self, input, invert_order, num_zero):
        '''
        Recover the origin order
        Input:
                input:        batch_size-num_zero, seq_len, hidden_dim
                invert_order: batch_size
                num_zero
        Output:
                out:   batch_size, seq_len, *
        '''
        if num_zero == 0:
            input = input[invert_order]

        else:
            dim0, dim1, dim2 = input.size()
            zero = torch.zeros((num_zero, dim1, dim2), device=input.device,
                    dtype=input.dtype)
            input = torch.cat((input, zero), dim=0)
            input = input[invert_order]

        return input

    def forward(self, text, text_len):
        '''
        Input: text, text_len
            text       Variable  batch_size * max_text_len * input_dim
            text_len   Tensor    batch_size
        Output: text
            text       Variable  batch_size * max_text_len * output_dim
        '''
        # Go through the rnn
        # Sort the word tensor according to the sentence length, and pack them together
        sort_text, sort_len, invert_order, num_zero = self._sort_tensor(input=text, lengths=text_len)
        text = pack_padded_sequence(sort_text, lengths=sort_len.cpu().numpy(), batch_first=True)

        # Run through the word level RNN
        text, _ = self.rnn(text)         # batch_size, max_doc_len, args.word_hidden_size

        # Unpack the output, and invert the sorting
        text = pad_packed_sequence(text, batch_first=True)[0] # batch_size, max_doc_len, rnn_size
        text = self._unsort_tensor(text, invert_order, num_zero) # batch_size, max_doc_len, rnn_size

        return text


class META(nn.Module):
    def __init__(self, ebd, args):
        super(META, self).__init__()

        self.args = args

        self.ebd = ebd
        self.aux = get_embedding(args)

        self.ebd_dim = 768

        input_dim = int(args.meta_idf) + self.aux.embedding_dim + \
            int(args.meta_w_target) + int(args.meta_iwf)

        if args.meta_ebd:
            # abalation use distributional signatures with word ebd may fail
            input_dim += self.ebd_dim

        if args.embedding == 'meta':
            self.rnn = RNN(input_dim, 25, 1, True, 0)

            self.seq = nn.Sequential(
                    nn.Dropout(self.args.dropout),
                    nn.Linear(50, 1),
                    )
        else:
            # use a mlp to predict the weight individually
            self.seq = nn.Sequential(
                nn.Linear(input_dim, 50),
                nn.ReLU(),
                nn.Dropout(self.args.dropout),
                nn.Linear(50, 1))

    def forward(self, data, return_score=False):
        '''
            @param data dictionary
                @key text: batch_size * max_text_len
                @key text_len: batch_size
            @param return_score bool
                set to true for visualization purpose
            @return output: batch_size * embedding_dim
        '''
        ebd = self.ebd(data)

        scale = self.compute_score(data, ebd)

        ebd = torch.sum(ebd * scale, dim=1)

        if return_score:
            return ebd, scale

        return ebd

    def _varlen_softmax(self, logit, text_len):
        '''
            Compute softmax for sentences with variable length
            @param: logit: batch_size * max_text_len
            @param: text_len: batch_size
            @return: score: batch_size * max_text_len
        '''
        logit = torch.exp(logit)
        mask = torch.arange(
                logit.size()[-1], device=logit.device,
                dtype=text_len.dtype).expand(*logit.size()
                        ) < text_len.unsqueeze(-1)

        logit = mask.float() * logit
        score = logit / torch.sum(logit, dim=1, keepdim=True)

        return score

    def compute_score(self, data, ebd, return_stats=False):
        '''
            Compute the weight for each word
            @param data dictionary
            @param return_stats bool
                return statistics (input and output) for visualization purpose
            @return scale: batch_size * max_text_len * 1
        '''

        # preparing the input for the meta model
        x = self.aux(data)
        if self.args.meta_idf:
            idf = F.embedding(data['text'], data['idf']).detach()
            x = torch.cat([x, idf], dim=-1)

        if self.args.meta_iwf:
            iwf = F.embedding(data['text'], data['iwf']).detach()
            x = torch.cat([x, iwf], dim=-1)

        if self.args.meta_ebd:
            x = torch.cat([x, ebd], dim=-1)

        if self.args.meta_w_target:
            if self.args.meta_target_entropy:
                w_target = ebd @ data['w_target']
                w_target = F.softmax(w_target, dim=2) * F.log_softmax(w_target,
                        dim=2)
                w_target = -torch.sum(w_target, dim=2, keepdim=True)
                w_target = 1.0 / w_target
                x = torch.cat([x, w_target.detach()], dim=-1)
            else:
                # for rr approxmiation, use the max weight to approximate
                # task-specific importance
                w_target = torch.abs(ebd @ data['w_target'])
                w_target = w_target.max(dim=2, keepdim=True)[0]
                x = torch.cat([x, w_target.detach()], dim=-1)

        if self.args.embedding == 'meta':
            # run the LSTM
            hidden = self.rnn(x, data['text_len'])
        else:
            hidden = x

        # predict the logit
        logit = self.seq(hidden).squeeze(-1)  # batch_size * max_text_len

        score = self._varlen_softmax(logit, data['text_len']).unsqueeze(-1)

        if return_stats:
            return score.squeeze(), idf.squeeze(), w_target.squeeze()
        else:
            return score


def get_embedding(args):
    '''
        @return AUX module with aggregated embeddings or None if args.aux
        did not provide additional embeddings
    '''
    aux = []
    for ebd in args.auxiliary:
        if ebd == 'pos':
            aux.append(POS(args))
        else:
            raise ValueError('Invalid argument for auxiliary ebd')

    if args.cuda != -1:
        aux = [a.cuda(args.cuda) for a in aux]

    model = AUX(aux, args)

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model


class AVG(nn.Module):
    '''
        An aggregation method that encodes every document by its average word
        embeddings.
    '''
    def __init__(self, ebd, args):
        super(AVG, self).__init__()

        self.ebd = ebd
        self.ebd_dim = 768


    def forward(self, ebd):
        '''
            @param data dictionary
            @param weights placeholder used for maml
            @return output: batch_size * embedding_dim
        '''
        ebd = self.ebd(data)

        # count length excluding <pad> and <unk>.
        is_zero = (torch.sum(torch.abs(ebd), dim=2) > 1e-8).float()
        soft_len = torch.sum(is_zero, dim=1, keepdim=True)

        soft_len[soft_len < 1] = 1

        # # don't need to mask out the <pad> tokens, as the embeddings are zero
        ebd = torch.sum(ebd, dim=1)

        ebd = ebd / soft_len

        return ebd


class AUX(nn.Module):
    '''
        Wrapper around combination of auxiliary embeddings
    '''

    def __init__(self, aux, args):
        super(AUX, self).__init__()
        self.args = args
        # this is a list of nn.Module
        self.aux = nn.ModuleList(aux)
        # this is 0 if self.aux is empty
        self.embedding_dim = sum(a.embedding_dim for a in self.aux)

    def forward(self, data, weights=None):
        # torch.cat will discard the empty tensor
        if len(self.aux) == 0:
            if self.args.cuda != -1:
                return torch.FloatTensor().cuda(self.args.cuda)
            return torch.FloatTensor()

        # aggregate results from each auxiliary module
        results = [aux(data, weights) for aux in self.aux]

        # aux embeddings should only be used with cnn, meta or meta_mlp.
        # concatenate together with word embeddings
        assert (self.args.embedding in ['cnn', 'meta', 'meta_mlp', 'lstmatt'])
        x = torch.cat(results, dim=2)

        return x


class POS(nn.Module):
    '''
        Embedding module that combines position-aware embedding
        and standard text embedding.
        Position embedding should only be used with CNN or META
        (sentences are of variable length)
    '''
    def __init__(self, args):
        super(POS, self).__init__()
        self.args = args

        self.embedding_dim = 2 * args.pos_ebd_dim

        # position embedding
        # 2 * length to account for -length to +length
        self.pos1 = nn.Embedding(
                2 * args.pos_max_len, args.pos_ebd_dim, padding_idx=0)
        self.pos2 = nn.Embedding(
                2 * args.pos_max_len, args.pos_ebd_dim, padding_idx=0)

    def forward(self, data, weights=None):
        text = data['text']
        head = data['head'].t()  # (2, n) where [0] is start and [1] is end
        tail = data['tail'].t()  # (2, n)

        assert head.shape[1] == tail.shape[1] == len(text)
        n = head.shape[1]
        max_len = max(data['text_len'])

        # (n, max_len)
        idx = torch.arange(max_len, device=data['text'].device).expand(n, -1)
        # (max_len, 1)
        h0, h1 = head[0].unsqueeze(1), head[1].unsqueeze(1)
        t0, t1 = tail[0].unsqueeze(1), tail[1].unsqueeze(1)
        # filler
        zero = torch.tensor(0, device=data['text'].device)

        # (n, max_len) + add max_len to center 0
        pos1 = torch.where(idx < h0, idx - h0, zero) + \
               torch.where(idx > h1, idx - h1, zero) + self.args.pos_max_len
        pos2 = torch.where(idx < t0, idx - t0, zero) + \
               torch.where(idx > t1, idx - t1, zero) + self.args.pos_max_len

        if weights is None:
            return torch.cat([self.pos1(pos1), self.pos2(pos2)], dim=2)
        else:
            return torch.cat([
                F.embedding(pos1, weights['aux.aux.0.pos1.weight']),
                F.embedding(pos2, weights['aux.aux.0.pos2.weight'])
            ], dim=2)


class DistSign(nn.Module):

    def __init__(self, ebd, args):
        super().__init__()
        self.args = args
        classifier = R2D2(768, args)
        self.classifier = classifier

    def forward(self, XS, YS, XQ, YQ):
        XS = XS.view(-1, 768).contiguous()
        XQ = XQ.view(-1, 768).contiguous()
        YS = YS.view(-1).contiguous()
        YQ = YQ.view(-1).contiguous()
        return self.classifier(XS, YS, XQ, YQ)
