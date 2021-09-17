"""
Sentence embeddings from BERT.
https://github.com/UKPLab/sentence-transformers
"""
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class SentenceBERT(nn.Module):

    def __init__(
            self,
            version='bert-base-nli-stsb-mean-tokens',
            device='cuda',
        ):
        super().__init__()
        self.model = SentenceTransformer(version, device=device)
        self.model.eval()

    def forward(self, sentences, batch_size=None, show_progress_bar=False):
        """Sentences are expect to be a list of strings, e.g.
        sentences = [
            'This framework generates embeddings for each input sentence',
            'Sentences are passed as a list of string.',
            'The quick brown fox jumps over the lazy dog.'
        ]
        """
        if batch_size is None:
            batch_size = len(sentences)
        sentence_embeddings = self.model.encode(
            sentences, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
        )
        return sentence_embeddings
