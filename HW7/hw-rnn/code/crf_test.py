#!/usr/bin/env python3

# Subclass ConditionalRandomFieldBackprop to get a model that uses some
# contextual features of your choice.  This lets you test the revision to hmm.py
# that uses those features.

#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Test non-stationary CRF features before implementing full biRNN.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import tensor, Tensor, cuda
from jaxtyping import Float

from corpus import Tag, Word, IntegerizedSentence
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)

torch.manual_seed(1337)
cuda.manual_seed(69_420)

class ConditionalRandomFieldTest(ConditionalRandomFieldBackprop):
    """A CRF with simple non-stationary features for testing.
    
    This tests that modifications to hmm.py work correctly by implementing
    simple context-dependent features that an HMM cannot learn.
    """
    
    neural = True  # Tells tag.py to pass lexicon and rnn_dim
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        """Construct a test CRF with simple non-stationary features.
        
        Args:
            tagset: Tag integerizer
            vocab: Word integerizer  
            lexicon: Word embeddings [V, e]
            rnn_dim: Not actually used in this simple test, but accepted for compatibility
            unigram: Whether to use unigram model
        """
        
        # Must call nn.Module.__init__ before super().__init__
        nn.Module.__init__(self)
        
        self.E = lexicon          # Word embeddings [V, e]
        self.e = lexicon.size(1)  # Embedding dimension
        self.rnn_dim = rnn_dim    # Accepted but not used
        
        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        """Initialize parameters for simple non-stationary features.
        
        We'll implement:
        1. Base stationary parameters (from parent)
        2. Position-based features (for 'pos' dataset)
        3. Next-word features (for 'next' dataset)
        """
        
        # Initialize base stationary parameters
        super().init_params()
        
        # Additional parameters for non-stationary features
        
        # Position features: bias for each (position mod 4, tag) pair
        # Useful for 'pos' dataset where tags depend on position mod 4
        self.position_bias = nn.Parameter(torch.randn(4, self.k) * 0.1)
        
        # Next-word features: map next word embedding to tag scores
        # Useful for 'next' dataset where tag depends on next word
        self.W_next = nn.Parameter(torch.randn(self.e, self.k) * 0.1)
        
        # Previous-word features: map previous word embedding to tag scores  
        self.W_prev = nn.Parameter(torch.randn(self.e, self.k) * 0.1)

    @override
    def updateAB(self) -> None:
        """Don't compute stationary A and B matrices.
        
        We'll compute position-specific potentials in A_at() and B_at() instead.
        """
        pass  # Do nothing - we use A_at() and B_at() instead

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Precompute word embeddings for all positions in sentence.
        
        This demonstrates the eager precomputation strategy.
        We extract all word embeddings once, then A_at() and B_at() can look them up.
        """
        # Extract word indices from integerized sentence
        word_indices = torch.tensor([w for w, t in isent], dtype=torch.long)
        
        # Get embeddings for all words
        self.word_embeds = self.E[word_indices]  # [n, e]
        
        # Store sentence length for boundary checks
        self.sent_length = len(isent)

    @override
    @typechecked
    def A_at(self, position: int, sentence) -> Tensor:
        """Compute non-stationary transition potentials at position.
        
        Features:
        1. Base stationary transitions from WA
        2. Position-specific bias (position mod 4)
        
        For 'pos' dataset: different positions favor different tags.
        
        Args:
            position: Position j in sentence
            sentence: Integerized sentence
            
        Returns:
            [k, k] transition potential matrix
        """
        k = self.k
        
        # Start with base stationary transitions
        if self.unigram:
            base_A = torch.exp(self.WA).repeat(k, 1)
        else:
            base_A = torch.exp(self.WA)
        
        # Add position-based modification
        # For 'pos' dataset: tags alternate based on position mod 4
        pos_idx = position % 4
        pos_bias = self.position_bias[pos_idx]  # [k] - bias for each target tag
        
        # Apply bias: affects all transitions s→t based on target tag t
        A_modified = base_A * torch.exp(pos_bias).unsqueeze(0)  # [k, k] * [1, k]
        
        # Enforce structural zeros
        A_modified[self.eos_t, :] = 0.0
        A_modified[:, self.bos_t] = 0.0
        if not self.unigram:
            A_modified[self.eos_t, self.bos_t] = 1.0  # Allow EOS→BOS
        
        return A_modified

    @override
    @typechecked
    def B_at(self, position: int, sentence) -> Tensor:
        """Compute non-stationary emission potentials at position.
        
        Features:
        1. Base stationary emissions from WB
        2. Next-word features (for 'next' dataset)
        3. Previous-word features (for better context)
        
        For 'next' dataset: tag at position j depends on word at position j+1.
        
        Args:
            position: Position j in sentence
            sentence: Integerized sentence
            
        Returns:
            [k, 1] emission potential vector for word at this position
        """
        k = self.k
        w_j = sentence[position][0]  # Current word index
        
        # Base stationary emission potentials
        base_B = torch.exp(self.WB[:, w_j])  # [k]
        
        # Add context features if sentence embeddings are available
        if hasattr(self, 'word_embeds'):
            context_scores = torch.zeros(k)
            
            # Feature 1: Next word (critical for 'next' dataset)
            if position + 1 < self.sent_length:
                next_word_emb = self.word_embeds[position + 1]  # [e]
                next_scores = self.W_next.T @ next_word_emb     # [k]
                context_scores = context_scores + next_scores
            
            # Feature 2: Previous word (helps with general context)
            if position - 1 >= 0:
                prev_word_emb = self.word_embeds[position - 1]  # [e]
                prev_scores = self.W_prev.T @ prev_word_emb     # [k]
                context_scores = context_scores + prev_scores
            
            # Combine base and context features
            B_modified = base_B * torch.exp(context_scores)
        else:
            B_modified = base_B
        
        # Enforce structural zeros
        B_modified[self.eos_t] = 0.0
        B_modified[self.bos_t] = 0.0
        
        return B_modified.unsqueeze(1)  # [k, 1]