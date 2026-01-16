#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""

    neural = True    # class attribute that indicates that constructor needs extra args
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) # dimensionality of word's embeddings
        

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)
        
        self.E = nn.Parameter(lexicon.clone())
        
        


    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """

        # See the "Parameterization" section of the reading handout to determine
        # what dimensions all your parameters will need.

       
        """"
        Dimensions (see Reading Section H.4):
        - M: [e + d, d]       (forward RNN)
        - M': [e + d, d]      (backward RNN)
        - U_A: [input_dim, feature_dim]  where input_dim = 2*d + 2*k
        - U_B: [input_dim, feature_dim]  where input_dim = 2*d + k + e
        - theta_A: [feature_dim, k]
        - theta_B: [feature_dim, k]
        """
        
        # RNN parameters (Equation 46 in reading)
        # Forward RNN: h_j = σ(M * [h_{j-1}; w_j])
        self.M = nn.Parameter(torch.empty(self.rnn_dim, self.e + self.rnn_dim))
        nn.init.xavier_uniform_(self.M)
        
        # Backward RNN: h'_{j-1} = σ(M' * [w_j; h'_j])
        # Input: [e + d], Output: [d]
        self.M_prime = nn.Parameter(torch.empty(self.rnn_dim, self.e + self.rnn_dim))
        nn.init.xavier_uniform_(self.M_prime)
        
        # Tag embeddings
        self.tag_embeddings = torch.eye(self.k)
        
        # Feature transformation for transitions
        transition_input_dim = 2 * self.rnn_dim + 2 * self.k
        feature_dim = 128
        
        self.U_A = nn.Parameter(torch.empty(transition_input_dim, feature_dim))
        nn.init.xavier_uniform_(self.U_A)
        
        self.theta_A = nn.Parameter(torch.empty(feature_dim, 1))
        nn.init.normal_(self.theta_A, mean=0, std=0.01)
        
        # Feature transformation for emissions
        emission_input_dim = 2 * self.rnn_dim + self.k + self.e
        
        self.U_B = nn.Parameter(torch.empty(emission_input_dim, feature_dim))
        nn.init.xavier_uniform_(self.U_B)
        
        self.theta_B = nn.Parameter(torch.empty(feature_dim, 1))
        nn.init.normal_(self.theta_B, mean=0, std=0.01)
    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use AdamW optimizer for better training stability
        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""

        """Pre-compute biRNN hidden states for all positions.
        
        Computes:
        - h vectors: forward RNN encoding prefixes (Equation 46, left)
        - h' vectors: backward RNN encoding suffixes (Equation 46, right)
        
        These will be accessed by A_at() and B_at().
        
        Args:
            isent: List of (word_idx, tag_idx) tuples
        """
        n = len(isent)  # Including BOS and EOS
        
        # Extract word indices and get embeddings
        word_indices = torch.tensor([w for w, t in isent], dtype=torch.long)
        word_embeds = self.E[word_indices]  # [n, e]
        
        # ===== Forward RNN: compute h_{-1}, h_0, h_1, ..., h_{n-1} =====
        # h_j encodes the prefix w_0...w_j (where w_0 = BOS_WORD)
        self.h_vectors = []
        h_prev = torch.zeros(self.rnn_dim)  # h_{-1} = 0 (before BOS)
        
        for j in range(n):
            # h_j = σ(M * [h_{j-1}; w_j])
            concat_input = torch.cat([h_prev, word_embeds[j]])  # [d + e]
            h_j = torch.sigmoid(self.M.mv(concat_input))          # [d]
            self.h_vectors.append(h_j)
            h_prev = h_j
        
        # ===== Backward RNN: compute h'_n, h'_{n-1}, ..., h'_0 =====
        # h'_j encodes the suffix w_{j+1}...w_{n-1} (where w_{n-1} = EOS_WORD)
        self.h_prime_vectors = [None] * n
        h_prime_next = torch.zeros(self.rnn_dim)  # h'_n = 0 (after EOS)
        
        for j in range(n - 1, -1, -1):
            # h'_{j-1} = σ(M' * [w_j; h'_j])
            concat_input = torch.cat([word_embeds[j], h_prime_next])  # [e + d]
            h_prime_j_minus_1 = torch.sigmoid(self.M_prime @ concat_input)  # [d]
            self.h_prime_vectors[j] = h_prime_next  # Store h'_j at position j
            h_prime_next = h_prime_j_minus_1
           # you fill this in!

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

        """Compute non-stationary transition potential matrix at position.
        
        Returns φ^A matrix of shape [k, k] where entry [s, t] is:
        φ^A(s, t, w, j) = exp(θ^A · f^A(s, t, w, j))
        
        where f^A(s,t,w,j) = σ(U^A * [h_{j-2}; s_emb; t_emb; h'_j])
        (Equations 45, 47 in reading)
        
        Args:
            position: Position j in the sentence (1-indexed for tags)
            sentence: Integerized sentence
        """
        j = position
        k = self.k
        
        # Get context vectors for position j
        # h_{j-2}: encodes prefix before the transition
        if j >= 2:
            h_before = self.h_vectors[j - 2]
        else:
            h_before = torch.zeros(self.rnn_dim)
        
        # h'_j: encodes suffix after the transition
        if j < len(self.h_prime_vectors):
            h_after = self.h_prime_vectors[j]
        else:
            h_after = torch.zeros(self.rnn_dim)
        
        # Vectorized computation for all (s, t) pairs
        # Expand context to [k, k, d]
        h_before_expanded = h_before.unsqueeze(0).unsqueeze(0).expand(k, k, -1)
        h_after_expanded = h_after.unsqueeze(0).unsqueeze(0).expand(k, k, -1)
        
        # Tag embeddings: create grid of all (s, t) combinations
        # s_emb: [k, 1, k] - each row s repeated k times
        # t_emb: [1, k, k] - each column t repeated k times
        s_emb = self.tag_embeddings.unsqueeze(1).expand(-1, k, -1)  # [k, 1, k] -> [k, k, k]
        t_emb = self.tag_embeddings.unsqueeze(0).expand(k, -1, -1)  # [1, k, k] -> [k, k, k]
        
        # Concatenate features: [k, k, 2d + 2k]
        features = torch.cat([h_before_expanded, s_emb, t_emb, h_after_expanded], dim=-1)
        
        # Flatten to [k*k, 2d + 2k] for batch matrix multiplication
        features_flat = features.view(k * k, -1)
        
        # Apply feature transformation: [k*k, 2d+2k] @ [2d+2k, feature_dim] = [k*k, feature_dim]
        hidden = torch.sigmoid(features_flat @ self.U_A)
        
        
        logits = hidden @ self.theta_A  # [k*k, feature_dim] @ [feature_dim, 1] = [k*k, 1]
        
        # Convert logits to potential matrix
        A_matrix = torch.exp(logits.squeeze().view(k, k))  # [k*k, 1] -> [k, k] -> exp
        return A_matrix  # you fill this in!
        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""

        """Compute non-stationary emission potential vector at position.
        
        Returns φ^B vector of shape [k, 1] where entry [t] is:
        φ^B(t, w, w, j) = exp(θ^B · f^B(t, w, w, j))
        
        where f^B(t,w,w,j) = σ(U^B * [h_{j-1}; t_emb; w_emb; h'_j])
        (Equations 45, 48 in reading)
        
        Args:
            position: Position j in the sentence
            sentence: Integerized sentence
        """
        j = position
        k = self.k
        
        # Get context vectors for position j
        # h_{j-1}: encodes prefix up to (but not including) current word
        if j >= 1:
            h_before = self.h_vectors[j - 1]
        else:
            h_before = torch.zeros(self.rnn_dim)
        
        # h'_j: encodes suffix starting after current word
        if j < len(self.h_prime_vectors):
            h_after = self.h_prime_vectors[j]
        else:
            h_after = torch.zeros(self.rnn_dim)
        
        # Get word embedding for current position
        word_idx = sentence[j][0]  # sentence is list of (word_idx, tag_idx)
        w_emb = self.E[word_idx]    # [e]
        
        # Expand context to [k, d]
        h_before_expanded = h_before.unsqueeze(0).expand(k, -1)
        h_after_expanded = h_after.unsqueeze(0).expand(k, -1)
        
        # Expand word embedding to [k, e]
        w_emb_expanded = w_emb.unsqueeze(0).expand(k, -1)
        
        # Tag embeddings: [k, k] (already in correct shape)
        t_emb = self.tag_embeddings  # [k, k]
        
        # Concatenate features: [k, 2d + k + e]
        features = torch.cat([h_before_expanded, t_emb, w_emb_expanded, h_after_expanded], dim=-1)
        
        # Apply feature transformation: [k, 2d+k+e] @ [2d+k+e, feature_dim] = [k, feature_dim]
        hidden = torch.sigmoid(features @ self.U_B)
        
        logits = hidden @ self.theta_B  # [k, feature_dim] @ [feature_dim, 1] = [k, 1]
        
    
        
        # Convert to potentials and reshape to [k, 1]
        B_vector = torch.exp(logits.squeeze())  # [k, 1]
        return B_vector
                               # you fill this in!
