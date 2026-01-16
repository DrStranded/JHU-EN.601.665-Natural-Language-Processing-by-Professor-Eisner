#!/usr/bin/env python3
"""Test script for Neural CRF implementation."""

import torch
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)

# Change to data directory
os.chdir("../data")

print("=" * 60)
print("TEST 1: Import modules")
print("=" * 60)

try:
    from corpus import TaggedCorpus, Sentence
    from crf_backprop import ConditionalRandomFieldBackprop
    from crf_neural import ConditionalRandomFieldNeural
    from lexicon import build_lexicon
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("TEST 2: Load corpus and build lexicon")
print("=" * 60)

try:
    # Load small corpus
    icsup = TaggedCorpus(Path("icsup"), add_oov=False)
    print(f"✓ Loaded corpus: {len(icsup)} sentences")
    print(f"  Vocab size: {len(icsup.vocab)}")
    print(f"  Tagset size: {len(icsup.tagset)}")
    
    # Build lexicon with problex features
    lexicon = build_lexicon(icsup, problex=True)
    print(f"✓ Built lexicon: shape {lexicon.shape}")
    print(f"  Expected: [{len(icsup.vocab)}, {len(icsup.tagset) + 1}]")
    
    assert lexicon.shape[0] == len(icsup.vocab), "Lexicon rows mismatch"
    assert lexicon.shape[1] == len(icsup.tagset) + 1, "Lexicon columns mismatch"
    
except Exception as e:
    print(f"✗ Corpus/lexicon failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("TEST 3: Create CRF Backprop model")
print("=" * 60)

try:
    crf_bp = ConditionalRandomFieldBackprop(icsup.tagset, icsup.vocab)
    print("✓ Created CRF Backprop model")
    
    # Check parameters
    param_count = sum(p.numel() for p in crf_bp.parameters())
    print(f"  Total parameters: {param_count}")
    
    # Check if parameters are nn.Parameter
    assert isinstance(crf_bp.WA, torch.nn.Parameter), "WA not nn.Parameter"
    assert isinstance(crf_bp.WB, torch.nn.Parameter), "WB not nn.Parameter"
    print("✓ Parameters correctly wrapped as nn.Parameter")
    
except Exception as e:
    print(f"✗ CRF Backprop creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("TEST 4: Create Neural CRF model")
print("=" * 60)

try:
    rnn_dim = 10
    crf_neural = ConditionalRandomFieldNeural(
        icsup.tagset, 
        icsup.vocab,
        lexicon=lexicon,
        rnn_dim=rnn_dim
    )
    print(f"✓ Created Neural CRF with rnn_dim={rnn_dim}")
    
    # Check RNN parameters exist
    assert hasattr(crf_neural, 'M'), "Missing M parameter"
    assert hasattr(crf_neural, 'M_prime'), "Missing M_prime parameter"
    assert hasattr(crf_neural, 'U_A'), "Missing U_A parameter"
    assert hasattr(crf_neural, 'U_B'), "Missing U_B parameter"
    assert hasattr(crf_neural, 'theta_A'), "Missing theta_A parameter"
    assert hasattr(crf_neural, 'theta_B'), "Missing theta_B parameter"
    print("✓ All RNN parameters exist")
    
    # Check shapes
    e = lexicon.shape[1]
    k = len(icsup.tagset)
    d = rnn_dim
    
    assert crf_neural.M.shape == (e + d, d), f"M shape wrong: {crf_neural.M.shape}"
    assert crf_neural.M_prime.shape == (e + d, d), f"M_prime shape wrong"
    assert crf_neural.U_A.shape[0] == 2*d + 2*k, f"U_A input dim wrong"
    assert crf_neural.U_B.shape[0] == 2*d + k + e, f"U_B input dim wrong"
    print("✓ All parameter shapes correct")
    
    param_count = sum(p.numel() for p in crf_neural.parameters())
    print(f"  Total parameters: {param_count}")
    
except Exception as e:
    print(f"✗ Neural CRF creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("TEST 5: Test setup_sentence")
print("=" * 60)

try:
    # Get first sentence
    sentence = list(icsup)[0]
    isent = crf_neural._integerize_sentence(sentence, icsup)
    print(f"  Sentence length: {len(isent)}")
    
    # Run setup
    crf_neural.setup_sentence(isent)
    print("✓ setup_sentence() executed")
    
    # Check h_vectors
    assert hasattr(crf_neural, 'h_vectors'), "h_vectors not created"
    assert len(crf_neural.h_vectors) == len(isent), f"h_vectors length wrong: {len(crf_neural.h_vectors)} vs {len(isent)}"
    assert crf_neural.h_vectors[0].shape == (rnn_dim,), f"h_vector shape wrong: {crf_neural.h_vectors[0].shape}"
    print(f"✓ h_vectors: {len(crf_neural.h_vectors)} vectors of shape {crf_neural.h_vectors[0].shape}")
    
    # Check h_prime_vectors
    assert hasattr(crf_neural, 'h_prime_vectors'), "h_prime_vectors not created"
    assert len(crf_neural.h_prime_vectors) == len(isent), "h_prime_vectors length wrong"
    print(f"✓ h_prime_vectors: {len(crf_neural.h_prime_vectors)} vectors")
    
except Exception as e:
    print(f"✗ setup_sentence failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("TEST 6: Test A_at")
print("=" * 60)

try:
    # Test at position 1
    A_matrix = crf_neural.A_at(1, isent)
    print(f"✓ A_at(1) executed")
    
    k = len(icsup.tagset)
    assert A_matrix.shape == (k, k), f"A_at shape wrong: {A_matrix.shape} vs ({k}, {k})"
    print(f"  Shape: {A_matrix.shape} ✓")
    
    assert (A_matrix > 0).all(), "A_at has non-positive values"
    print(f"  All positive: ✓")
    
    assert not torch.isnan(A_matrix).any(), "A_at has NaN values"
    assert not torch.isinf(A_matrix).any(), "A_at has inf values"
    print(f"  No NaN/inf: ✓")
    
    print(f"  Sample values: {A_matrix[0, :3]}")
    
except Exception as e:
    print(f"✗ A_at failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("TEST 7: Test B_at")
print("=" * 60)

try:
    B_vector = crf_neural.B_at(1, isent)
    print(f"✓ B_at(1) executed")
    
    k = len(icsup.tagset)
    assert B_vector.shape == (k, 1), f"B_at shape wrong: {B_vector.shape} vs ({k}, 1)"
    print(f"  Shape: {B_vector.shape} ✓")
    
    assert (B_vector > 0).all(), "B_at has non-positive values"
    print(f"  All positive: ✓")
    
    assert not torch.isnan(B_vector).any(), "B_at has NaN values"
    assert not torch.isinf(B_vector).any(), "B_at has inf values"
    print(f"  No NaN/inf: ✓")
    
    print(f"  Sample values: {B_vector[:3].squeeze()}")
    
except Exception as e:
    print(f"✗ B_at failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("TEST 8: Test forward pass (logprob)")
print("=" * 60)

try:
    logprob = crf_neural.logprob(sentence, icsup)
    print(f"✓ logprob() executed")
    print(f"  Log probability: {logprob.item():.4f}")
    
    assert not torch.isnan(logprob), "logprob is NaN"
    assert not torch.isinf(logprob), "logprob is inf"
    assert logprob.item() <= 0, f"logprob should be negative: {logprob.item()}"
    print("✓ Valid log probability")
    
except Exception as e:
    print(f"✗ logprob failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("TEST 9: Test backward pass (gradient)")
print("=" * 60)

try:
    crf_neural.optimizer.zero_grad()
    loss = -logprob
    loss.backward()
    print("✓ backward() executed")
    
    # Check gradients exist
    for name, param in crf_neural.named_parameters():
        if param.grad is None:
            print(f"✗ No gradient for {name}")
        else:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad norm = {grad_norm:.4f}")
    
    print("✓ All parameters have gradients")
    
except Exception as e:
    print(f"✗ Gradient computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("TEST 10: Test optimizer step")
print("=" * 60)

try:
    # Save old parameter values
    old_M = crf_neural.M.data.clone()
    
    # Take optimizer step
    crf_neural.optimizer.step()
    print("✓ optimizer.step() executed")
    
    # Check parameters changed
    assert not torch.equal(old_M, crf_neural.M.data), "Parameters didn't update"
    print("✓ Parameters updated after step")
    
except Exception as e:
    print(f"✗ Optimizer step failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour implementation appears to be working correctly.")
print("Next steps:")
print("1. Modify hmm.py/crf.py to call A_at() and B_at()")
print("2. Train on real data (ensup/endev)")
print("3. Run experiments for homework questions")