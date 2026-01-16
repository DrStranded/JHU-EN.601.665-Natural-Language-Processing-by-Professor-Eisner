#!/usr/bin/env python3
"""
Speech recognition using language model.
Selects best transcription from candidates using Bayes' Theorem.
"""

import logging
import math
from pathlib import Path
from probs import LanguageModel, BOS, EOS
import argparse

log = logging.getLogger(Path(__file__).stem)

def compute_lm_score(words: list, lm: LanguageModel) -> float:
    """
    Compute log2 p(word_sequence) using the language model.
    
    Args:
        words: list like ['<s>', 'word1', 'word2', ..., '</s>']
        lm: trained language model
    
    Returns:
        log2 probability of the word sequence
    """
    log_prob = 0.0  # in nats (natural log)
    
    # Initialize context with BOS BOS
    x, y = BOS, BOS
    
    # Process each word in the sequence
    # Skip the first '<s>' and process up to and including '</s>'
    for i in range(1, len(words)):
        word = words[i]
        
        # Convert </s> to EOS
        if word == '</s>':
            z = EOS
        else:
            z = word
        
        # Get log probability from language model (in nats)
        log_prob += lm.log_prob(x, y, z)
        
        # Update context for next word
        if z == EOS:
            break  # Don't update context after EOS
        x, y = y, z
    
    # Convert from nats to bits
    log2_prob = log_prob / math.log(2)
    
    return log2_prob


def process_utterance_file(filepath: Path, lm: LanguageModel):
    """
    Process one utterance file and select best candidate.
    
    Returns:
        (best_wer, true_length)
    """
    with open(filepath) as f:
        lines = f.readlines()
    
    # Line 1: true transcription - only read the length
    true_length = int(lines[0].split()[0])
    
    # Lines 2-10: candidate transcriptions
    best_candidate = None
    best_score = float('-inf')
    
    for line in lines[1:]:
        parts = line.split()
        wer = float(parts[0])
        acoustic_score = float(parts[1])  # log2 p(u|w)
        length = int(parts[2])
        words = parts[3:]  # ['<s>', 'word1', ..., '</s>']
        
        # Compute language model score: log2 p(w)
        lm_score = compute_lm_score(words, lm)
        
        # Total score: log2 p(w|u) ∝ log2 p(u|w) + log2 p(w)
        total_score = acoustic_score + lm_score
        
        if total_score > best_score:
            best_score = total_score
            best_candidate = wer
    
    return best_candidate, true_length


def main():
    parser = argparse.ArgumentParser(description='Speech recognition using language model')
    parser.add_argument('model', type=Path, help='Path to trained language model')
    parser.add_argument('files', type=Path, nargs='+', help='Speech utterance files')
    args = parser.parse_args()
    
    # Load language model
    lm = LanguageModel.load(args.model)
    
    total_errors = 0.0
    total_words = 0
    
    # Process each utterance file
    for filepath in args.files:
        wer, true_length = process_utterance_file(filepath, lm)
        
        # Print WER for this file
        print(f"{wer:.3f} {filepath.name}")
        
        # Accumulate for overall WER
        num_errors = wer * true_length
        total_errors += num_errors
        total_words += true_length
    
    # Print overall WER
    overall_wer = total_errors / total_words
    print(f"{overall_wer:.3f} OVERALL")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()