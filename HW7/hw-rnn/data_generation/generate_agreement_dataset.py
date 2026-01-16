#!/usr/bin/env python3
"""
Generate an artificial tagging dataset with long-distance subject-verb agreement.

This dataset challenges models to learn:
1. Long-distance dependencies (verb tag depends on subject 1-5 words back)
2. Bidirectional context (must look back to find the subject)
3. Noisy patterns (15% of sentences violate the agreement rule)

Expected results:
- Stationary CRF: ~60-70% (can't capture long-distance agreement)
- biRNN-CRF: ~85-95% (backward RNN can see the subject)
"""

import random
from pathlib import Path

# Vocabulary with grammatical properties
VOCAB = {
    'sg_nouns': ['dog', 'cat', 'bird', 'car', 'book'],
    'pl_nouns': ['dogs', 'cats', 'birds', 'cars', 'books'],
    'sg_verbs': ['runs', 'eats', 'flies', 'moves', 'reads'],
    'pl_verbs': ['run', 'eat', 'fly', 'move', 'read'],
    'adj': ['big', 'small', 'red', 'blue', 'fast'],
    'adv': ['quickly', 'slowly', 'loudly', 'quietly', 'carefully'],
}

# Tag set
TAGS = {
    'NOUN_SG': 'singular noun',
    'NOUN_PL': 'plural noun', 
    'VERB_SG': 'singular verb',
    'VERB_PL': 'plural verb',
    'ADJ': 'adjective',
    'ADV': 'adverb',
}

def generate_sentence(noise_prob=0.15):
    """
    Generate a sentence with subject-verb agreement pattern.
    
    Pattern: [NOUN] [ADJ/ADV]* [VERB] [ADJ/ADV]*
    
    Agreement rule: NOUN number must match VERB number
    - NOUN_SG → VERB_SG (correct)
    - NOUN_PL → VERB_PL (correct)
    - With noise_prob, violate this rule
    
    Returns:
        List of (word, tag) tuples
    """
    # Choose subject (singular or plural)
    is_singular = random.choice([True, False])
    
    if is_singular:
        subject = random.choice(VOCAB['sg_nouns'])
        subject_tag = 'NOUN_SG'
        correct_verb_tag = 'VERB_SG'
        correct_verbs = VOCAB['sg_verbs']
        incorrect_verbs = VOCAB['pl_verbs']
    else:
        subject = random.choice(VOCAB['pl_nouns'])
        subject_tag = 'NOUN_PL'
        correct_verb_tag = 'VERB_PL'
        correct_verbs = VOCAB['pl_verbs']
        incorrect_verbs = VOCAB['sg_verbs']
    
    # Build sentence
    sentence = [(subject, subject_tag)]
    
    # Add 0-3 modifiers before verb (creates variable distance)
    num_pre_modifiers = random.randint(0, 3)
    for _ in range(num_pre_modifiers):
        if random.random() < 0.5:
            word = random.choice(VOCAB['adj'])
            tag = 'ADJ'
        else:
            word = random.choice(VOCAB['adv'])
            tag = 'ADV'
        sentence.append((word, tag))
    
    # Add verb (with or without noise)
    add_noise = random.random() < noise_prob
    if add_noise:
        # Violate agreement rule
        verb = random.choice(incorrect_verbs)
        verb_tag = 'VERB_PL' if is_singular else 'VERB_SG'
    else:
        # Follow agreement rule
        verb = random.choice(correct_verbs)
        verb_tag = correct_verb_tag
    
    sentence.append((verb, verb_tag))
    
    # Add 0-2 modifiers after verb
    num_post_modifiers = random.randint(0, 2)
    for _ in range(num_post_modifiers):
        if random.random() < 0.5:
            word = random.choice(VOCAB['adj'])
            tag = 'ADJ'
        else:
            word = random.choice(VOCAB['adv'])
            tag = 'ADV'
        sentence.append((word, tag))
    
    return sentence

def format_sentence(sentence):
    """Format sentence in corpus format: word/tag word/tag ..."""
    return ' '.join(f"{word}/{tag}" for word, tag in sentence)

def generate_corpus(num_sentences, noise_prob=0.15):
    """Generate corpus with multiple sentences."""
    sentences = []
    for _ in range(num_sentences):
        sentence = generate_sentence(noise_prob)
        sentences.append(format_sentence(sentence))
    return sentences

def save_corpus(sentences, filepath):
    """Save corpus to file."""
    with open(filepath, 'w') as f:
        for sent in sentences:
            f.write(sent + '\n')
    print(f"Saved {len(sentences)} sentences to {filepath}")

def main():
    random.seed(42)  # For reproducibility
    
    # Generate datasets
    train_sup = generate_corpus(500, noise_prob=0.15)  # Training: 500 sentences
    dev = generate_corpus(100, noise_prob=0.15)        # Dev: 100 sentences
    test = generate_corpus(200, noise_prob=0.15)       # Test: 200 sentences
    
    # Save to files
    save_corpus(train_sup, 'agreementsup')
    save_corpus(dev, 'agreementdev')
    save_corpus(test, 'agreementtest')
    
    # Print examples
    print("\n=== Example sentences ===")
    for i in range(5):
        print(f"{i+1}. {train_sup[i]}")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Training: {len(train_sup)} sentences")
    print(f"Dev: {len(dev)} sentences")
    print(f"Test: {len(test)} sentences")
    
    # Count tokens
    total_tokens = sum(len(sent.split()) for sent in train_sup)
    print(f"Total training tokens: {total_tokens}")
    
    # Vocabulary size
    all_words = set()
    for sent in train_sup + dev + test:
        for token in sent.split():
            word = token.split('/')[0]
            all_words.add(word)
    print(f"Vocabulary size: {len(all_words)}")
    print(f"Tag set size: {len(TAGS)}")

if __name__ == '__main__':
    main()
