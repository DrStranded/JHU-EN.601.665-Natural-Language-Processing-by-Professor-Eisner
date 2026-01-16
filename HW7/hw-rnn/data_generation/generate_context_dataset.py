#!/usr/bin/env python3
"""
Generate artificial dataset where word identity provides NO information.
Only context determines the correct tag.

Task: Nested Bracket Matching with Ambiguous Words
- All words are from vocabulary {a, b, c, d, e}
- Each word's tag depends on its ROLE in the structure, not its identity
- Roles: OPEN (opens a context), MID (middle of context), CLOSE (closes context)

Pattern examples:
  a b c d e  →  OPEN MID MID MID CLOSE
  a b c b a  →  OPEN MID CLOSE MID END
  
The key: word "a" can be OPEN, MID, CLOSE, or END depending on position in structure!

This requires:
1. Counting depth (how many opens without closes)
2. Looking forward (is there a matching close later?)
3. Looking backward (are we inside a context?)

Stationary CRF will fail because it can't count or look far enough.
biRNN-CRF should succeed because forward/backward RNNs track structure.
"""

import random
from pathlib import Path

# Minimal vocabulary - same words appear with different tags
WORDS = ['a', 'b', 'c', 'd', 'e']

# Tags representing structural roles
TAGS = ['OPEN', 'MID', 'CLOSE', 'SINGLE']

def generate_nested_structure(max_depth=3, max_length=15):
    """
    Generate a sequence with nested structure.
    
    Structure rules:
    - OPEN starts a context
    - CLOSE ends the most recent unclosed context
    - MID appears inside a context
    - SINGLE appears alone (not in any context)
    
    Returns list of tags representing the structure.
    """
    length = random.randint(5, max_length)
    tags = []
    depth = 0  # Current nesting depth
    
    for i in range(length):
        if depth == 0:
            # Not in any context
            if random.random() < 0.6 and length - i > 2:
                # Open a new context
                tags.append('OPEN')
                depth += 1
            else:
                # Single element
                tags.append('SINGLE')
        else:
            # Inside a context
            if random.random() < 0.2 and depth < max_depth and length - i > 3:
                # Open a nested context
                tags.append('OPEN')
                depth += 1
            elif random.random() < 0.4 or i == length - 1:
                # Close current context
                tags.append('CLOSE')
                depth -= 1
            else:
                # Middle element
                tags.append('MID')
    
    # Close any remaining open contexts
    while depth > 0:
        tags.append('CLOSE')
        depth -= 1
    
    return tags

def tags_to_sentence(tags):
    """
    Convert tag sequence to (word, tag) pairs.
    Words are randomly chosen - they provide NO information!
    """
    sentence = []
    for tag in tags:
        word = random.choice(WORDS)
        sentence.append((word, tag))
    return sentence

def format_sentence(sentence):
    """Format as word/tag word/tag ..."""
    return ' '.join(f"{word}/{tag}" for word, tag in sentence)

def generate_corpus(num_sentences):
    """Generate corpus."""
    sentences = []
    for _ in range(num_sentences):
        tags = generate_nested_structure()
        sentence = tags_to_sentence(tags)
        sentences.append(format_sentence(sentence))
    return sentences

def save_corpus(sentences, filepath):
    """Save corpus to file."""
    with open(filepath, 'w') as f:
        for sent in sentences:
            f.write(sent + '\n')
    print(f"Saved {len(sentences)} sentences to {filepath}")

def analyze_corpus(sentences):
    """Print statistics about the corpus."""
    # Count tag frequencies per word
    word_tag_counts = {}
    for sent in sentences:
        for token in sent.split():
            word, tag = token.split('/')
            if word not in word_tag_counts:
                word_tag_counts[word] = {}
            word_tag_counts[word][tag] = word_tag_counts[word].get(tag, 0) + 1
    
    print("\n=== Word-Tag Distribution (showing ambiguity) ===")
    for word in sorted(word_tag_counts.keys()):
        counts = word_tag_counts[word]
        total = sum(counts.values())
        print(f"{word}: ", end='')
        for tag in sorted(counts.keys()):
            pct = counts[tag] / total * 100
            print(f"{tag}={counts[tag]}({pct:.1f}%) ", end='')
        print()

def main():
    random.seed(465)  # For reproducibility
    
    # Generate datasets
    print("Generating context-dependent dataset...")
    train_sup = generate_corpus(600)
    dev = generate_corpus(120)
    test = generate_corpus(200)
    
    # Save files
    save_corpus(train_sup, 'contextsup')
    save_corpus(dev, 'contextdev')
    save_corpus(test, 'contexttest')
    
    # Print examples
    print("\n=== Example sentences ===")
    print("(Notice: same word appears with different tags!)")
    for i in range(10):
        print(f"{i+1}. {train_sup[i]}")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Training: {len(train_sup)} sentences")
    print(f"Dev: {len(dev)} sentences")
    print(f"Test: {len(test)} sentences")
    
    total_tokens = sum(len(sent.split()) for sent in train_sup)
    print(f"Total training tokens: {total_tokens}")
    print(f"Vocabulary size: {len(WORDS)} (deliberately small!)")
    print(f"Tag set size: {len(TAGS)}")
    
    # Analyze ambiguity
    analyze_corpus(train_sup)
    
    print("\n=== Why this is hard ===")
    print("1. Same word appears with ALL tags (maximum ambiguity)")
    print("2. Only context determines correct tag")
    print("3. Requires tracking nested structure (counting depth)")
    print("4. Stationary CRF has no way to resolve ambiguity")
    print("5. biRNN-CRF can use forward/backward context to track structure")

if __name__ == '__main__':
    main()
