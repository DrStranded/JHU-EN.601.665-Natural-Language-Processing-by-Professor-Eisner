# JHU CS 601.465/665 Natural Language Processing ClassNotes

Course materials and homework notes for **Natural Language Processing** (Fall 2025) at Johns Hopkins University, taught by Prof. Jason Eisner.

## Course Overview

This course covers fundamental and modern techniques in Natural Language Processing, from classical probabilistic models to contemporary deep learning approaches. Topics include language modeling, parsing, semantics, sequence labeling, and large language models.

**Course Website**: [https://www.cs.jhu.edu/~jason/465/](https://www.cs.jhu.edu/~jason/465/)

---

## Repository Structure

```
NLP/
├── HW1/          # Designing Context-Free Grammars
├── HW2/          # Probability and Information Theory
├── HW3/          # N-gram Language Models
├── HW4/          # Probabilistic Parsing
├── HW5/          # Compositional Semantics
├── HW6/          # Unsupervised Tagging with HMMs
├── HW7/          # Discriminative Tagging with CRFs
└── HW8/          # Large Language Models
```

---

## Knowledge Map

### Module 1: Foundations 
- Formal languages (Regular Expressions, FSAs, CFGs)
- N-gram language models
- Probability theory (joint/conditional probability, chain rule, Bayes' theorem)
- Information theory (surprisal, cross-entropy, perplexity)

### Module 2: Probabilistic Modeling 
- Maximum Likelihood Estimation (MLE)
- Smoothing techniques (Add-λ, Backoff, Good-Turing, Witten-Bell)
- Log-linear models
- Gradient-based optimization
- Regularization (L1/L2)

### Module 3: Syntax and Parsing 
- Context-free parsing algorithms (CKY, Earley)
- Probabilistic CFGs (PCFGs)
- Dependency grammar
- Lexicalized parsing

### Module 4: Semantics 
- Lambda calculus
- Compositional semantics
- Semantic attachment to grammar rules

### Module 5: Neural Networks for NLP 
- Backpropagation
- Word embeddings (word2vec)
- Recurrent Neural Networks (RNN, BiRNN, ELMo)
- PyTorch fundamentals

### Module 6: Sequence Labeling 
- Hidden Markov Models (HMMs)
- Forward-Backward algorithm
- Expectation-Maximization (EM)
- Conditional Random Fields (CRFs)
- Structured prediction

### Module 7: Modern NLP 
- Transformer architecture (attention, positional embeddings)
- Pre-trained language models (BERT, GPT)
- Decoding strategies (greedy, beam search, sampling, MBR)
- Prompting and few-shot learning
- Fine-tuning techniques (PEFT, RLHF, DPO)

---

## Homework Details

### HW1: Designing CFGs
**Topics**: Context-Free Grammars, Formal Language Theory

**Key Concepts**:
- Designing production rules for natural language phenomena
- Handling recursion and ambiguity in grammars
- Understanding the expressiveness of CFGs vs. regular languages

**Skills**:
- Writing CFG rules that generate specific languages
- Analyzing grammatical structures

---

### HW2: Probabilities
**Topics**: Probability Theory, Information Theory, Log-linear Models

**Key Concepts**:
- Joint and conditional probability
- Chain rule and Bayes' theorem
- Surprisal, cross-entropy, and perplexity
- Maximum likelihood estimation
- Bias-variance tradeoff
- Smoothing basics (add-λ, backoff)
- Log-linear model fundamentals
- Gradient ascent and regularization

**Skills**:
- Deriving probability expressions
- Computing perplexity
- Understanding smoothing effects on language models

---

### HW3: Language Models
**Topics**: N-gram Language Models, Smoothing Implementation

**Key Concepts**:
- N-gram model implementation
- Various smoothing techniques in practice
- Model evaluation and hyperparameter tuning

**Skills**:
- Implementing smoothing algorithms
- Working with PyTorch tensors
- Conducting experiments and analyzing results
- Handling out-of-vocabulary (OOV) words

**Tools**: Python, PyTorch

---

### HW4: Parsing
**Topics**: Probabilistic Parsing Algorithms

**Key Concepts**:
- CKY algorithm (bottom-up dynamic programming)
- Earley algorithm (top-down with prediction)
- Probabilistic CFGs (PCFGs)
- Dotted rules and chart parsing
- Lexicalized parsing

**Skills**:
- Implementing dynamic programming for parsing
- Optimizing parser efficiency (sparse matrices, pruning)
- Best-first and beam search strategies

**Difficulty**: This is the most conceptually challenging homework, requiring both theoretical understanding and efficient implementation.

---

### HW5: Semantics
**Topics**: Compositional Semantics, Lambda Calculus

**Key Concepts**:
- Lambda expressions and β-reduction
- Type theory for semantics
- Attaching semantic representations to CFG rules
- Compositional interpretation

**Skills**:
- Writing and reducing λ-expressions
- Designing syntax-semantics interfaces
- Handling semantic phenomena (quantifier scope, etc.)

---

### HW6: Unsupervised Tagging with HMMs
**Topics**: Hidden Markov Models, EM Algorithm

**Key Concepts**:
- HMM structure (states, observations, transitions, emissions)
- Forward algorithm (likelihood computation)
- Backward algorithm
- Forward-Backward algorithm (posterior inference)
- Viterbi decoding (most likely state sequence)
- Expectation-Maximization for parameter learning
- Inside-Outside algorithm (for PCFGs)

**Skills**:
- Implementing dynamic programming in log-space
- Numerical stability techniques
- Convergence monitoring
- Symmetry breaking and avoiding local optima

---

### HW7: Discriminative Tagging with CRFs
**Topics**: Conditional Random Fields, Structured Prediction

**Key Concepts**:
- Perceptron algorithm for structured prediction
- Conditional Random Fields (CRFs)
- Feature engineering for sequence labeling
- Generative vs. discriminative model comparison
- Neural CRFs

**Skills**:
- Designing feature templates
- Implementing CRF training and inference
- Combining neural networks with structured prediction

**Prerequisites**: Builds on HW6 codebase

---

### HW8: Large Language Models
**Topics**: Transformers, Modern LLM Techniques

**Key Concepts**:
- Transformer architecture
  - Self-attention mechanism
  - Positional embeddings
  - Encoder-only (BERT), Decoder-only (GPT), Encoder-Decoder
- Tokenization (BPE, WordPiece)
- Decoding strategies
  - Greedy, beam search
  - Temperature sampling
  - Minimum Bayes Risk (MBR)
- Prompting and few-shot learning
- Fine-tuning approaches
  - Parameter-efficient fine-tuning (LoRA, etc.)
  - Knowledge distillation
  - RLHF (REINFORCE, PPO, DPO)

**Skills**:
- Understanding and analyzing transformer models
- Prompt engineering
- Working with pre-trained LLMs

**Format**: Short written assignment with Jupyter notebooks

---

## Dependency Graph

```
HW1 (CFG) ──────────────────┬──────────────────→ HW4 (Parsing)
                            │                         │
                            │                         ↓
                            └───────────────────→ HW5 (Semantics)

HW2 (Probability) ──┬──→ HW3 (Language Models)
                    │
                    ├──→ HW4 (PCFG Parsing)
                    │
                    └──→ HW6 (HMM/EM) ──→ HW7 (CRF) ──→ HW8 (LLM)
```

---

## Recommended Study Order

1. **Foundations**: HW1 → HW2
2. **Modeling**: HW3 → HW4
3. **Semantics**: HW5
4. **Sequence Models**: HW6 → HW7
5. **Modern NLP**: HW8

---

## Tools and Libraries

- **Python 3.x**
- **PyTorch** - Neural network implementation
- **NumPy** - Numerical computing
- **Jupyter Notebook** - Interactive development (especially HW8)

---

## References

### Textbooks
- **J&M**: Jurafsky & Martin, *Speech and Language Processing* (3rd ed.)
- **M&S**: Manning & Schütze, *Foundations of Statistical Natural Language Processing*

### Key Readings by Topic
| Topic | Reference |
|-------|-----------|
| N-gram LMs | J&M Ch. 3, M&S Ch. 6 |
| Smoothing | M&S Ch. 6, J&M Ch. 4 |
| Parsing | J&M Ch. 18, M&S Ch. 12 |
| Log-linear Models | Collins tutorial, Smith Ch. 3.5 |
| Semantics | J&M Ch. 17-18 |
| HMM/EM | J&M Appendix A, M&S Ch. 11 |
| Neural Networks | J&M Ch. 7-9 |
| Transformers | J&M Ch. 9-12, "The Illustrated Transformer" |

---

## Disclaimer

These materials are provided for **educational reference only**. 

- **Do not copy solutions directly** - Academic integrity policies apply
- Use these as a learning resource to understand concepts and approaches
- Solutions may contain errors or suboptimal implementations
- Course content and assignments may change between semesters

If you are currently enrolled in this course, please complete assignments independently and consult official course resources.

---

## Acknowledgments

- **Prof. Jason Eisner** for designing this comprehensive NLP curriculum
- Johns Hopkins University, Whiting School of Engineering
- Course TAs and staff

---

## License

This repository is for educational purposes. All course materials belong to their respective owners. Please respect academic integrity policies.
