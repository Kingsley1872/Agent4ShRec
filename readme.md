# AgentAttack

AgentAttack is a project designed to simulate and analyze attacks on recommendation systems. It provides tools for generating fake user profiles, poisoning reviews with fake ratings, and evaluating the impact of such attacks on recommendation models.

## Features

- **Attack Simulation**: Tools to simulate various types of attacks on recommendation systems.
- **Evaluation Metrics**: Evaluate the impact of attacks on model performance.
- **Preprocessing Utilities**: Includes utilities for data preprocessing and poisoned review generation.
- **Support for BERT**: Leverages BERT for advanced attack processing.

## Project Structure

```
AgentAttack/
├── data/                     # Dataset storage
├── model/                    # Recommendation and attack models
│   ├── nnmf.py               # Non-negative matrix factorization model
│   ├── trainer_rec.py        # Recommendation model trainer
│   ├── trainer_rec_surprise.py # Trainer using Surprise library
│   └── attack_model/         # Attack-related models
├── result/                   # Results storage
├── review_attack_main/       # Review attack logic
│   ├── bert_process_attack.py # BERT-based attack processing
│   ├── preprocess.py         # Data preprocessing
│   ├── review_attack_main.py # Main entry for review attacks
│   └── stopwords.txt         # Stopwords list
├── test_main/                # Main testing scripts
│   ├── main_attack.py        # Attack testing
│   ├── main_eval_attack.py   # Attack evaluation
│   ├── main_train_rec.py     # Recommendation model training
├── textProcess/              # Text processing utilities
│   ├── poison_review.py      # Poisoned review generation
│   ├── embedding/            # Embedding files (e.g., GloVe)
│   └── utils.py              # Helper functions
├── utils/                    # General utilities
└── readme.md                 # Project documentation
```

## Getting Started

### Prerequisites

Ensure you have Python installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Usage

#### Run Attack Simulation

Simulate an attack on the recommendation system:

```bash
python test_main/main_attack.py
```

#### Train Recommendation Model

Train the recommendation model:

```bash
python test_main/main_train_rec.py
```

#### Evaluate Attack Impact

Evaluate the impact of attacks on the recommendation model:

```bash
python test_main/main_eval_attack.py
```
