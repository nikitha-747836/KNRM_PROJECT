"""
Data utilities for K-NRM project.

DATASET STRUCTURE:
  TRAIN : 2000 triples — 8 TRAIN topics  (model learns these)
  VAL   :  400 triples — HARD pairs from TRAIN topics
            Hard = relevant doc from ONE topic vs relevant doc from DIFFERENT topic
            This makes val harder so models don't all hit 1.0 at epoch 2
  TEST  :  400 triples — 5 TEST topics   (completely UNSEEN during training)

Why hard val pairs matter:
  Easy val: relevant doc vs pasta recipe → both models score 1.0 immediately
  Hard val: ML doc vs IR doc → model must distinguish similar-looking topics
            This shows the TRUE learning curve difference between K-NRM and LK-NRM
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import re


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class SimpleVocab:
    def __init__(self):
        self.word2id   = {'<PAD>': 0, '<UNK>': 1}
        self.id2word   = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = {}

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def build_vocab(self, texts):
        for text in texts:
            for w in self.tokenize(text):
                self.word_freq[w] = self.word_freq.get(w, 0) + 1
        for w in self.word_freq:
            if w not in self.word2id:
                idx = len(self.word2id)
                self.word2id[w] = idx
                self.id2word[idx] = w

    def encode(self, text, max_len):
        ids = [self.word2id.get(w, 1) for w in self.tokenize(text)]
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        return ids

    @property
    def size(self):
        return len(self.word2id)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN TOPICS  (8 topics — model trains on these)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_TOPICS = [
    {
        "query": "machine learning neural networks deep learning",
        "relevant": [
            "deep learning backpropagation neural network weights activation layers training",
            "convolutional network image classification feature extraction pooling deep architecture",
            "recurrent lstm sequence prediction neural architecture hidden state output",
            "neural network batch training epochs activation function relu sigmoid output layer",
            "feedforward network hidden layers nonlinear activation gradient update weights bias",
        ],
        "irrelevant": [
            "ancient roman architecture colosseum gladiator emperor forum ruins",
            "pasta tomato sauce garlic olive oil cooking recipe dinner",
            "football goal penalty kick stadium crowd championship trophy match",
            "ocean beach waves surfing sunscreen vacation snorkel coral",
            "music guitar chord melody harmony symphony orchestra concert",
        ]
    },
    {
        "query": "information retrieval document ranking search",
        "relevant": [
            "inverted index query term frequency document ranking score retrieval",
            "bm25 okapi term weighting document score ranking retrieval system",
            "relevance feedback query expansion retrieval model term overlap score",
            "document ranking relevance query matching term frequency inverse document",
            "search engine index query document score ranking retrieval result",
        ],
        "irrelevant": [
            "climate change warming temperature sea level carbon emissions",
            "stock market trading shares investment portfolio dividend earnings",
            "surgery patient hospital diagnosis prescription treatment medicine",
            "car engine horsepower transmission fuel mileage acceleration speed",
            "painting canvas brush stroke art museum gallery exhibition",
        ]
    },
    {
        "query": "natural language processing text classification sentiment",
        "relevant": [
            "sentiment analysis opinion mining text classification model prediction",
            "named entity recognition sequence labeling tagging annotation corpus",
            "tokenization stemming lemmatization preprocessing stop words vocabulary",
            "text classification categories labels feature extraction bag words",
            "language model text representation corpus training annotation label",
        ],
        "irrelevant": [
            "basketball court dribble three point shot championship game score",
            "hiking trail campsite wildlife nature outdoor adventure camping",
            "chemistry periodic table element reaction laboratory experiment",
            "dentistry tooth filling crown root canal hygiene treatment",
            "fashion clothing design trend runway textile pattern fabric",
        ]
    },
    {
        "query": "support vector machine kernel classification",
        "relevant": [
            "svm margin hyperplane kernel rbf gaussian classification boundary",
            "kernel function feature space support vector machine optimization",
            "soft margin regularization kernel polynomial linear svm classifier",
            "support vector regression kernel decision boundary margin slack",
            "classification kernel trick high dimensional feature space svm",
        ],
        "irrelevant": [
            "sailing boat wind navigation harbour anchor crew voyage",
            "economics supply demand market price equilibrium consumer",
            "carpentry wood joint nail furniture chair table shelf",
            "swimming pool lane stroke freestyle backstroke technique",
            "veterinary animal pet vaccination health checkup treatment",
        ]
    },
    {
        "query": "word embeddings vector representation semantic",
        "relevant": [
            "word2vec skip gram embedding semantic space vector representation",
            "glove co occurrence matrix word vector representation training corpus",
            "contextual embedding language model pretraining representation learning",
            "semantic vector analogy similarity distance embedding space word",
            "distributed representation word vector dense embedding neural model",
        ],
        "irrelevant": [
            "pizza topping cheese crust bake oven restaurant delivery",
            "marathon training shoes pace distance calories endurance",
            "world war treaty peace nation empire battle victory",
            "geography mountain river lake continent population capital",
            "aviation aircraft pilot altitude wing fuel flight runway",
        ]
    },
    {
        "query": "gradient descent optimization loss function",
        "relevant": [
            "stochastic gradient descent batch learning rate schedule momentum update",
            "adam optimizer weight decay regularization loss gradient update step",
            "backpropagation chain rule gradient neural network training computation",
            "convergence optimization loss minima saddle point gradient step learning",
            "learning rate schedule warmup decay gradient clipping optimization step",
        ],
        "irrelevant": [
            "architecture building design glass steel urban city planning",
            "literature novel character plot theme fiction biography author",
            "virus infection immune antibody vaccine disease treatment",
            "photography camera lens aperture shutter exposure composition",
            "archaeology fossil excavation artifact museum ancient civilization",
        ]
    },
    {
        "query": "ranking evaluation metrics ndcg mrr",
        "relevant": [
            "ndcg normalized discounted cumulative gain evaluation ranking metric",
            "mean reciprocal rank mrr precision recall evaluation query document",
            "relevance judgment evaluation benchmark query document assessment metric",
            "evaluation metric ranking performance precision recall ndcg score",
            "graded relevance assessment ranking quality benchmark evaluation dataset",
        ],
        "irrelevant": [
            "theater drama stage actor director play script audience",
            "geology rock mineral tectonic plate earthquake volcano",
            "finance tax income deduction audit regulation policy",
            "gardening plant soil sunlight fertilizer harvest vegetable",
            "plumbing pipe valve water pressure drain fixture repair",
        ]
    },
    {
        "query": "neural ranking model query document matching",
        "relevant": [
            "cosine similarity query document vector space model retrieval score",
            "soft matching word similarity score document retrieval ranking neural",
            "translation matrix word pair similarity neural ranking model relevance",
            "query document matching score relevance neural ranking output model",
            "interaction based neural ranking model term matching query document",
        ],
        "irrelevant": [
            "basketball coach drill practice team player court season",
            "cooking ingredient oven temperature roast simmer stir fry",
            "sailing navigation compass harbour anchor crew boat voyage",
            "economics demand equilibrium market price producer surplus",
            "veterinary vaccination pet health animal checkup treatment",
        ]
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TEST TOPICS  (5 topics — NEVER seen during training)
# ─────────────────────────────────────────────────────────────────────────────

TEST_TOPICS = [
    {
        "query": "attention mechanism transformer self attention heads",
        "relevant": [
            "self attention query key value transformer multi head mechanism layer",
            "attention weight softmax scaled dot product key value matrix model",
            "transformer encoder decoder attention position embedding sequence layer",
            "attention heads parallel scaled dot product query key value output",
            "multi head attention concatenate project linear transformer architecture",
        ],
        "irrelevant": [
            "basketball coach drill practice team court season game player",
            "cooking recipe oven temperature roast simmer stir fry ingredient",
            "geography mountain river lake continent country capital population",
            "carpentry wood joint nail furniture shelf table cabinet screw",
            "sailing boat navigation compass harbour anchor crew voyage wind",
        ]
    },
    {
        "query": "reinforcement learning reward agent policy environment",
        "relevant": [
            "reinforcement learning agent environment reward state action policy",
            "q learning bellman equation value function policy gradient signal",
            "markov decision process state transition reward discount factor episode",
            "deep reinforcement learning actor critic policy optimization reward",
            "policy gradient reward signal environment state action value update",
        ],
        "irrelevant": [
            "fashion clothing style trend runway textile pattern fabric design",
            "gardening plant soil sunlight fertilizer seed harvest vegetable",
            "photography camera lens aperture shutter exposure composition",
            "swimming pool lane freestyle butterfly backstroke technique race",
            "veterinary animal pet vaccination health checkup disease treatment",
        ]
    },
    {
        "query": "convolutional neural network image feature extraction",
        "relevant": [
            "cnn convolutional filter feature map pooling stride image classification",
            "image recognition resnet vgg inception feature extraction deep learning",
            "convolutional layer activation pooling fully connected softmax output",
            "object detection segmentation pixel feature map neural network layer",
            "filter kernel stride padding feature map convolutional activation pool",
        ],
        "irrelevant": [
            "sailing boat ocean wind navigation harbour anchor crew voyage",
            "economics supply demand market price equilibrium consumer producer",
            "dentistry tooth cavity filling crown root canal hygiene",
            "plumbing pipe valve water pressure drain fixture repair leak",
            "archaeology fossil excavation artifact museum civilization site",
        ]
    },
    {
        "query": "transfer learning pretrained fine tuning downstream",
        "relevant": [
            "transfer learning pretrained model fine tune downstream task feature",
            "bert gpt pretrained language model fine tuning classification adapt",
            "domain adaptation source target knowledge pretrained weights layer",
            "feature extraction freeze layers fine tune classification head model",
            "pretrained weights initialization fine tune domain specific task adapt",
        ],
        "irrelevant": [
            "carpentry wood joint nail furniture chair table shelf cabinet",
            "swimming lane stroke freestyle butterfly backstroke technique race",
            "veterinary animal pet dog cat vaccination health checkup",
            "aviation aircraft pilot altitude wing fuel flight runway cockpit",
            "geology rock mineral tectonic earthquake volcano crater formation",
        ]
    },
    {
        "query": "language model text generation perplexity tokens",
        "relevant": [
            "language model text generation perplexity token prediction gpt decoder",
            "autoregressive sequence generation probability distribution token sample",
            "perplexity evaluation metric language model cross entropy log likelihood",
            "text generation beam search sampling temperature nucleus decoding token",
            "next token prediction autoregressive language model probability sequence",
        ],
        "irrelevant": [
            "aviation aircraft pilot altitude wing turbine fuel runway cockpit",
            "archaeology fossil excavation artifact museum ancient civilization dig",
            "plumbing pipe valve water pressure drain fixture installation repair",
            "fashion clothing design style trend runway fabric textile pattern",
            "economics supply demand market price equilibrium consumer surplus",
        ]
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Triple Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_train_triples(topics, n_samples, seed=42):
    """
    Generate EASY training triples:
      positive = relevant doc for this query
      negative = 50% from topic's own irrelevant docs
                 50% relevant doc from a DIFFERENT topic (hard negative)
    """
    random.seed(seed)
    np.random.seed(seed)
    triples = []
    for _ in range(n_samples):
        topic = random.choice(topics)
        query = topic["query"]
        pos   = random.choice(topic["relevant"])
        if random.random() < 0.5:
            neg = random.choice(topic["irrelevant"])
        else:
            other = random.choice([t for t in topics if t is not topic])
            neg   = random.choice(other["relevant"])
        triples.append((query, pos, neg))
    return triples


def generate_hard_val_triples(topics, n_samples, seed=99):
    """
    Generate HARD validation triples:
      positive = relevant doc for query A
      negative = relevant doc for query B (different topic, but also ML/IR related)
    
    This forces the model to distinguish between SIMILAR topics.
    E.g. 'ML neural networks' vs 'Information retrieval' — both are technical
    but the model must learn which is more relevant to the query.
    
    Hard val prevents all models from hitting 1.0 in epoch 1-2,
    so we can see the REAL learning curve difference.
    """
    random.seed(seed)
    np.random.seed(seed)
    triples = []
    for _ in range(n_samples):
        topic_pos = random.choice(topics)
        query     = topic_pos["query"]
        pos       = random.choice(topic_pos["relevant"])

        # Hard negative: always a RELEVANT doc from a DIFFERENT topic
        # (not an obvious irrelevant like pasta or football)
        topic_neg = random.choice([t for t in topics if t is not topic_pos])
        neg       = random.choice(topic_neg["relevant"])

        triples.append((query, pos, neg))
    return triples


def generate_test_triples(topics, n_samples, seed=7):
    """Generate test triples from TEST topics (never seen during training)."""
    random.seed(seed)
    np.random.seed(seed)
    triples = []
    for _ in range(n_samples):
        topic = random.choice(topics)
        query = topic["query"]
        pos   = random.choice(topic["relevant"])
        # Mix easy and hard negatives for realistic test
        if random.random() < 0.3:
            neg = random.choice(topic["irrelevant"])
        else:
            other = random.choice([t for t in topics if t is not topic])
            neg   = random.choice(other["relevant"])
        triples.append((query, pos, neg))
    return triples


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, triples, vocab, q_max=12, d_max=35):
        self.triples = triples
        self.vocab   = vocab
        self.q_max   = q_max
        self.d_max   = d_max

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        q, p, n = self.triples[idx]
        return (torch.LongTensor(self.vocab.encode(q, self.q_max)),
                torch.LongTensor(self.vocab.encode(p, self.d_max)),
                torch.LongTensor(self.vocab.encode(n, self.d_max)))


# ─────────────────────────────────────────────────────────────────────────────
# Loader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_all_loaders(batch_size=32):
    """
    Returns vocab, train_loader, val_loader, test_loader.

    Train : 2000 easy triples  from TRAIN_TOPICS
    Val   :  400 HARD triples  from TRAIN_TOPICS (all negatives are cross-topic relevant docs)
    Test  :  400 mixed triples from TEST_TOPICS  (completely unseen)
    """
    train_triples = generate_train_triples(TRAIN_TOPICS, n_samples=2000, seed=42)
    val_triples   = generate_hard_val_triples(TRAIN_TOPICS, n_samples=400, seed=99)
    test_triples  = generate_test_triples(TEST_TOPICS,  n_samples=400, seed=7)

    # Build vocab from ALL splits
    vocab = SimpleVocab()
    all_texts = []
    for q, p, n in train_triples + val_triples + test_triples:
        all_texts += [q, p, n]
    vocab.build_vocab(all_texts)

    train_ds = RankingDataset(train_triples, vocab)
    val_ds   = RankingDataset(val_triples,   vocab)
    test_ds  = RankingDataset(test_triples,  vocab)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"\n  Dataset split:")
    print(f"    Train : {len(train_triples):>5} triples | {len(TRAIN_TOPICS)} topics (seen) | easy negatives")
    print(f"    Val   : {len(val_triples):>5} triples | {len(TRAIN_TOPICS)} topics (seen) | HARD cross-topic negatives")
    print(f"    Test  : {len(test_triples):>5} triples | {len(TEST_TOPICS)} topics (UNSEEN)")
    print(f"    Vocab : {vocab.size} unique words\n")

    return vocab, train_loader, val_loader, test_loader


def get_bert_loaders(tokenizer, batch_size=16):
    """Build BERT data loaders for C-KNRM."""
    import torch
    train_triples = generate_train_triples(TRAIN_TOPICS, n_samples=800, seed=42)
    val_triples   = generate_hard_val_triples(TRAIN_TOPICS, n_samples=160, seed=99)
    test_triples  = generate_test_triples(TEST_TOPICS,  n_samples=160, seed=7)

    class BERTDataset(torch.utils.data.Dataset):
        def __init__(self, triples, tok, q_max=32, d_max=64):
            self.triples = triples
            self.tok = tok
            self.q_max = q_max
            self.d_max = d_max
        def __len__(self): return len(self.triples)
        def _enc(self, text, mx):
            e = self.tok(text, max_length=mx, padding='max_length',
                         truncation=True, return_tensors='pt')
            return e['input_ids'].squeeze(0), e['attention_mask'].squeeze(0)
        def __getitem__(self, idx):
            q, p, n = self.triples[idx]
            qi, qm = self._enc(q, self.q_max)
            pi, pm = self._enc(p, self.d_max)
            ni, nm = self._enc(n, self.d_max)
            return qi, qm, pi, pm, ni, nm

    tr = BERTDataset(train_triples, tokenizer)
    va = BERTDataset(val_triples,   tokenizer)
    te = BERTDataset(test_triples,  tokenizer)
    return (torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False),
            torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False))
