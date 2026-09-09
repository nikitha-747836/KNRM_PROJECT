"""
demo.py — Interactive ranking demo for K-NRM project.

Shows how K-NRM and LK-NRM rank documents for a given query.
Labels are:
  [REL] = Relevant document (should be ranked HIGH)
  [SEM] = Semi-relevant (related topic but not the best match)
  [IRR] = Irrelevant document (should be ranked LOW)

Usage (Windows):
  python demo.py                   shows K-NRM vs LK-NRM side by side
  python demo.py --model knrm      shows only K-NRM ranking
  python demo.py --model lknrm     shows only LK-NRM ranking
  python demo.py --query 0         scenario 0 (default: machine learning)
  python demo.py --query 1         scenario 1 (information retrieval)
  python demo.py --query 2         scenario 2 (kernel methods / SVM)
  python demo.py --query 3         scenario 3 (gradient descent)
  python demo.py --query 4         scenario 4 (ranking evaluation)

Run AFTER training:
  python main.py --no-bert
  python demo.py
"""

import sys
import os
import torch
import argparse
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Demo Scenarios
# Each doc has a label: "REL", "SEM", or "IRR"
# Labels are used ONLY for display — the model does NOT see them
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name"  : "Scenario 0",
        "query" : "machine learning neural networks deep learning",
        "docs"  : [
            ("REL", "deep learning gradient descent backpropagation neural network weights layers"),
            ("REL", "convolutional network image classification deep learning feature extraction pooling"),
            ("REL", "recurrent lstm sequence model time series prediction neural architecture"),
            ("SEM", "inverted index query document term frequency ranking score retrieval"),
            ("SEM", "word embeddings word2vec fasttext language model representation learning"),
            ("IRR", "ancient roman architecture colosseum gladiator emperor forum ruins"),
            ("IRR", "recipe pasta tomato sauce garlic olive oil cooking dinner"),
            ("IRR", "football match score goal penalty kick stadium crowd championship"),
        ]
    },
    {
        "name"  : "Scenario 1",
        "query" : "information retrieval document ranking search",
        "docs"  : [
            ("REL", "inverted index query term frequency document ranking score retrieval"),
            ("REL", "bm25 okapi term weighting document score ranking retrieval system"),
            ("REL", "relevance feedback query expansion retrieval model term overlap score"),
            ("SEM", "deep learning gradient descent backpropagation neural network weights"),
            ("SEM", "word embeddings word2vec fasttext language model representation learning"),
            ("IRR", "climate change global warming temperature sea level carbon emissions"),
            ("IRR", "stock market trading shares investment portfolio dividend earnings"),
            ("IRR", "surgery patient hospital diagnosis prescription treatment medicine"),
        ]
    },
    {
        "name"  : "Scenario 2",
        "query" : "support vector machine kernel classification",
        "docs"  : [
            ("REL", "svm margin hyperplane kernel rbf gaussian classification boundary"),
            ("REL", "kernel function feature space support vector machine optimization"),
            ("REL", "soft margin regularization kernel polynomial linear svm classifier"),
            ("SEM", "deep learning gradient descent backpropagation neural network weights"),
            ("SEM", "sentiment analysis opinion mining text classification deep learning"),
            ("IRR", "music guitar chord melody harmony composition orchestra concert"),
            ("IRR", "painting canvas brush stroke art museum gallery exhibition"),
            ("IRR", "car engine horsepower transmission fuel economy mileage acceleration"),
        ]
    },
    {
        "name"  : "Scenario 3",
        "query" : "gradient descent optimization learning rate",
        "docs"  : [
            ("REL", "stochastic gradient descent batch learning rate schedule momentum update"),
            ("REL", "adam optimizer weight decay regularization loss gradient update step"),
            ("REL", "backpropagation chain rule gradient neural network training computation"),
            ("SEM", "svm margin hyperplane kernel rbf gaussian classification boundary"),
            ("SEM", "inverted index query document term frequency ranking score retrieval"),
            ("IRR", "architecture building design glass steel urban city planning"),
            ("IRR", "literature novel character plot theme fiction biography author"),
            ("IRR", "virus infection immune antibody vaccine disease treatment health"),
        ]
    },
    {
        "name"  : "Scenario 4",
        "query" : "ranking evaluation metrics ndcg mrr",
        "docs"  : [
            ("REL", "ndcg normalized discounted cumulative gain evaluation ranking metric"),
            ("REL", "mean reciprocal rank mrr precision recall evaluation query document"),
            ("REL", "relevance judgment evaluation benchmark query document assessment metric"),
            ("SEM", "bm25 okapi term weighting document score ranking retrieval system"),
            ("SEM", "inverted index query document term frequency ranking score retrieval"),
            ("IRR", "theater drama stage actor director play script audience performance"),
            ("IRR", "geology rock mineral tectonic plate earthquake volcano crater"),
            ("IRR", "finance tax income deduction audit regulation government policy"),
        ]
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_class, weights_path, vocab_size, embed_dim=128, num_kernels=11):
    """Create model and load saved weights. Returns (model, loaded_ok)."""
    model = model_class(vocab_size=vocab_size, embed_dim=embed_dim,
                        num_kernels=num_kernels)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.eval()
        return model, True
    model.eval()
    return model, False


def get_scores(model, vocab, query, docs, q_max=12, d_max=35):
    """Score every (query, doc) pair. Returns list of floats."""
    model.eval()
    q_ids = torch.LongTensor([vocab.encode(query, q_max)])
    scores = []
    with torch.no_grad():
        for _lbl, doc in docs:
            d_ids = torch.LongTensor([vocab.encode(doc, d_max)])
            score = model(q_ids, d_ids).item()
            scores.append(score)
    return scores


def label_tag(lbl):
    """Convert label string to display tag."""
    if lbl == "REL":
        return "REL"
    elif lbl == "SEM":
        return "SEM"
    else:
        return "IRR"


def label_meaning(lbl):
    """Full meaning of label for display."""
    if lbl == "REL":
        return "Relevant"
    elif lbl == "SEM":
        return "Semi-relevant"
    else:
        return "Irrelevant"


# ─────────────────────────────────────────────────────────────────────────────
# Print functions
# ─────────────────────────────────────────────────────────────────────────────

def print_single_model(model_name, query, docs, scores):
    """Print ranking results for one model clearly."""
    # Zip scores with (label, doc) then sort by score descending
    ranked = sorted(zip(scores, docs), key=lambda x: -x[0])

    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"  Query : '{query}'")
    print(f"{'='*70}")
    print(f"  {'Rank':<6} {'Label':<6} {'Score':>8}   Document")
    print(f"  {'─'*65}")

    for rank, (score, (lbl, doc)) in enumerate(ranked, 1):
        tag     = label_tag(lbl)
        # Truncate doc to 55 chars for clean display
        doc_str = doc[:55] + "..." if len(doc) > 55 else doc
        # Add visual indicator
        marker = " ✓" if lbl == "REL" else ("  " if lbl == "SEM" else " ✗")
        print(f"  {rank:<6} [{tag}]  {score:>+8.4f}   {doc_str}{marker}")

    print(f"  {'─'*65}")

    # Summary stats
    rel_scores = [s for s, (lbl, _) in ranked if lbl == "REL"]
    irr_scores = [s for s, (lbl, _) in ranked if lbl == "IRR"]
    sem_scores = [s for s, (lbl, _) in ranked if lbl == "SEM"]

    top3_labels = [lbl for _, (lbl, _) in ranked[:3]]
    top3_rel    = top3_labels.count("REL")

    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0
    avg_irr = sum(irr_scores) / len(irr_scores) if irr_scores else 0
    gap     = avg_rel - avg_irr

    print(f"\n  Relevant docs in Top-3  : {top3_rel}/3  "
          f"{'(PERFECT)' if top3_rel == 3 else '(GOOD)' if top3_rel >= 2 else '(POOR)'}")
    print(f"  Avg REL score           : {avg_rel:+.4f}")
    print(f"  Avg IRR score           : {avg_irr:+.4f}")
    print(f"  REL vs IRR score gap    : {gap:+.4f}  "
          f"{'(model learned well)' if gap > 0.3 else '(model needs more training)' if gap < 0 else ''}")
    print(f"\n  Legend: [REL]=Relevant ✓   [SEM]=Semi-relevant   [IRR]=Irrelevant ✗")
    print(f"{'='*70}\n")


def print_side_by_side(scenario, scores_knrm, scores_lknrm):
    """Print K-NRM and LK-NRM rankings side by side for easy comparison."""
    query = scenario["query"]
    docs  = scenario["docs"]
    name  = scenario["name"]

    # Sort both by score
    ranked_k  = sorted(zip(scores_knrm,  docs), key=lambda x: -x[0])
    ranked_lk = sorted(zip(scores_lknrm, docs), key=lambda x: -x[0])

    W = 50  # column width

    print(f"\n{'='*(W*2+7)}")
    print(f"  {name}  |  Query: '{query}'")
    print(f"{'='*(W*2+7)}")
    print(f"  {'K-NRM  (Original Paper)':<{W}}  |  {'LK-NRM  (Learnable Kernels + CNN)':<{W}}")
    print(f"  {'─'*W}  |  {'─'*W}")

    for rank in range(len(docs)):
        ks,  (k_lbl,  k_doc)  = ranked_k[rank]
        lks, (lk_lbl, lk_doc) = ranked_lk[rank]

        k_tag  = label_tag(k_lbl)
        lk_tag = label_tag(lk_lbl)

        k_doc_str  = k_doc[:30]  + "..."
        lk_doc_str = lk_doc[:30] + "..."

        k_line  = f"Rank {rank+1} [{k_tag}]  {k_doc_str}  ({ks:+.3f})"
        lk_line = f"Rank {rank+1} [{lk_tag}]  {lk_doc_str}  ({lks:+.3f})"

        print(f"  {k_line:<{W}}  |  {lk_line:<{W}}")

    print(f"  {'─'*W}  |  {'─'*W}")

    # Summary comparison
    k_top3_rel  = sum(1 for _, (lbl, _) in ranked_k[:3]  if lbl == "REL")
    lk_top3_rel = sum(1 for _, (lbl, _) in ranked_lk[:3] if lbl == "REL")

    rel_idx = [i for i, (lbl, _) in enumerate(docs) if lbl == "REL"]
    irr_idx = [i for i, (lbl, _) in enumerate(docs) if lbl == "IRR"]

    k_gap  = (sum(scores_knrm[i]  for i in rel_idx) / len(rel_idx)) - \
             (sum(scores_knrm[i]  for i in irr_idx) / len(irr_idx))
    lk_gap = (sum(scores_lknrm[i] for i in rel_idx) / len(rel_idx)) - \
             (sum(scores_lknrm[i] for i in irr_idx) / len(irr_idx))

    print(f"\n  Relevant in Top-3  :  K-NRM = {k_top3_rel}/3        "
          f"LK-NRM = {lk_top3_rel}/3")
    print(f"  REL vs IRR gap     :  K-NRM = {k_gap:+.4f}    "
          f"LK-NRM = {lk_gap:+.4f}")
    winner = "LK-NRM" if lk_gap > k_gap else "K-NRM"
    print(f"  Better separation  :  {winner}  "
          f"({'LK-NRM improved gap by ' + f'{lk_gap - k_gap:+.4f}' if winner == 'LK-NRM' else 'equal'})")
    print(f"\n  Legend: [REL]=Relevant ✓   [SEM]=Semi-relevant   [IRR]=Irrelevant ✗")
    print(f"{'='*(W*2+7)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="K-NRM Ranking Demo")
    p.add_argument('--model', default='both',
                   choices=['both', 'knrm', 'lknrm'],
                   help='Which model to show (default: both side by side)')
    p.add_argument('--query', type=int, default=0,
                   help='Scenario index 0-4 (default: 0)')
    return p.parse_args()


def main():
    args = parse_args()

    # Pick scenario
    scenario = SCENARIOS[args.query % len(SCENARIOS)]

    print(f"\n  K-NRM Demo  —  {scenario['name']}")
    print(f"  Query: '{scenario['query']}'")

    # ── Load vocabulary ────────────────────────────────────────────────────
    vocab_path = 'results/vocab.pkl'
    if not os.path.exists(vocab_path):
        print("\n  ERROR: results/vocab.pkl not found!")
        print("  Run training first:  python main.py --no-bert\n")
        sys.exit(1)

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"  Vocabulary: {vocab.size} words loaded from results/vocab.pkl")

    # ── Load models ────────────────────────────────────────────────────────
    from models.knrm          import KNRM
    from models.enhanced_knrm import LKNRM

    torch.manual_seed(0)
    knrm,  k_ok  = load_model(KNRM,  'results/knrm.pt',  vocab.size)
    torch.manual_seed(0)
    lknrm, lk_ok = load_model(LKNRM, 'results/lknrm.pt', vocab.size)

    print(f"  K-NRM  weights : {'✓ loaded from results/knrm.pt'  if k_ok  else '✗ NOT FOUND — run main.py first'}")
    print(f"  LK-NRM weights : {'✓ loaded from results/lknrm.pt' if lk_ok else '✗ NOT FOUND — run main.py first'}")

    # ── Score documents ────────────────────────────────────────────────────
    if args.model == 'knrm':
        scores = get_scores(knrm, vocab, scenario["query"], scenario["docs"])
        print_single_model("K-NRM  (Original Paper)",
                           scenario["query"], scenario["docs"], scores)

    elif args.model == 'lknrm':
        scores = get_scores(lknrm, vocab, scenario["query"], scenario["docs"])
        print_single_model("LK-NRM  (Learnable Kernels + CNN)",
                           scenario["query"], scenario["docs"], scores)

    else:  # both
        scores_k  = get_scores(knrm,  vocab, scenario["query"], scenario["docs"])
        scores_lk = get_scores(lknrm, vocab, scenario["query"], scenario["docs"])
        print_side_by_side(scenario, scores_k, scores_lk)


if __name__ == '__main__':
    main()
