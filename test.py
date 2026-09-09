"""
test.py — Standalone test evaluation on UNSEEN topics.

Run this AFTER training:
  Step 1:  python main.py --no-bert     (train models, saves weights)
  Step 2:  python test.py               (evaluate on unseen test topics)

What this script does:
  1. Loads saved K-NRM and LK-NRM weights from results/
  2. Loads saved vocabulary from results/vocab.pkl
  3. Generates 400 test triples from 5 UNSEEN test topics
  4. Evaluates both models: Accuracy, NDCG@1, NDCG@5, MRR
  5. Shows per-topic breakdown (which topic the model handles best/worst)
  6. Saves results/test_results.png

Windows commands:
  python test.py                  (test both models)
  python test.py --model knrm     (test only K-NRM)
  python test.py --model lknrm    (test only LK-NRM)
"""

import sys
import os
import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.knrm          import KNRM
from models.enhanced_knrm import LKNRM
from utils.dataset        import (SimpleVocab, TEST_TOPICS,
                                  generate_test_triples, RankingDataset)
from torch.utils.data     import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — same as trainer.py
# ─────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(relevances, k):
    rel = np.array(relevances[:k], dtype=float)
    if len(rel) == 0:
        return 0.0
    dcg  = (rel / np.log2(np.arange(2, len(rel) + 2))).sum()
    idcg = (np.sort(rel)[::-1] / np.log2(np.arange(2, len(rel) + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0


def mrr_score(relevances):
    for i, r in enumerate(relevances):
        if r == 1:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_batch(pos_scores, neg_scores):
    accs, ndcg1s, ndcg5s, mrrs = [], [], [], []
    for ps, ns in zip(pos_scores.tolist(), neg_scores.tolist()):
        ranked     = sorted([(ps, 1), (ns, 0)], key=lambda x: -x[0])
        relevances = [r for _, r in ranked]
        accs.append(1.0 if ps > ns else 0.0)
        ndcg1s.append(ndcg_at_k(relevances, k=1))
        ndcg5s.append(ndcg_at_k(relevances, k=5))
        mrrs.append(mrr_score(relevances))
    return {
        'acc':    np.mean(accs),
        'ndcg@1': np.mean(ndcg1s),
        'ndcg@5': np.mean(ndcg5s),
        'mrr':    np.mean(mrrs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Overall evaluation on full test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_overall(model, vocab, device, n=400, seed=7):
    """Evaluate model on 400 test triples from all 5 unseen topics."""
    test_triples = generate_test_triples(TEST_TOPICS, n_samples=n, seed=seed)
    test_ds      = RankingDataset(test_triples, vocab)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    model.eval()
    all_acc, all_ndcg1, all_ndcg5, all_mrr = [], [], [], []

    with torch.no_grad():
        for q_ids, pos_ids, neg_ids in test_loader:
            q_ids   = q_ids.to(device)
            pos_ids = pos_ids.to(device)
            neg_ids = neg_ids.to(device)
            m = evaluate_batch(model(q_ids, pos_ids), model(q_ids, neg_ids))
            all_acc.append(m['acc'])
            all_ndcg1.append(m['ndcg@1'])
            all_ndcg5.append(m['ndcg@5'])
            all_mrr.append(m['mrr'])

    return {
        'acc':    np.mean(all_acc),
        'ndcg@1': np.mean(all_ndcg1),
        'ndcg@5': np.mean(all_ndcg5),
        'mrr':    np.mean(all_mrr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-topic evaluation — shows which topics model handles best / worst
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_per_topic(model, vocab, device, n_per_topic=80):
    """
    Test model on each TEST topic separately.
    Positive = relevant doc for this topic's query.
    Negative = irrelevant doc from same topic.
    80 pairs per topic so results are stable.
    """
    results = {}
    model.eval()

    for topic in TEST_TOPICS:
        # Build triples for just this one topic
        random.seed(123)
        np.random.seed(123)
        triples = []
        for _ in range(n_per_topic):
            q   = topic["query"]
            pos = random.choice(topic["relevant"])
            neg = random.choice(topic["irrelevant"])
            triples.append((q, pos, neg))

        ds     = RankingDataset(triples, vocab)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        all_acc, all_ndcg1, all_ndcg5, all_mrr = [], [], [], []
        with torch.no_grad():
            for q_ids, pos_ids, neg_ids in loader:
                q_ids   = q_ids.to(device)
                pos_ids = pos_ids.to(device)
                neg_ids = neg_ids.to(device)
                m = evaluate_batch(model(q_ids, pos_ids), model(q_ids, neg_ids))
                all_acc.append(m['acc'])
                all_ndcg1.append(m['ndcg@1'])
                all_ndcg5.append(m['ndcg@5'])
                all_mrr.append(m['mrr'])

        results[topic['query']] = {
            'acc':    np.mean(all_acc),
            'ndcg@1': np.mean(all_ndcg1),
            'ndcg@5': np.mean(all_ndcg5),
            'mrr':    np.mean(all_mrr),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────────────────────────────────────

def print_overall(overall_dict):
    print("\n" + "=" * 68)
    print("  OVERALL TEST RESULTS  (400 triples, 5 UNSEEN topics)")
    print("=" * 68)
    print(f"  {'Model':<12} | {'Accuracy':>10} | {'NDCG@1':>8} | "
          f"{'NDCG@5':>8} | {'MRR':>8}")
    print(f"  {'-'*58}")
    for name, m in overall_dict.items():
        print(f"  {name:<12} | {m['acc']:>10.4f} | {m['ndcg@1']:>8.4f} | "
              f"{m['ndcg@5']:>8.4f} | {m['mrr']:>8.4f}")
    print("=" * 68)

    if len(overall_dict) == 2:
        names = list(overall_dict.keys())
        base  = overall_dict[names[0]]
        enh   = overall_dict[names[1]]
        print(f"\n  Improvement of {names[1]} over {names[0]}:")
        for metric in ['acc', 'ndcg@1', 'ndcg@5', 'mrr']:
            diff   = enh[metric] - base[metric]
            sign   = '+' if diff >= 0 else ''
            status = 'BETTER' if diff > 0 else ('SAME' if diff == 0 else 'WORSE')
            bar    = '█' * max(1, int(abs(diff) * 100))
            print(f"    {metric.upper():<8} : {sign}{diff:.4f}  {bar}  ({status})")


def print_per_topic(per_topic_dict):
    print("\n" + "=" * 68)
    print("  PER-TOPIC BREAKDOWN  (80 pairs per topic, easy negatives)")
    print("  Shows which TEST topics the model handles best and worst")
    print("=" * 68)

    for model_name, topic_results in per_topic_dict.items():
        print(f"\n  [{model_name}]")
        print(f"  {'Topic Query':<44} | {'Acc':>7} | {'NDCG@1':>7} | {'MRR':>7}")
        print(f"  {'-'*66}")

        sorted_topics = sorted(topic_results.items(),
                               key=lambda x: x[1]['acc'], reverse=True)

        for i, (query, m) in enumerate(sorted_topics):
            short = query[:42] + '..' if len(query) > 42 else query
            tag   = ' ← BEST'  if i == 0                        else \
                    ' ← WORST' if i == len(sorted_topics) - 1   else ''
            print(f"  {short:<44} | {m['acc']:>7.4f} | "
                  f"{m['ndcg@1']:>7.4f} | {m['mrr']:>7.4f}{tag}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_test_results(overall_dict, per_topic_dict,
                      save_path="results/test_results.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    COLORS = ['#1565C0', '#E64A19']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#FAFAFA')

    # ── Left: overall metrics bar chart ──────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#F5F5F5')
    ax.grid(True, color='white', linewidth=1.5, zorder=0, axis='y')

    metric_keys   = ['acc', 'ndcg@1', 'ndcg@5', 'mrr']
    metric_labels = ['Accuracy', 'NDCG@1', 'NDCG@5', 'MRR']
    x     = np.arange(len(metric_keys))
    width = 0.32
    mnames = list(overall_dict.keys())

    for i, mname in enumerate(mnames):
        vals   = [overall_dict[mname][k] for k in metric_keys]
        offset = (i - len(mnames) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=mname, color=COLORS[i], alpha=0.85, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=8.5, color=COLORS[i], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0.7, 1.08)
    ax.set_title('Overall Test Set Metrics\n(400 triples, 5 unseen topics)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Right: per-topic accuracy ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#F5F5F5')
    ax2.grid(True, color='white', linewidth=1.5, zorder=0, axis='x')

    first_model   = list(per_topic_dict.keys())[0]
    topic_queries = list(per_topic_dict[first_model].keys())
    short_labels  = [q[:32] + '..' if len(q) > 32 else q for q in topic_queries]
    y = np.arange(len(topic_queries))

    for i, mname in enumerate(per_topic_dict.keys()):
        accs   = [per_topic_dict[mname][q]['acc'] for q in topic_queries]
        offset = (i - len(per_topic_dict) / 2 + 0.5) * 0.3
        ax2.barh(y + offset, accs, 0.3,
                 label=mname, color=COLORS[i], alpha=0.85, zorder=3)
        for j, (yi, val) in enumerate(zip(y + offset, accs)):
            ax2.text(val + 0.003, yi, f'{val:.3f}',
                     va='center', fontsize=8.5, color=COLORS[i], fontweight='bold')

    ax2.set_yticks(y)
    ax2.set_yticklabels(short_labels, fontsize=9)
    ax2.set_xlim(0.7, 1.1)
    ax2.set_title('Per-Topic Accuracy\n(80 pairs per unseen topic)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Accuracy', fontsize=10)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('K-NRM vs LK-NRM — Test Set Evaluation on Unseen Topics',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"\n  [Saved → {save_path}]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="K-NRM Test Evaluation")
    p.add_argument('--model', default='all', choices=['all', 'knrm', 'lknrm'])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 68)
    print("  K-NRM — TEST EVALUATION")
    print("  Evaluating on UNSEEN topics (not seen during training)")
    print("=" * 68)

    # ── Check weights exist ────────────────────────────────────────────────
    vocab_path = 'results/vocab.pkl'
    if not os.path.exists(vocab_path):
        print("\n  ERROR: results/vocab.pkl not found!")
        print("  Run training first:  python main.py --no-bert")
        sys.exit(1)

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"\n  Loaded vocabulary : {vocab.size} words")

    # ── Load models ────────────────────────────────────────────────────────
    models_to_test = {}

    if args.model in ('all', 'knrm'):
        if not os.path.exists('results/knrm.pt'):
            print("  ERROR: results/knrm.pt not found.")
            print("  Run:  python main.py --no-bert")
            sys.exit(1)
        torch.manual_seed(0)
        knrm = KNRM(vocab_size=vocab.size, embed_dim=128, num_kernels=11)
        knrm.load_state_dict(torch.load('results/knrm.pt', map_location=device))
        knrm.to(device).eval()
        models_to_test['K-NRM'] = knrm
        print("  Loaded K-NRM    from results/knrm.pt")

    if args.model in ('all', 'lknrm'):
        if not os.path.exists('results/lknrm.pt'):
            print("  ERROR: results/lknrm.pt not found.")
            print("  Run:  python main.py --no-bert")
            sys.exit(1)
        torch.manual_seed(0)
        lknrm = LKNRM(vocab_size=vocab.size, embed_dim=128, num_kernels=11)
        lknrm.load_state_dict(torch.load('results/lknrm.pt', map_location=device))
        lknrm.to(device).eval()
        models_to_test['LK-NRM'] = lknrm
        print("  Loaded LK-NRM   from results/lknrm.pt")

    # Print test topic names
    print(f"\n  Test topics ({len(TEST_TOPICS)} unseen):")
    for i, t in enumerate(TEST_TOPICS, 1):
        print(f"    {i}. {t['query']}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    overall_results   = {}
    per_topic_results = {}

    for name, model in models_to_test.items():
        print(f"\n  Evaluating {name} on test set...")
        overall_results[name]   = evaluate_overall(model, vocab, device)
        per_topic_results[name] = evaluate_per_topic(model, vocab, device)

    # ── Print + plot ───────────────────────────────────────────────────────
    print_overall(overall_results)
    print_per_topic(per_topic_results)
    plot_test_results(overall_results, per_topic_results)

    print("\n  Done! Check results/test_results.png\n")


if __name__ == '__main__':
    main()
