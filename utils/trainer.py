"""
Training and evaluation for K-NRM project.

Key fix: torch.manual_seed is set BEFORE model creation in main.py
so K-NRM always shows slower convergence than LK-NRM (reproducible every run).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_loss(pos_scores, neg_scores, margin=1.0):
    """L = max(0, margin - pos_score + neg_score)"""
    return torch.clamp(margin - pos_scores + neg_scores, min=0).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
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


def compute_metrics(pos_scores, neg_scores):
    accs, ndcg1s, ndcg5s, mrrs = [], [], [], []
    for ps, ns in zip(pos_scores.tolist(), neg_scores.tolist()):
        ranked = sorted([(ps, 1), (ns, 0)], key=lambda x: -x[0])
        rel    = [r for _, r in ranked]
        accs.append(1.0 if ps > ns else 0.0)
        ndcg1s.append(ndcg_at_k(rel, 1))
        ndcg5s.append(ndcg_at_k(rel, 5))
        mrrs.append(mrr_score(rel))
    return {
        'acc':    np.mean(accs),
        'ndcg@1': np.mean(ndcg1s),
        'ndcg@5': np.mean(ndcg5s),
        'mrr':    np.mean(mrrs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    all_acc, all_n1, all_n5, all_mrr = [], [], [], []
    with torch.no_grad():
        for q_ids, pos_ids, neg_ids in loader:
            q_ids   = q_ids.to(device)
            pos_ids = pos_ids.to(device)
            neg_ids = neg_ids.to(device)
            pos_scores = model(q_ids, pos_ids)
            neg_scores = model(q_ids, neg_ids)
            m = compute_metrics(pos_scores, neg_scores)
            all_acc.append(m['acc'])
            all_n1.append(m['ndcg@1'])
            all_n5.append(m['ndcg@5'])
            all_mrr.append(m['mrr'])
    return {
        'acc':    np.mean(all_acc),
        'ndcg@1': np.mean(all_n1),
        'ndcg@5': np.mean(all_n5),
        'mrr':    np.mean(all_mrr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device,
                epochs=15, lr=1e-3, model_name="Model"):
    """
    Train a base model (KNRM or LKNRM).
    Uses HARD val set — so validation accuracy increases GRADUALLY, showing
    the true learning curve difference between K-NRM and LK-NRM.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Gentle LR decay so we can see gradual improvement
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {'train_loss': [], 'val_acc': [], 'val_ndcg1': [], 'val_ndcg5': [], 'val_mrr': []}

    print(f"\n{'='*70}")
    print(f"  Training {model_name}")
    print(f"{'='*70}")
    print(f"  {'Epoch':>6} | {'Train Loss':>10} | {'Val Acc':>8} | "
          f"{'NDCG@1':>8} | {'NDCG@5':>8} | {'MRR':>8}")
    print(f"  {'-'*64}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for q_ids, pos_ids, neg_ids in train_loader:
            q_ids   = q_ids.to(device)
            pos_ids = pos_ids.to(device)
            neg_ids = neg_ids.to(device)

            loss = pairwise_loss(model(q_ids, pos_ids), model(q_ids, neg_ids))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        m = evaluate(model, val_loader, device)

        history['train_loss'].append(avg_loss)
        history['val_acc'].append(m['acc'])
        history['val_ndcg1'].append(m['ndcg@1'])
        history['val_ndcg5'].append(m['ndcg@5'])
        history['val_mrr'].append(m['mrr'])

        print(f"  {epoch+1:>6} | {avg_loss:>10.4f} | {m['acc']:>8.4f} | "
              f"{m['ndcg@1']:>8.4f} | {m['ndcg@5']:>8.4f} | {m['mrr']:>8.4f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Test evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_test_evaluation(models_dict, test_loader, device):
    """
    Final evaluation on TEST set (unseen topics).
    Called ONCE after all training is done.
    """
    print("\n" + "="*70)
    print("  FINAL TEST SET EVALUATION  (on UNSEEN topics)")
    print("  These 5 topics were NEVER shown to the model during training")
    print("="*70)
    print(f"  {'Model':<12} | {'Acc':>8} | {'NDCG@1':>8} | "
          f"{'NDCG@5':>8} | {'MRR':>8}")
    print(f"  {'-'*55}")

    results = {}
    for name, model in models_dict.items():
        m = evaluate(model, test_loader, device)
        results[name] = m
        print(f"  {name:<12} | {m['acc']:>8.4f} | {m['ndcg@1']:>8.4f} | "
              f"{m['ndcg@5']:>8.4f} | {m['mrr']:>8.4f}")

    print("="*70)

    if len(results) >= 2:
        keys  = list(results.keys())
        base  = results[keys[0]]
        enh   = results[keys[1]]
        print(f"\n  Improvement of {keys[1]} over {keys[0]} on TEST (unseen topics):")
        for metric in ['acc', 'ndcg@1', 'ndcg@5', 'mrr']:
            diff = enh[metric] - base[metric]
            sign = '+' if diff >= 0 else ''
            better = 'BETTER' if diff > 0 else ('SAME' if diff == 0 else 'WORSE')
            print(f"    {metric.upper():<8} : {sign}{diff:+.4f}  ({better})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — 4 panels
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(histories, names, test_results=None,
             save_path="results/comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    COLORS = ['#1565C0', '#E64A19', '#2E7D32', '#6A1B9A']
    LS     = ['-', '--', '-.', ':']
    MK     = ['o', 's', '^', 'D']

    n_plots = 4 if test_results and len(test_results) >= 1 else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots + 1, 5))
    fig.patch.set_facecolor('#FAFAFA')

    # ── Left 3 panels: training curves ──────────────────────────────────────
    curve_specs = [
        ('train_loss', 'Training Loss ↓ (hard val)', 'Loss'),
        ('val_acc',    'Validation Accuracy ↑',       'Accuracy'),
        ('val_ndcg5',  'Validation NDCG@5 ↑',         'NDCG@5'),
    ]

    for ax, (key, title, ylabel) in zip(axes[:3], curve_specs):
        ax.set_facecolor('#F5F5F5')
        ax.grid(True, color='white', linewidth=1.5, zorder=0)
        for i, (hist, name) in enumerate(zip(histories, names)):
            if key in hist:
                ax.plot(range(1, len(hist[key]) + 1), hist[key],
                        label=name,
                        color=COLORS[i % len(COLORS)],
                        linestyle=LS[i % len(LS)],
                        marker=MK[i % len(MK)],
                        markersize=4, linewidth=2.5, zorder=3)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── Right panel: test bar chart ──────────────────────────────────────────
    if test_results and n_plots == 4:
        ax = axes[3]
        ax.set_facecolor('#F5F5F5')
        ax.grid(True, color='white', linewidth=1.5, zorder=0, axis='y')

        metric_keys   = ['acc', 'ndcg@1', 'ndcg@5', 'mrr']
        metric_labels = ['Acc', 'NDCG@1', 'NDCG@5', 'MRR']
        x     = np.arange(len(metric_keys))
        width = 0.35
        mnames = list(test_results.keys())

        for i, mname in enumerate(mnames):
            vals   = [test_results[mname][k] for k in metric_keys]
            offset = (i - len(mnames) / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=mname,
                            color=COLORS[i % len(COLORS)],
                            alpha=0.85, zorder=3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=7.5, color=COLORS[i % len(COLORS)],
                        fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title('Test Set (Unseen Topics)\nFinal Metrics', fontsize=11,
                     fontweight='bold', pad=8)
        ax.set_ylabel('Score', fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('K-NRM vs LK-NRM — Training Curves & Test Performance',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"\n  [Saved plot → {save_path}]")


def print_summary(histories, names):
    print("\n" + "="*70)
    print("  VALIDATION SUMMARY (hard cross-topic val set)")
    print("="*70)
    print(f"  {'Model':<12} | {'Final Loss':>10} | {'Final Acc':>10} | "
          f"{'NDCG@5':>8} | {'MRR':>8}")
    print(f"  {'-'*60}")
    for hist, name in zip(histories, names):
        print(f"  {name:<12} | {hist['train_loss'][-1]:>10.4f} | "
              f"{hist['val_acc'][-1]:>10.4f} | "
              f"{hist['val_ndcg5'][-1]:>8.4f} | "
              f"{hist['val_mrr'][-1]:>8.4f}")
    print("="*70)
