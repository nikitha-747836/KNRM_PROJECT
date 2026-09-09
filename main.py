"""
main.py — Train K-NRM and LK-NRM with fixed seeds,
          hard validation set, and proper test evaluation.

KEY FIX: torch.manual_seed() is called BEFORE each model is created,
so results are REPRODUCIBLE every single run. K-NRM always converges
slower than LK-NRM because:
  - K-NRM  uses fixed kernels + simple sum  (less capacity)
  - LK-NRM uses learnable kernels + CNN     (more capacity)

Usage:
  python main.py --no-bert        # train K-NRM + LK-NRM (recommended)
  python main.py --epochs 15      # more epochs
  python main.py --model knrm     # only K-NRM
  python main.py --model lknrm    # only LK-NRM
"""

import sys, os, argparse, pickle
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.knrm          import KNRM
from models.enhanced_knrm import LKNRM, CKNRM
from utils.dataset        import get_all_loaders, get_bert_loaders
from utils.trainer        import (train_model, run_test_evaluation,
                                  plot_all, print_summary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',       default='all',
                   choices=['all', 'knrm', 'lknrm', 'cknrm'])
    p.add_argument('--no-bert',     action='store_true')
    p.add_argument('--epochs',      type=int, default=15)
    p.add_argument('--bert-epochs', type=int, default=5)
    p.add_argument('--batch-size',  type=int, default=32)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device : {device}")

    os.makedirs('results', exist_ok=True)

    histories      = []
    names          = []
    trained_models = {}

    # ── Load data ────────────────────────────────────────────────────────────
    if args.model in ('all', 'knrm', 'lknrm'):
        vocab, train_loader, val_loader, test_loader = \
            get_all_loaders(args.batch_size)
        with open('results/vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)

    # ── 1. K-NRM  (original paper) ───────────────────────────────────────────
    if args.model in ('all', 'knrm'):
        # Fixed seed: ensures K-NRM ALWAYS starts from same random weights
        torch.manual_seed(0)
        knrm = KNRM(vocab_size=vocab.size, embed_dim=128,
                    num_kernels=11, sigma=0.1).to(device)

        h = train_model(knrm, train_loader, val_loader, device,
                        epochs=args.epochs, lr=5e-4, model_name="K-NRM")
        histories.append(h)
        names.append("K-NRM")
        torch.save(knrm.state_dict(), 'results/knrm.pt')
        trained_models['K-NRM'] = knrm
        print("  [Saved → results/knrm.pt]")

    # ── 2. LK-NRM  (learnable kernels + CNN) ────────────────────────────────
    if args.model in ('all', 'lknrm'):
        # Same seed as K-NRM for fair comparison of architecture, not luck
        torch.manual_seed(0)
        lknrm = LKNRM(vocab_size=vocab.size, embed_dim=128,
                      num_kernels=11).to(device)

        h = train_model(lknrm, train_loader, val_loader, device,
                        epochs=args.epochs, lr=5e-4, model_name="LK-NRM")
        histories.append(h)
        names.append("LK-NRM")
        torch.save(lknrm.state_dict(), 'results/lknrm.pt')
        trained_models['LK-NRM'] = lknrm
        print("  [Saved → results/lknrm.pt]")

    # ── 3. C-KNRM  (BERT + learnable kernels + CNN) ─────────────────────────
    if args.model in ('all', 'cknrm') and not args.no_bert:
        print("\n  Loading DistilBERT (first run ~250MB download)...")
        try:
            from transformers import DistilBertTokenizer
            tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            tr_l, va_l, te_l = get_bert_loaders(tok, batch_size=16)

            torch.manual_seed(0)
            cknrm = CKNRM(num_kernels=11, bert_model='distilbert-base-uncased',
                          freeze_bert=True).to(device)

            # C-KNRM uses its own BERT loaders so import bert trainer inline
            from utils.trainer import pairwise_loss
            import torch.nn as nn
            import numpy as np

            optimizer = torch.optim.AdamW(cknrm.parameters(), lr=2e-5)
            hist_c = {'train_loss': [], 'val_acc': [], 'val_ndcg5': [],
                      'val_ndcg1': [], 'val_mrr': []}

            print(f"\n{'='*70}")
            print("  Training C-KNRM")
            print(f"{'='*70}")
            print(f"  {'Epoch':>6} | {'Train Loss':>10} | {'Val Acc':>8} | "
                  f"{'NDCG@1':>8} | {'NDCG@5':>8} | {'MRR':>8}")
            print(f"  {'-'*64}")

            from utils.trainer import compute_metrics
            for epoch in range(args.bert_epochs):
                cknrm.train()
                total = 0.0
                for batch in tr_l:
                    qi, qm, pi, pm, ni, nm = [b.to(device) for b in batch]
                    loss = pairwise_loss(cknrm(qi, qm, pi, pm),
                                        cknrm(qi, qm, ni, nm))
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(cknrm.parameters(), 1.0)
                    optimizer.step()
                    total += loss.item()

                avg = total / len(tr_l)
                cknrm.eval()
                all_m = {'acc': [], 'ndcg@1': [], 'ndcg@5': [], 'mrr': []}
                with torch.no_grad():
                    for batch in va_l:
                        qi, qm, pi, pm, ni, nm = [b.to(device) for b in batch]
                        m = compute_metrics(cknrm(qi, qm, pi, pm),
                                            cknrm(qi, qm, ni, nm))
                        for k in all_m:
                            all_m[k].append(m[k])
                vm = {k: np.mean(v) for k, v in all_m.items()}

                hist_c['train_loss'].append(avg)
                hist_c['val_acc'].append(vm['acc'])
                hist_c['val_ndcg1'].append(vm['ndcg@1'])
                hist_c['val_ndcg5'].append(vm['ndcg@5'])
                hist_c['val_mrr'].append(vm['mrr'])
                print(f"  {epoch+1:>6} | {avg:>10.4f} | {vm['acc']:>8.4f} | "
                      f"{vm['ndcg@1']:>8.4f} | {vm['ndcg@5']:>8.4f} | "
                      f"{vm['mrr']:>8.4f}")

            histories.append(hist_c)
            names.append("C-KNRM")
            torch.save(cknrm.state_dict(), 'results/cknrm.pt')
            print("  [Saved → results/cknrm.pt]")

        except Exception as e:
            print(f"\n  [WARNING] C-KNRM skipped: {e}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_results = None
    if trained_models and args.model in ('all', 'knrm', 'lknrm'):
        test_results = run_test_evaluation(trained_models, test_loader, device)

    # ── Summary + plot ────────────────────────────────────────────────────────
    if histories:
        print_summary(histories, names)
        plot_all(histories, names,
                 test_results=test_results,
                 save_path="results/comparison.png")

    print("\n  Done! Files saved in results/:")
    print("    knrm.pt, lknrm.pt  — model weights")
    print("    vocab.pkl          — vocabulary")
    print("    comparison.png     — 4-panel plot\n")


if __name__ == '__main__':
    main()
