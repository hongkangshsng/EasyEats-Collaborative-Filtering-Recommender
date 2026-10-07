"""EasyEats collaborative-filtering portfolio demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recommender import UserBasedCF
from matrix_factorization import MatrixFactorization


def save_plots(cf: UserBasedCF, user: str, recommendations: pd.DataFrame):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    neighbors = cf.neighbors(user)
    plt.figure(figsize=(7, 4))
    plt.bar(neighbors["neighbor"], neighbors["similarity"])
    plt.ylabel("Similarity")
    plt.title(f"Top neighbors for {user} ({cf.metric})")
    plt.tight_layout()
    plt.savefig(out / "neighbor_similarity.png", dpi=180)
    plt.close()

    if not recommendations.empty:
        plt.figure(figsize=(7, 4))
        plt.bar(recommendations["restaurant"], recommendations["predicted_rating"])
        plt.ylabel("Predicted rating")
        plt.ylim(0, 5)
        plt.title(f"EasyEats recommendations for {user}")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out / "recommendations.png", dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="EasyEats recommendation demo")
    parser.add_argument("--user", default="U1")
    parser.add_argument("--metric", choices=["jaccard", "cosine", "pearson", "rmsd"], default="pearson")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    ratings = pd.read_csv(ROOT / "data" / "sample_ratings.csv")

    cf = UserBasedCF(metric=args.metric, k=args.k, normalize=True).fit(ratings)
    print("\n=== Top-K neighbors ===")
    print(cf.neighbors(args.user).to_string(index=False))

    cf_recs = cf.recommend(args.user, n=args.top_n)
    print("\n=== User-based Collaborative Filtering ===")
    print(cf_recs.to_string(index=False))

    mf = MatrixFactorization().fit(ratings)
    print("\n=== Matrix Factorization ===")
    print(mf.recommend(args.user, n=args.top_n).to_string(index=False))

    if args.save_plots:
        save_plots(cf, args.user, cf_recs)
        print("\nSaved figures to:", ROOT / "results")


if __name__ == "__main__":
    main()
