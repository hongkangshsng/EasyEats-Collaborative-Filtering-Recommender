[English](README.md) | [繁體中文](README_zh-TW.md)

# EasyEats Collaborative Filtering Recommender

> 2023 Shanghai Jiao Tong University × Microsoft Engage project portfolio  
> A Python-based restaurant recommendation demo centered on collaborative filtering, similarity metrics, rating normalization, KNN-style neighbor selection, and matrix factorization.

## Project Overview

This repository reorganizes the technical topics from my 2023 Engage project **Easy Eat** into a reproducible portfolio project for software, data, machine-learning, and graduate-school applications.

The original project studied how a restaurant recommendation system can infer user preference from historical ratings, identify similar users (neighbors), and rank restaurants for a target user. The project material covered **Collaborative Filtering**, **Jaccard / Cosine / Pearson similarity**, rating statistics and normalization, **KNN**, and **Matrix Factorization / SVD concepts**.

> **Portfolio reconstruction note:** the code in this repository is a clean reimplementation based on the algorithms and workflow documented in the original project material. The included sample ratings are synthetic demonstration data rather than the original training dataset.

## Project Context

- Program: 2023 Microsoft Engage training project
- Collaboration context: Shanghai Jiao Tong University × Microsoft
- Project theme: Easy Eat restaurant recommendation
- Main language: Python
- Core topics: Collaborative Filtering, Similarity, Neighbor/KNN, normalization, Matrix Factorization
- Deliverables: Design Doc, algorithm study, project presentation

## Recommendation Pipeline

```text
User–restaurant ratings
        ↓
Data cleaning / rating statistics
        ↓
Optional per-user normalization
        ↓
Similarity calculation
  Jaccard / Cosine / Pearson / RMSD
        ↓
Top-K neighbor selection
        ↓
Weighted collaborative-filtering score
        ↓
Candidate restaurant ranking
        ↓
Top-N recommendations
```

## Algorithms Implemented

| Topic | Implementation | Purpose |
|---|---|---|
| Collaborative Filtering | User-based CF | Recommend from similar users' preferences |
| Jaccard Similarity | Set overlap of rated items | Measure common rating coverage |
| Cosine Similarity | Vector-angle similarity | Compare rating-direction patterns |
| Pearson Correlation | Mean-centered correlation | Compare relative preference patterns |
| RMSD Similarity | Distance converted to similarity | Compare rating deviation |
| Rating Normalization | Per-user z-score | Reduce differences in personal rating scales |
| KNN-style Neighbor Search | Top-K most similar users | Restrict prediction to relevant neighbors |
| Matrix Factorization | Latent-factor SGD model | Learn compact user/item representations |
| Evaluation | MAE / RMSE helpers | Quantify prediction error |

## Source Code

- [`src/main.py`](src/main.py) — runnable demonstration pipeline
- [`src/similarity.py`](src/similarity.py) — Jaccard, Cosine, Pearson and RMSD similarity
- [`src/recommender.py`](src/recommender.py) — user-based collaborative filtering recommender
- [`src/matrix_factorization.py`](src/matrix_factorization.py) — latent-factor matrix factorization
- [`src/evaluation.py`](src/evaluation.py) — MAE / RMSE utilities
- [`data/sample_ratings.csv`](data/sample_ratings.csv) — synthetic restaurant-rating example
- [`docs/algorithm-notes.md`](docs/algorithm-notes.md) — algorithm notes and formulas

## Quick Start

```bash
git clone https://github.com/hongkangshsng/EasyEats-Collaborative-Filtering-Recommender.git
cd EasyEats-Collaborative-Filtering-Recommender
pip install -r requirements.txt
python src/main.py
```

Optional visualization:

```bash
python src/main.py --save-plots
```

Generated figures are written to `results/`.

## Example Questions This Project Answers

- Which users have the most similar restaurant preferences?
- How do Jaccard, Cosine and Pearson similarity behave differently?
- Can user-specific rating scales be normalized before recommendation?
- How can Top-K neighbors be used to score unseen restaurants?
- How does a latent-factor model provide an alternative to neighborhood-based CF?
- How can recommendation quality be evaluated with MAE / RMSE?

## Engineering Focus

This portfolio version emphasizes separation of concerns:

- similarity functions are isolated from recommendation logic;
- recommendation and evaluation are independent modules;
- sample data is separated from source code;
- the main script demonstrates the full pipeline end to end;
- the README distinguishes original project topics from reconstructed code.

## Repository Structure

```text
EasyEats-Collaborative-Filtering-Recommender/
├── README.md
├── README_zh-TW.md
├── requirements.txt
├── data/
│   └── sample_ratings.csv
├── src/
│   ├── main.py
│   ├── similarity.py
│   ├── recommender.py
│   ├── matrix_factorization.py
│   └── evaluation.py
├── docs/
│   └── algorithm-notes.md
└── results/
    └── README.md
```

## Skills Demonstrated

`Python` · `Collaborative Filtering` · `Recommendation Systems` · `Similarity Metrics` · `KNN` · `Matrix Factorization` · `Data Normalization` · `RMSE` · `Data Analysis`

---

**Author:** 洪鏮展  
**Background:** Electrical Engineering, Fu Jen Catholic University  
**Project area:** Recommendation Systems / Machine Learning / Data Analysis