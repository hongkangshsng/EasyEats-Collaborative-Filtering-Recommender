# Algorithm Notes

This document summarizes the algorithm families studied in the Easy Eat project and implemented in this portfolio reconstruction.

## 1. Collaborative Filtering

User-based collaborative filtering assumes that users with similar historical preferences are useful references for predicting unseen items.

The basic workflow is:

1. construct a user-item rating matrix;
2. compute similarity between the target user and other users;
3. select the Top-K most similar neighbors;
4. combine neighbor ratings into a weighted prediction;
5. rank unseen restaurants by predicted score.

## 2. Jaccard Similarity

Jaccard similarity compares the overlap of items rated by two users:

`J(A,B) = |A ∩ B| / |A ∪ B|`

It measures overlap, not the actual rating values.

## 3. Cosine Similarity

Cosine similarity measures the angle between two rating vectors on co-rated items:

`cos(x,y) = (x · y) / (||x|| ||y||)`

Large positive values indicate similar rating directions.

## 4. Pearson Correlation

Pearson similarity mean-centers the two users before computing correlation. This makes it useful when two users have similar preference patterns but systematically use different rating scales.

## 5. RMSD-based Similarity

RMSD measures the rating-distance between two users on co-rated items. This repository converts the distance to similarity using:

`similarity = 1 / (1 + RMSD)`

This is a practical reconstruction for demonstration; it should not be interpreted as a unique canonical formula from the original project.

## 6. Rating Normalization

A user's ratings can be transformed using a z-score:

`z = (rating - user_mean) / user_std`

This reduces the effect of generous vs. strict raters. The original project material discussed rating distributions, mean, standard deviation, and normalization; this implementation uses per-user z-score normalization as a clean engineering realization of that idea.

## 7. KNN-style Neighbor Selection

KNN is used conceptually by selecting the K users with the highest similarity to the target user. Predictions then use only these nearby users rather than the whole population.

## 8. Matrix Factorization

Matrix factorization approximates the sparse user-item rating matrix by lower-dimensional user and item vectors. The implementation in this repository uses stochastic gradient descent with user/item biases and latent factors.

This provides a second recommendation approach that can capture latent preference structure beyond direct neighborhood similarity.

## 9. Evaluation

Common rating-prediction metrics include:

- **MAE** — mean absolute error
- **RMSE** — root mean squared error

Lower values indicate predictions closer to observed ratings.

## Scope Note

The formulas and code here are intended to make the original project topics reproducible and interview-friendly. They are a portfolio reconstruction, not a claim that every implementation detail exactly matches the original 2023 codebase.
