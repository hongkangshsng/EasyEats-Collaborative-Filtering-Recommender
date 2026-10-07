"""Simple matrix-factorization recommender using SGD."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MatrixFactorization:
    def __init__(self, factors=8, epochs=250, lr=0.01, reg=0.02, seed=42):
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, ratings: pd.DataFrame):
        rng = np.random.default_rng(self.seed)
        users = sorted(ratings["user"].unique())
        items = sorted(ratings["restaurant"].unique())
        self.user_to_i = {u: i for i, u in enumerate(users)}
        self.item_to_i = {m: i for i, m in enumerate(items)}
        self.i_to_item = {i: m for m, i in self.item_to_i.items()}
        self.global_mean = float(ratings["rating"].mean())

        self.P = rng.normal(0, 0.1, (len(users), self.factors))
        self.Q = rng.normal(0, 0.1, (len(items), self.factors))
        self.bu = np.zeros(len(users))
        self.bi = np.zeros(len(items))

        triples = list(ratings[["user", "restaurant", "rating"]].itertuples(index=False, name=None))

        for _ in range(self.epochs):
            rng.shuffle(triples)
            for user, item, rating in triples:
                u = self.user_to_i[user]
                i = self.item_to_i[item]
                pred = self.global_mean + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err = float(rating) - pred

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                p_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_old - self.reg * self.Q[i])

        self.ratings = ratings.copy()
        return self

    def predict(self, user, restaurant) -> float:
        if user not in self.user_to_i or restaurant not in self.item_to_i:
            return self.global_mean
        u = self.user_to_i[user]
        i = self.item_to_i[restaurant]
        pred = self.global_mean + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user, n=5) -> pd.DataFrame:
        seen = set(self.ratings.loc[self.ratings["user"] == user, "restaurant"])
        scored = [
            (item, self.predict(user, item))
            for item in self.item_to_i
            if item not in seen
        ]
        return (
            pd.DataFrame(scored, columns=["restaurant", "predicted_rating"])
            .sort_values("predicted_rating", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
