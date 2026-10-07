"""User-based collaborative filtering for restaurant recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from similarity import METRICS


class UserBasedCF:
    def __init__(self, metric: str = "pearson", k: int = 3, normalize: bool = True):
        if metric not in METRICS:
            raise ValueError(f"Unknown metric: {metric}. Choose from {list(METRICS)}")
        self.metric = metric
        self.k = k
        self.normalize = normalize

    def fit(self, ratings: pd.DataFrame):
        required = {"user", "restaurant", "rating"}
        if not required.issubset(ratings.columns):
            raise ValueError(f"ratings must contain columns: {sorted(required)}")

        self.ratings = ratings.copy()
        self.matrix = self.ratings.pivot_table(
            index="user", columns="restaurant", values="rating", aggfunc="mean"
        ).sort_index()

        self.user_mean = self.matrix.mean(axis=1)
        self.user_std = self.matrix.std(axis=1).replace(0, 1.0).fillna(1.0)

        if self.normalize:
            self.work_matrix = self.matrix.sub(self.user_mean, axis=0).div(self.user_std, axis=0)
        else:
            self.work_matrix = self.matrix.copy()

        return self

    def _similarity(self, user_a, user_b) -> float:
        metric = METRICS[self.metric]
        a = self.work_matrix.loc[user_a].to_numpy(dtype=float)
        b = self.work_matrix.loc[user_b].to_numpy(dtype=float)
        return metric(a, b)

    def neighbors(self, user, k: int | None = None) -> pd.DataFrame:
        if user not in self.matrix.index:
            raise KeyError(f"Unknown user: {user}")

        rows = []
        for other in self.matrix.index:
            if other == user:
                continue
            rows.append((other, self._similarity(user, other)))

        result = pd.DataFrame(rows, columns=["neighbor", "similarity"])
        result = result.sort_values("similarity", ascending=False)
        return result.head(k or self.k).reset_index(drop=True)

    def predict(self, user, restaurant) -> float:
        if restaurant not in self.matrix.columns:
            return float(self.user_mean.loc[user])

        neigh = self.neighbors(user, k=len(self.matrix.index) - 1)
        weighted_sum = 0.0
        weight_total = 0.0

        for row in neigh.itertuples(index=False):
            sim = float(row.similarity)
            value = self.work_matrix.loc[row.neighbor, restaurant]
            if pd.isna(value) or sim <= 0:
                continue
            weighted_sum += sim * float(value)
            weight_total += abs(sim)
            if weight_total > 0 and sum(
                1
                for r in neigh.itertuples(index=False)
                if r.similarity > 0
                and not pd.isna(self.work_matrix.loc[r.neighbor, restaurant])
            ) >= self.k:
                break

        if weight_total == 0:
            return float(self.user_mean.loc[user])

        pred = weighted_sum / weight_total
        if self.normalize:
            pred = pred * float(self.user_std.loc[user]) + float(self.user_mean.loc[user])

        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user, n: int = 5) -> pd.DataFrame:
        seen = set(self.matrix.loc[user].dropna().index)
        candidates = [item for item in self.matrix.columns if item not in seen]
        scored = [(item, self.predict(user, item)) for item in candidates]
        return (
            pd.DataFrame(scored, columns=["restaurant", "predicted_rating"])
            .sort_values("predicted_rating", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
