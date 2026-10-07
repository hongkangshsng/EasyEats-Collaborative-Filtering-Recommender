[English](README.md) | **繁體中文**

# EasyEats 協同過濾餐廳推薦系統

> 2023 上海交通大學 × Microsoft Engage 專案作品集  
> 以 Python 重建餐廳推薦流程，涵蓋協同過濾、相似度計算、評分正規化、KNN 鄰居選擇與 Matrix Factorization。

## 專案簡介

本 Repository 將 2023 Engage **Easy Eat** 餐廳推薦專案中的演算法主題重新整理成可執行、可閱讀的工程作品集。

原始專案研究如何利用使用者歷史餐廳評分，計算使用者之間的相似程度，選出鄰居（Neighbor），再依鄰居偏好產生個人化餐廳推薦。專案內容涵蓋 **Collaborative Filtering、Jaccard / Cosine / Pearson Similarity、評分統計與正規化、KNN，以及 Matrix Factorization / SVD 概念**。

> **作品集重建說明：** Repository 中的程式碼是依原始專案所記錄的演算法與流程重新工程化實作；`sample_ratings.csv` 為示範用合成資料，不宣稱為當年原始訓練資料。

## 專案背景

- 計畫：2023 Microsoft Engage 培訓專案
- 合作背景：上海交通大學 × Microsoft
- 專案主題：Easy Eat 餐廳推薦
- 主要語言：Python
- 核心：協同過濾、Similarity、Neighbor/KNN、正規化、Matrix Factorization
- 成果：Design Doc、演算法分析與成果發表

## 推薦系統流程

```text
使用者－餐廳評分
      ↓
資料整理 / 評分統計
      ↓
可選：使用者評分正規化
      ↓
相似度計算
Jaccard / Cosine / Pearson / RMSD
      ↓
Top-K Neighbor 選擇
      ↓
協同過濾加權預測
      ↓
候選餐廳排序
      ↓
Top-N 個人化推薦
```

## 核心演算法

| 主題 | 實作 | 用途 |
|---|---|---|
| Collaborative Filtering | User-based CF | 利用相似使用者偏好推薦餐廳 |
| Jaccard Similarity | 已評分項目集合交集/聯集 | 比較共同評分覆蓋度 |
| Cosine Similarity | 向量夾角 | 比較評分方向相似度 |
| Pearson Correlation | 去平均後相關係數 | 比較相對偏好模式 |
| RMSD Similarity | 將評分距離轉為相似度 | 衡量評分差異 |
| Rating Normalization | 使用者 z-score | 降低不同使用者評分尺度差異 |
| KNN-style Neighbor | 取 Top-K 相似使用者 | 聚焦最相關鄰居 |
| Matrix Factorization | SGD 潛在因子模型 | 學習使用者與餐廳 latent factors |
| Evaluation | MAE / RMSE | 衡量預測誤差 |

## 程式檔案

- [`src/main.py`](src/main.py)：完整示範流程
- [`src/similarity.py`](src/similarity.py)：Jaccard、Cosine、Pearson、RMSD
- [`src/recommender.py`](src/recommender.py)：User-based Collaborative Filtering
- [`src/matrix_factorization.py`](src/matrix_factorization.py)：潛在因子矩陣分解
- [`src/evaluation.py`](src/evaluation.py)：MAE / RMSE
- [`data/sample_ratings.csv`](data/sample_ratings.csv)：合成示範評分資料
- [`docs/algorithm-notes.md`](docs/algorithm-notes.md)：公式與演算法筆記

## 快速執行

```bash
git clone https://github.com/hongkangshsng/EasyEats-Collaborative-Filtering-Recommender.git
cd EasyEats-Collaborative-Filtering-Recommender
pip install -r requirements.txt
python src/main.py
```

若要輸出圖表：

```bash
python src/main.py --save-plots
```

圖表會輸出到 `results/`。

## 此專案可以展示的問題

- 如何判斷哪些使用者的餐廳偏好最相似？
- Jaccard、Cosine、Pearson 三種相似度有何差異？
- 評分習慣不同時，如何先正規化再進行推薦？
- 如何用 Top-K Neighbor 對未評分餐廳產生預測分數？
- Matrix Factorization 如何作為 Neighborhood CF 的另一種推薦方法？
- 如何利用 MAE / RMSE 評估推薦預測？

## 工程化重點

此版本特別將原本專案概念整理為模組化程式：相似度、推薦器、矩陣分解、評估與資料彼此分離，使 GitHub 不只是專題介紹，而是一個可以閱讀與執行的演算法作品。

## 技術關鍵字

`Python` · `Collaborative Filtering` · `Recommendation Systems` · `Jaccard` · `Cosine` · `Pearson` · `KNN` · `Matrix Factorization` · `Normalization` · `RMSE`

---

**作者：** 洪鏮展  
**背景：** 輔仁大學電機工程學系  
**專案領域：** Recommendation Systems / Machine Learning / Data Analysis