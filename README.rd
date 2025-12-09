# Cluster-Kmeans-Symnmf
Code to compare the performance of two clustering methods — K-Means and SymNMF — using silhouette scores as the evaluation metric.

## Implementation Details

#### Core algorithm (C) — 
given a dataset, it builds a similarity matrix (how close each pair of points is), a degree matrix (how connected each point is), and a normalized similarity matrix (scales the similarities so they’re comparable). It then applies the Symmetric Non-negative Matrix Factorization method to group similar points together.

#### Python wrapper —
exposes the C implementation as a Python module via the CPython API, allowing the high-performance clustering algorithms to be imported and executed directly within Python analysis scripts without re-implementing the logic.

#### Analysis script —
runs both SymNMF and K-Means on the same dataset and calculates their silhouette scores (a standard way to measure clustering quality). This part uses scikit-learn’s silhouette_score function to provide an objective, quantitative comparison between the two clustering results.

### Allows the following:
Build matrices that represent the relationships between data points.

Group data into clusters using SymNMF.

Group data into clusters using K-Means.

Compare both methods using the same quantitative quality score.

### Takeaways:
- Learned how to connect high-performance C code to Python with the CPython API.

- Gained experience implementing mathematical algorithms for clustering from scratch.

- Practiced designing software that separates fast, low-level computation from high-level analysis.

- Applied real-world machine learning tooling (scikit-learn) to evaluate and validate clustering performance.