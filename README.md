# Assignment 2: Min-Hashing and Locality Sensitive Hashing

**Course:** CSL7110 — Machine Learning with Big Data  
**Topic:** Document Similarity via Min-Hashing & LSH

---

## Problem Statement

Implement and evaluate Min-Hashing and Locality Sensitive Hashing (LSH) for approximate similarity search on both text documents and the MovieLens-100K dataset.

## Structure

| File | Description |
|------|-------------|
| `assign.ipynb` | Main notebook — all 5 questions with code, outputs, and analysis |
| `M25CSA010_CSL7110_Assignment` | PDF report |
| `screenshots/` | Output screenshots used in the report |
| `D1.txt`–`D4.txt` | Input documents for Questions 1–3 |
| `ml-100k/` | MovieLens 100K dataset (943 users, 1682 movies) |

## Questions & Results

### Q1 — K-Gram Construction & Jaccard Similarity (20 pts)

Built character 2-grams, character 3-grams, and word 2-grams for four documents. Computed all 18 Jaccard similarity values across 6 document pairs × 3 gram types.

- D1–D2 are near-duplicates: J = 0.98 (char 3-gram), 0.94 (word 2-gram)
- D4 (different topic) has low similarity with all others: J ≈ 0.01–0.31
- Character 2-grams yield the highest similarity; word 2-grams the lowest

### Q2 — MinHash Approximation (20 pts)

Estimated Jaccard similarity for D1–D2 using MinHash with t = 20, 60, 150, 300, 600 hash functions. Universal hash family: h(x) = (ax + b) mod 100003.

| t | Approx. Jaccard | Absolute Error |
|---|-----------------|----------------|
| 20 | 1.0000 | 0.0220 |
| 60 | 0.9667 | 0.0113 |
| 150 | 0.9933 | 0.0154 |
| 300 | 0.9767 | 0.0013 |
| 600 | 0.9800 | 0.0020 |

**Best t ≈ 150** — error drops below 2% with fast computation. Beyond t = 300, diminishing returns.

### Q3 — LSH with Banding (20 pts)

For t = 160 and threshold τ = 0.7, enumerated all factor pairs of 160 to find optimal banding.

- **Selected:** b = 8, r = 20 → s* = 0.6877 (closest to τ)
- D1–D2: P(candidate) ≈ 1.0 (correctly detected)
- D1–D4, D2–D4: P(candidate) ≈ 0.001 (correctly rejected)
- S-curve provides sharp separation at the threshold

### Q4 — MinHash on MovieLens (20 pts)

Computed exact Jaccard for all 444,153 user pairs. Found **10 pairs with J ≥ 0.5** (highest: users 408–898 at J = 0.84). Evaluated MinHash with t = 50, 100, 200 over 5 runs.

| t | Avg FP | Avg FN | Avg Time |
|---|--------|--------|----------|
| 50 | 149.6 | 3.2 | 2.11s |
| 100 | 39.0 | 2.8 | 3.73s |
| 200 | 7.6 | 2.2 | 6.97s |

### Q5 — LSH on MovieLens (20 pts)

Tested 4 configurations across thresholds τ = 0.6 and τ = 0.8, averaged over 5 runs.

**τ = 0.6** (3 exact pairs):

| Config | t | r | b | Avg FP | Avg FN |
|--------|---|---|---|--------|--------|
| 1 | 50 | 5 | 10 | 729.2 | 0.0 |
| 2 | 100 | 5 | 20 | 1441.0 | 0.0 |
| 3 | 200 | 5 | 40 | 2675.0 | 0.0 |
| 4 | 200 | 10 | 20 | 5.6 | 1.2 |

**τ = 0.8** (1 exact pair):

| Config | t | r | b | Avg FP | Avg FN |
|--------|---|---|---|--------|--------|
| 1 | 50 | 5 | 10 | 731.2 | 0.0 |
| 2 | 100 | 5 | 20 | 1443.0 | 0.0 |
| 3 | 200 | 5 | 40 | 2677.0 | 0.0 |
| 4 | 200 | 10 | 20 | 6.4 | 0.0 |

**Config 4 (t=200, r=10, b=20)** is the only practical choice — FP drops from thousands to single digits. Configs 1–3 with small r=5 are too permissive.

## Key Takeaways

1. MinHash accuracy scales as ~1/√t — doubling accuracy costs 4× computation
2. LSH banding is highly sensitive to the r/b ratio: small r → high recall but massive FP; large r → precise but risks FN on borderline pairs
3. The S-curve f(s) = 1 − (1 − s^b)^r governs the precision-recall tradeoff and must be tuned to the target threshold
4. On real data (MovieLens), borderline pairs (J ≈ threshold) are the primary source of both FP and FN

## How to Run

```bash
# Install dependencies
pip install numpy matplotlib

# Run the notebook
jupyter notebook assign.ipynb
```

Execute cells sequentially. MovieLens data is auto-downloaded if not present.

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
