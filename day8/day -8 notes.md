# Day 8 Notes - Telecom Churn, Probability Calibration & Unsupervised Learning

---

# Telecom Churn Prediction

## What is Churn?

Churn occurs when a customer stops using a company's service.

Examples:

* Cancelling a mobile subscription
* Switching to another telecom provider
* Discontinuing a service

---

## Telecom Churn Use Case

Objective:

```text id="y21b6x"
Predict which customers
are likely to leave
```

Benefits:

* Customer retention
* Personalized offers
* Reduced revenue loss

---

## Telecom Ecosystem Example

Telecom companies:

* Airtel
* Jio
* Vodafone Idea

manage customer relationships and services.

Telecom infrastructure companies such as:

[Indus Towers](https://www.industowers.com?utm_source=chatgpt.com)

primarily manage and maintain telecom tower infrastructure used by service providers.

---

# Probability Calibration

## What is Probability Calibration?

Probability calibration ensures that predicted probabilities match actual observed outcomes.

Example:

Suppose a model predicts:

```text id="xvdl4k"
80% chance of churn
```

Ideally:

```text id="7lklgo"
Around 80 out of 100
similar customers should churn
```

---

## Why Calibration Matters?

Some models produce:

* Overconfident probabilities
* Underconfident probabilities

Calibration improves reliability.

---

## Applications

* Medical Diagnosis
* Credit Risk
* Fraud Detection
* Churn Prediction

---

# Unsupervised Learning

Unsupervised Learning learns patterns from unlabeled data.

Training Data:

```text id="a4byyo"
Input Data Only
(No Labels)
```

Goal:

Discover hidden structures and relationships.

---

# Major Tasks in Unsupervised Learning

## 1. Clustering

Group similar observations together.

Examples:

* Customer Segmentation
* Market Analysis
* Image Grouping

---

## 2. Dimensionality Reduction

Reduce the number of features while preserving information.

Examples:

* PCA
* SVD
* Autoencoders

Benefits:

* Faster computation
* Better visualization
* Reduced noise

---

## 3. Anomaly Detection

Identify unusual observations.

Examples:

* Fraud Detection
* Network Intrusion Detection
* Equipment Failure Detection

---

## 4. Association Rule Mining

Find items frequently purchased together.

Examples:

```text id="efqvsi"
Tea
Coffee
Milk
```

or

```text id="h5lfui"
Bread
Butter
Jam
```

Applications:

* Market Basket Analysis
* Recommendation Systems

Popular Algorithm:

```text id="63jyo0"
Apriori Algorithm
```

---

# Clustering

Clustering groups similar data points into clusters.

Goal:

```text id="slohyv"
Maximize Similarity
Within Clusters

Minimize Similarity
Between Clusters
```

---

## Good Clustering

Characteristics:

* Points inside a cluster are close together.
* Clusters are well separated.
* Compact clusters are preferred.

---

## Why Smaller Cluster Radius Is Better?

Smaller radius indicates:

* Higher similarity
* Better compactness
* Stronger cluster quality

A compact cluster generally contains more similar observations.

---

# K-Means Clustering

One of the most popular clustering algorithms.

---

## How K-Means Works

Step 1:

Choose K cluster centers.

↓

Step 2:

Assign points to nearest centroid.

↓

Step 3:

Recompute centroids.

↓

Step 4:

Repeat until convergence.

---

# Choosing the Best K

Selecting the correct number of clusters is important.

---

## Elbow Method

Plot:

```text id="0ub3du"
K
vs
WCSS
```

Where:

WCSS = Within Cluster Sum of Squares

Choose the elbow point where improvement begins to slow.

---

## Silhouette Score

Measures cluster quality.

Range:

```text id="e3nzyt"
-1 to +1
```

Interpretation:

| Score   | Meaning              |
| ------- | -------------------- |
| Near +1 | Excellent Clustering |
| Near 0  | Overlapping Clusters |
| Below 0 | Poor Clustering      |

Higher is better.

---

## Davies-Bouldin Index

Measures:

* Cluster compactness
* Cluster separation

Characteristics:

```text id="5d1d4u"
Lower Value = Better
```

---

## Calinski-Harabasz Index

Also called:

```text id="3yxtjw"
Variance Ratio Criterion
```

Measures:

```text id="ltn2zc"
Between Cluster Dispersion
vs
Within Cluster Dispersion
```

Characteristics:

```text id="mw5zzf"
Higher Value = Better
```

---

# Variance Ratio Criterion

Another name for:

```text id="ib2h32"
Calinski-Harabasz Score
```

A larger value indicates:

* Better separation
* Better clustering quality

---

# K-Means Assumptions

K-Means works best when certain assumptions hold.

---

## Spherical Clusters

Assumes clusters are roughly circular (2D) or spherical (higher dimensions).

Example:

```text id="4h8e5v"
○     ○     ○
```

---

## Similar Cluster Sizes

Works best when clusters have similar sizes and densities.

---

## Must Choose K

User must specify:

```text id="nxhnwl"
Number of Clusters (K)
```

before training.

---

## Sensitive to Outliers

Outliers can pull centroids away from their ideal locations.

Result:

* Poor cluster quality
* Incorrect centroids

---

# Failure Cases of K-Means

K-Means struggles when:

* Clusters have irregular shapes
* Clusters have different densities
* Many outliers exist
* True K is unknown

---

# DBSCAN

Density-Based Spatial Clustering of Applications with Noise

A density-based clustering algorithm.

---

# Key Idea

Clusters are formed from dense regions of data.

Sparse regions are treated as noise.

---

## Advantages of DBSCAN

### No Need to Specify K

Unlike K-Means:

```text id="p9xjmb"
No predefined K required
```

---

### Handles Outliers Naturally

Noise points are automatically identified.

---

### Detects Arbitrary Shapes

Can identify:

```text id="czj4na"
Circular Clusters
Elongated Clusters
Irregular Clusters
```

---

### Works Well with Noise

More robust than K-Means in noisy datasets.

---

# DBSCAN Parameters

## Epsilon (ε)

Defines neighborhood radius.

---

## MinPts

Minimum number of neighboring points required to form a cluster.

---

# Types of Points in DBSCAN

## Core Point

Has sufficient neighboring points.

---

## Border Point

Near a core point but lacks enough neighbors itself.

---

## Noise Point

Does not belong to any cluster.

Outlier.

---

# K-Means vs DBSCAN

| K-Means                    | DBSCAN                    |
| -------------------------- | ------------------------- |
| Centroid-Based             | Density-Based             |
| Requires K                 | No K Required             |
| Sensitive to Outliers      | Handles Outliers          |
| Assumes Spherical Clusters | Handles Arbitrary Shapes  |
| Faster on Large Data       | Better for Complex Shapes |

---

# Key Takeaways

* Churn prediction helps retain customers.
* Probability calibration improves confidence estimates.
* Unsupervised learning works with unlabeled data.
* Main tasks include clustering, dimensionality reduction, anomaly detection, and association analysis.
* K-Means is simple and efficient for clustering.
* Elbow, Silhouette, Davies-Bouldin, and Calinski-Harabasz help evaluate clusters.
* K-Means assumes spherical clusters and is sensitive to outliers.
* DBSCAN is density-based and can identify noise points.
* DBSCAN works well for irregularly shaped clusters and noisy datasets.

---
