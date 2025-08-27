# Similarity Metrics

In this repository are stored a bunch of utilities for calculating the similarity between two datapoints

## Applications

In many different applications, embeddings are created, wherein complex data are mapped to vectors. The distances between these vectors is then used in decision-making (for example, k nearest neighbours).

This project should help the reader build an intuition to bridge the gap between geometric grounding and real-world utility.

## Metrics

1. Euclidean Distance

The most commonly accepted formula for distance. Finds the continuous amount of space between two points. Given by the formula:

$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$

Here are the formulas using single dollar signs:

## 2. Manhattan Distance

Also known as taxicab or L1 distance. Measures the sum of absolute differences between coordinates, like navigating city blocks:

$d = |x_2 - x_1| + |y_2 - y_1|$

For n-dimensional space:

$d = \sum_{i=1}^{n} |x_i - y_i|$

## 3. Cosine Similarity

Measures the cosine of the angle between two vectors, focusing on orientation rather than magnitude:

$\text{cosine similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \times ||\mathbf{B}||} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$

## 4. Minkowski Distance

A generalized distance metric that includes both Euclidean and Manhattan as special cases:

$d = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$

Where:

- $p = 1$: Manhattan distance
- $p = 2$: Euclidean distance  
- $p = \infty$: Chebyshev distance (maximum difference)
