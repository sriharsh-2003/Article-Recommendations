# Article Recommendation System README

This repository contains code and resources for building an article recommendation system using cosine similarity based on TF-IDF vectors. The recommendation system suggests articles similar to the ones provided as input.

## Table of Contents
- [Introduction](#introduction)
- [What is TF-IDF Vectorizer?](#tf-idf-vectorizer)
- [What is Cosine Similarity?](#cosine-similarity)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Understanding the Dataset](#understanding-the-dataset)
- [Creating Functions for Modeling the Dataset](#creating-functions-for-modeling-the-dataset)
- [ML Model: Cosine Similarity](#ml-model-cosine-similarity)
- [How to Use](#how-to-use)

## Introduction
In this project, we've implemented an article recommendation system using TF-IDF (Term Frequency-Inverse Document Frequency) vectors and cosine similarity. The system takes a set of articles, computes their TF-IDF vectors, and then calculates the cosine similarity between them to provide recommendations.

## What is TF-IDF Vectorizer?
TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical representation of text documents. It captures the importance of words considering their frequency within a document and rarity across the entire corpus.

## What is Cosine Similarity?
Cosine similarity measures the similarity between two vectors by calculating the cosine of the angle between them. It is commonly used in information retrieval and recommendation systems to compare the similarity of documents or items based on their feature vectors.

## Importing Libraries
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

## Importing Dataset
```python
articles_df = pd.read_csv('/content/drive/MyDrive/archive/shared_articles.csv')
```

## Understanding the Dataset
- Various operations to understand the dataset, such as viewing columns, filtering, and retrieving unique values.

## Creating Functions for Modeling the Dataset
- Functions to preprocess data, create TF-IDF vectors, and prepare the dataset for modeling.

## ML Model: Cosine Similarity
- Calculation of cosine similarity between TF-IDF vectors to recommend similar articles.

## How to Use
- To get recommendations for a specific article, use the `get_recommendations()` function, providing the article title, indices, cosine similarity matrix, and metadata.

This repository serves as a basis for building recommendation systems for articles or similar textual content.

Feel free to explore and contribute to this project!

---
*Note: This README.md file serves as a high-level guide. Refer to the provided code for detailed implementation.*
