# Movie-to-Book Recommendation System 

## Overview
This project implements an AI/ML-powered recommendation system that suggest books similar to a given movie. 
The system addresses a business need for cross-media content discovery by bridging movie and book datasets using Natural
Language Processing (NLP) and semantic similarity. 

The solution uses a pretrained transformer model to generate text embeddings and cosine similarity to identify relevant
book recommendations.

## Business Problem 
Viewers often want to read books similar to the movies they enjoy, but existing platforms do not provide cross-domain 
recommendations. This project demonstrates how AI/ML can improve user engagement and content discovery. 

## Dataset 
- **Movies:** TMDB 5000 Movie Dataset (Kaggle)
- **Books:** GoodBooks - 10k Dataset (Kaggle)

## How to Run the project 
### Install Dependencies 
```code 
pip install -r requirements.txt
```
### Run the application 
```code
python main.py
``` 

## Expected Output 
- Cleaned movie and book dataset 
- Generated text embeddings 
- Book recommendations for sample movie inputs 
- Model evaluation metrics printed to console
