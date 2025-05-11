# Mental Health Sentiment Analysis API

## Overview
The Mental Health Sentiment Analysis API is a RESTful API built with ASP.NET Core that classifies user-submitted statements into mental health categories (e.g. Normal, Anxiety, Depression, Stress, Suicidal, etc.). It uses ML.NET to process [mental health data](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) and train a multiclass classification model on the data. The API accepts a raw text statement via HTTP and returns the predicted mental health status.

I developed this to demonstrate end-to-end ML model training and deployment using C#, ML.NET and ASP.NET Core.

## Features

- **Text Cleaning:** Custom mapping removes URLs, HTML tags, punctuation, digits, bracketed text, and normalizes to lowercase.
- **TF-IDF Featurization:** Tokenizes, removes stop-words, maps tokens to keys, then produces TF-IDF weighted n-grams.
- **Hyperparameter Sweep:** Search over varying L2 regularization strengths with 5-fold cross-validation to find the best multiclass classifier.
- **ML Model:** Trains and saves the classifier using ML.NET, which is loaded by the RESTful API developed in ASP.NET Core.
- **Swagger Integration:** Provides comprehensive API documentation and testing capabilities.

## Prerequisites

- .NET 8.0
- A self-signed certificate to enable HTTPS (optional if using HTTP).
- An API client or Swagger UI (included with the project).

## Setup and Running the API

Build and Run the Project

```bash
dotnet build
dotnet run
```

The server will be hosted at `https://localhost:5001` (or `http://localhost:5000`). You can test the API using tools like Postman or Swagger UI.

## API Usage and Documentation

Once the API is running, you can access the Swagger UI at:

```
https://localhost:5001/swagger
http://localhost:5000/swagger
```

## Architecture

<div align="center">
  <img src="images/uml_diagram.png" alt="UML Diagram" width="80%" />
</div>

## Swagger UI

<div align="center">
  <img src="images/swagger.png" alt="Swagger UI" width="80%" />
</div>
