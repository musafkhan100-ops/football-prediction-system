# ⚽ Football Match Outcome Prediction System

## Overview

This project predicts football match outcomes (**Home Win / Draw / Away Win**) using historical match data, feature engineering, bookmaker odds, and machine learning models.

The project was built to investigate whether machine learning models can outperform bookmaker odds when predicting football match results.

---

## Objectives

* Build a complete football prediction pipeline.
* Engineer meaningful pre-match features from historical data.
* Train and evaluate multiple machine learning models.
* Compare model performance against bookmaker odds.
* Analyze whether machine learning adds predictive value beyond betting markets.

---

## Dataset

The project uses historical football match data covering multiple Premier League seasons.

Each match contains information such as:

* Match date
* Home team
* Away team
* Goals scored
* Match result
* Bookmaker odds
* Match statistics

The target variable is:

* **Home** = Home team wins
* **Draw** = Match ends in a draw
* **Away** = Away team wins

---

## Feature Engineering

The model uses only information available **before a match starts**.

### Team Strength (ELO)

An ELO rating system is maintained for each team.

Features:

* Home ELO
* Away ELO
* ELO Difference

---

### Team Form

Exponential time-decay weighting is used to emphasize recent matches.

Features include:

* Recent points
* Goal difference
* Goals scored
* Goals conceded

---

### Rest and Scheduling

Features:

* Home team days of rest
* Away team days of rest
* Rest difference

---

### Head-to-Head Statistics

Historical performance between two teams:

* Previous meetings
* Average goals
* Average points

---

### Market Features

Bookmaker odds are converted into normalized implied probabilities.

Features:

* Home win probability
* Draw probability
* Away win probability
* Market confidence
* Probability spread

---

## Models Evaluated

The following models were tested:

### Logistic Regression

A linear baseline model used to evaluate whether simple relationships exist within the feature set.

### LightGBM

Gradient boosting model designed for structured tabular data.

### XGBoost

Boosted decision tree model commonly used in predictive analytics competitions.

### Decision Tree

Interpretable baseline model.

---

## Evaluation Methodology

### Time-Based Train/Test Split

To avoid future information leakage:

* First 80% of matches used for training
* Last 20% used for testing

---

### Time-Series Cross Validation

Validation uses time-aware splits rather than random shuffling.

This better reflects real-world forecasting conditions.

---

### Metrics

Models are evaluated using:

* Accuracy
* Log Loss

Accuracy measures prediction correctness.

Log Loss evaluates probability quality and confidence calibration.

---

## Results

### Odds-Only Baseline

Bookmaker implied probabilities provide a strong benchmark.

Approximate performance:

* Accuracy: ~51%
* Log Loss: ~0.995

---

### Machine Learning Models

Observed accuracy range:

* Logistic Regression: ~46–48%
* LightGBM: ~44–45%
* XGBoost: ~41–45%
* Decision Tree: ~37%

---

### Hybrid Model

A weighted combination of:

* Bookmaker probabilities
* Machine learning probabilities

Best configuration:

* 90% bookmaker odds
* 10% LightGBM probabilities

This produced only a marginal improvement in log loss.

---

## Key Findings

### Bookmaker Odds Are Extremely Strong Predictors

The betting market already captures a large amount of available information.

Examples:

* Team strength
* Recent form
* Injuries
* Public sentiment
* Expert analysis

---

### Machine Learning Adds Limited Signal

Models learned meaningful patterns but were unable to significantly outperform bookmaker odds.

The strongest results came from combining market probabilities with machine learning predictions.

---

### Data Quality Matters More Than Model Complexity

The project demonstrates that additional information sources are likely required to improve performance, including:

* Expected Goals (xG)
* Lineups
* Injuries
* Player availability
* Advanced event data

---

## Project Structure

```text
football-prediction-system/
│
├── data/
│   └── raw/
│
├── src/
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── predict.py
│
├── models/
│
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python -m src.model_training data/raw/
```

---

## Prediction

```bash
python -m src.predict data/raw/ "Home Team" "Away Team"
```

---

## Future Improvements

* Expected Goals (xG)
* Injury and lineup information
* Probability calibration improvements
* Model ensembling
* Interactive web application
* Live match prediction dashboard

---

## What I Learned

Through this project I learned:

* Data cleaning and preprocessing
* Feature engineering
* ELO rating systems
* Time-series validation
* Model evaluation
* Probability calibration
* Sports analytics workflows
* The importance of strong real-world baselines

