# Football Match Prediction

This project trains a simple model to predict the outcome of a football match: `home`, `draw`, or `away`.

## Project structure
- `data/raw/` — raw match data CSV files
- `src/` — Python source code
- `requirements.txt` — Python dependencies

## What changed
- The model now uses historical team statistics instead of future match scores.
- The prediction pipeline computes pre-match features from the last 5 matches and overall club history.
- Raw E0 season files are now normalized automatically.
- Model and encoder are saved to `models/` so predictions can be made later.

## How it works
1. `src/model_training.py` reads raw match CSV files or a folder of CSV files.
2. It normalizes raw column names such as `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, and `FTAG`.
3. It cleans the data and adds the target label.
4. The feature builder computes recent and overall team form from previous matches.
5. A `RandomForestClassifier` model is trained and saved.
6. `src/predict.py` loads the saved model and predicts a new match using team names and competition.

## Install dependencies
```bash
pip install -r requirements.txt
```

## Merge raw data
```bash
python -m src.merge_data data/raw/ --output data/raw/merged_matches.csv
```

## Train the model
```bash
python -m src.model_training data/raw/
```

This command automatically creates `data/raw/merged_matches.csv` when the input path is a folder.

## Predict a match
```bash
python -m src.predict data/raw/ "Home Team" "Away Team" --competition "Premier League"
```

## What to expect
- The training script runs 5-fold cross-validation and prints mean accuracy.
- It also evaluates the final model on the last 20% of the dataset.
- Prediction output includes the home team, away team, competition, and probability scores.

## Notes
- The training path can be a single CSV file or a folder containing many CSV files.
- The project will combine all CSV files in the folder automatically.
- The loader now accepts raw E0-style files and normalizes columns.
- The training data must still include: `date`, `home_team`, `away_team`, `home_goals`, `away_goals`.
- If a file has raw E0 naming (like `FTHG` and `FTAG`), the project renames them automatically.
- If a team has no history in the data, the model uses zeros for recent form.
