# Domain classifier

Proposal domains are predicted by a TF-IDF vectorizer and a scikit-learn Random Forest classifier. It trains once on first use from `domain_training.csv` and predicts across the 12 domain labels listed in `main.py`.

The seed data is English-only. For non-English submissions, the app avoids forcing an English-only Random Forest label and uses the multilingual embedding classifier when that model is available.

## Training data

The CSV has `domain`, `text`, and `source` columns. It contains 144 template-generated synthetic examples built from curated terms, with 12 examples per domain. These are prototype examples, not abstracts collected from research papers. Their predictions and confidence scores have not been validated against a human-labeled test set.

To improve the classifier, add representative, human-reviewed examples for each label to the CSV. Keep each `domain` value identical to one of the 12 labels in `main.py`, and include a clear English title or abstract in `text`. Keep any held-out evaluation examples in a separate file so they are not used in training.

Install `requirements.txt` to include scikit-learn. If that dependency or the CSV is unavailable, the backend falls back to the existing multilingual embedding-based classifier when its model is installed.
