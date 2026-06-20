# Retraining

`train_models.py` expects three files in a `data/` subfolder next to it:

```bash
mkdir -p data
curl -s -o data/sms_spam.csv "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
curl -s -o data/imdb.csv "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
curl -s -o data/newsgroups.json "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"
```

These aren't bundled in the output since imdb.csv and newsgroups.json together
are ~85MB. Once downloaded, run:

```bash
pip install -r requirements.txt
python3 train_models.py
```

This regenerates `results.json` and the `models/*.joblib` files — copy both
into `backend/` to use the new training run in the dashboard.
