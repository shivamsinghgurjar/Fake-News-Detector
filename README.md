# 🔍 Fake News Detector

ML + Deep Learning + Google Fact Check API

## Project Structure

```
fake_news_detector/
├── app.py              ← Streamlit web app (simple white UI)
├── pipeline.py         ← Full training pipeline (run once)
├── requirements.txt
├── data/
│   ├── politifact_data.csv    ← raw scraped data
│   └── processed_data.csv     ← cleaned & balanced data
├── models/
│   ├── best_model.pkl         ← best ML model
│   ├── ann_model.h5           ← trained ANN
│   ├── rnn_model.h5           ← trained RNN (LSTM)
│   └── tokenizer.pkl          ← Keras tokenizer
└── src/
    ├── scraper.py      ← PolitiFact web scraper
    ├── preprocess.py   ← cleaning, labelling, balancing
    ├── ml_pipeline.py  ← LR, DT, RF, NB, SVM + best model
    ├── dl_pipeline.py  ← ANN + RNN (LSTM)
    ├── fact_check.py   ← Google Fact Check API
    └── activations.py  ← Sigmoid, ReLU, Tanh, Softmax plots
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Pipeline (train models)

```bash
# Use existing CSV data:
python pipeline.py

# OR scrape fresh data (slow, ~300 pages):
python pipeline.py --scrape --pages 300
```

## Launch App

```bash
streamlit run app.py
```

## Notes

- Place your raw CSV at `data/politifact_data.csv` before running the pipeline.
- The app auto-loads models from `models/` at startup.
- API key is in `src/fact_check.py` — replace with your own if needed.
