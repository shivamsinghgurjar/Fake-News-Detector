import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.utils import resample

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def get_binary_target(text):
    if text in ['true', 'mostly-true']:
        return 1
    else:
        return 0


def better_label(x):
    x = str(x).lower()
    if x in ['true', 'mostly-true']:
        return 1
    elif x in ['false', 'pants-fire']:
        return 0
    else:
        return None


def clean_text(text):
    words = str(text).split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    # Drop missing & duplicates
    data = data.dropna()
    data = data.drop_duplicates()

    # Clean text
    data['statements'] = data['statements'].apply(clean_text)

    # Add features
    data['text_length'] = data['statements'].apply(len)
    data['word_count'] = data['statements'].apply(lambda x: len(str(x).split()))

    # Labels
    data['BinaryTarget'] = data['targets'].apply(get_binary_target)
    data['BinaryNumTarget'] = data['BinaryTarget']
    data['label'] = data['targets'].apply(better_label)
    data = data.dropna(subset=['label'])

    # Remove outliers (very long texts)
    data = data[data['text_length'] < 200]

    return data


def balance_data(data: pd.DataFrame) -> pd.DataFrame:
    df_majority = data[data.BinaryNumTarget == 0]
    df_minority = data[data.BinaryNumTarget == 1]

    df_minority_up = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    data_balanced = pd.concat([df_majority, df_minority_up])
    return data_balanced


if __name__ == "__main__":
    data = pd.read_csv("data/politifact_data.csv")
    data = preprocess(data)
    data_balanced = balance_data(data)
    data_balanced.to_csv("data/processed_data.csv", index=False)
    print(f"Processed data: {data_balanced.shape}")
