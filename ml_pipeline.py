import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


def build_pipelines():
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                      stop_words='english', min_df=2)),
            ("model", LogisticRegression(max_iter=2000, class_weight='balanced'))
        ]),
        "Decision Tree": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                      stop_words='english')),
            ("model", DecisionTreeClassifier(class_weight='balanced', max_depth=20))
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                      stop_words='english')),
            ("model", RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                              random_state=42))
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, stop_words='english')),
            ("model", MultinomialNB())
        ]),
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                      stop_words='english')),
            ("model", SVC(class_weight='balanced', probability=True))
        ])
    }


def train_and_evaluate(data: pd.DataFrame):
    X = data['statements']
    y = data['BinaryNumTarget']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = build_pipelines()
    results = {}

    for name, pipe in pipelines.items():
        print(f"\n Training: {name}")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        results[name] = acc

    results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    results_df = results_df.sort_values(by="Accuracy", ascending=False)

    best_name = results_df.iloc[0]['Model']
    best_model = pipelines[best_name]

    print(f"\n Best Model: {best_name}")
    print(results_df.to_string(index=False))

    # Save best model
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("Saved best model to models/best_model.pkl")
    return best_model, results_df


def load_best_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    data = pd.read_csv("data/politifact_data.csv")

    print("Columns:", data.columns)
    print("\nVerdict value counts:")
    print(data['verdict'].value_counts())

    # Drop rows where verdict or statements are missing
    data = data.dropna(subset=['verdict', 'statements'])

    # Map verdict labels to binary numeric target
    # 1 = credible (true/mostly-true/half-true), 0 = not credible
    true_labels = {'true', 'mostly-true', 'half-true'}
    data['BinaryNumTarget'] = data['verdict'].str.lower().str.strip().apply(
        lambda x: 1 if x in true_labels else 0
    )

    print(f"\nBinaryNumTarget distribution:")
    print(data['BinaryNumTarget'].value_counts())

    train_and_evaluate(data)