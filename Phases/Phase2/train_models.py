from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from preprocess import load_data, build_preprocessor, split_data

MODELS = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVM": LinearSVC()
}

def train():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    trained_models = {}

    for name, model in MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("tfidf", build_preprocessor()),
                ("classifier", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

        print(f"[+] Trained {name}")

    return trained_models, X_test, y_test

if __name__ == "__main__":
    train()
