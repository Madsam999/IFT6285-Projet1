import argparse, numpy as np, os
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def load_ds(task):
    if task == "sst2":
        dataset_id = "stanfordnlp/sst2"
        X_col = "sentence"
        y_col = "label"
    elif task == "mr":
        dataset_id = "ajaykarthick/imdb-movie-reviews"
        X_col = "review"
        y_col = "label"
        
    ds = load_dataset(dataset_id)
    X_train = [ex[X_col] for ex in ds["train"]]
    y_train = np.array([ex[y_col] for ex in ds["train"]], dtype=int)
    X_test   = [ex[X_col] for ex in ds["test"]]
    y_test   = np.array([ex[y_col] for ex in ds["test"]], dtype=int)
    return X_train, y_train, X_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument("max_features", type=int, default=40000)
    parser.add_argument("cv_folds", type=int, default=5)
    parser.add_argument("save_preds", type=str, default="", help="Path to save validation predictions (csv).")
    parser.add_argument("seed", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_ds(args.task)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=args.max_features,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=args.seed
        )),
    ])

    # Hyperparameters grid
    param_grid = {
        "tfidf__ngram_range": [(1,1), (1,2)],
        "clf__C": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    }

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    gs = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print("\nBest params:", gs.best_params_)
    print("CV best F1:  %.4f" % gs.best_score_)

    # Evaluate on validation
    y_pred = gs.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    print("\n==== Test Results (SST-2) ====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    if args.save_preds:
        import csv
        os.makedirs(os.path.dirname(args.save_preds), exist_ok=True) if os.path.dirname(args.save_preds) else None
        with open(args.save_preds, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "y_true", "y_pred"])
            for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
                w.writerow([i, int(yt), int(yp)])
        print(f"\nSaved predictions to: {args.save_preds}")

if __name__ == "__main__":
    main()
