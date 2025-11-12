import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_score
from sklearn.linear_model import LogisticRegression # New import for LR
import json

# Import the multi-label strategy
from skmultilearn.problem_transform import ClassifierChain # New import for CC
# NOTE: While ClassifierChain is also in sklearn.multioutput, the skmultilearn version 
# is often preferred for more robust multi-label problem transformation strategies.

# --- CONFIGURATION (Keep the same) ---
PATH_TO_DATA = "./Datasets/"
TRAIN_PATH = PATH_TO_DATA + "train.tsv"
TEST_PATH = PATH_TO_DATA + "test.tsv"
VALIDATION_PATH = PATH_TO_DATA + "dev.tsv"
topic_num_map = {"cs.it": 0, "math.it": 1, "cs.lg": 2, "cs.ai": 3, "stat.ml": 4, "cs.ds": 5, "cs.si": 6, "cs.dm": 7, "physics.soc-ph": 8, "cs.lo": 9, "math.co": 10, "cs.cc": 11, "math.oc": 12, "cs.ni": 13, "cs.cv": 14, "cs.cl": 15, "cs.cr": 16, "cs.sy": 17, "cs.dc": 18, "cs.ne": 19, "cs.ir": 20, "quant-ph": 21, "cs.gt": 22, "cs.cy": 23, "cs.pl": 24, "cs.se": 25, "math.pr": 26, "cs.db": 27, "cs.cg": 28, "cs.na": 29, "cs.hc": 30, "math.na": 31, "cs.ce": 32, "cs.ma": 33, "cs.ro": 34, "cs.fl": 35, "math.st": 36, "stat.th": 37, "cs.dl": 38, "cmp-lg": 39, "cs.mm": 40, "cond-mat.stat-mech": 41, "cs.pf": 42, "math.lo": 43, "stat.ap": 44, "cs.ms": 45, "stat.me": 46, "cs.sc": 47, "cond-mat.dis-nn": 48, "q-bio.nc": 49, "physics.data-an": 50, "nlin.ao": 51, "q-bio.qm": 52, "math.nt": 53}
num_topic_map = {v: k for k, v in topic_num_map.items()}
NUM_LABELS = len(topic_num_map) # Get the total number of unique labels

class LogisticRegression_Predictor:
    # We remove k and s as they are MLkNN hyperparameters
    def __init__(self, C=1.0, penalty='l2', solver='liblinear'):
        # Hyperparameters for Logistic Regression
        self.C = C
        self.penalty = penalty
        self.solver = solver

        # Model and Data Components
        self.vectorizer = None
        self.classifier = None # This will hold the ClassifierChain model
        self.X_train_raw = None
        self.X_train_sparse = None
        self.Y_train = None

    # --- 1. Data Loading and Preparation (Identical to previous) ---
    def load_data(self, file_path):
        """Loads data, separates text (X) and converts label string (Y) to matrix."""
        try:
            df = pd.read_csv(file_path, sep="\t", header=None, names=['labels_str', 'text'])
            
            X_raw = df['text'].to_numpy()
            # Ensure the conversion handles the labels as a matrix of 0s and 1s,
            # padded to the total number of labels if necessary.
            # Assuming the input format is a string of 1s/0s for all labels (e.g., "01001...")
            label_lists = [[int(char) for char in label_str] for label_str in df['labels_str']]
            
            # Pad the label lists to ensure all samples have the same length (NUM_LABELS)
            Y = np.array([lst + [0] * (NUM_LABELS - len(lst)) for lst in label_lists])

            return X_raw, Y
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Please check the path.")
            return None, None

    def createTrainDataset(self, train_path):
        """Loads and stores the training data."""
        X_train_raw, Y_train = self.load_data(train_path)
        
        if X_train_raw is not None:
            self.X_train_raw = X_train_raw
            self.Y_train = Y_train
            print(f"Training set loaded with {self.Y_train.shape[0]} samples.")
        
    # --- 2. Feature Extraction (Identical to previous) ---
    def fit_vectorizer(self):
        """Fits the TF-IDF vectorizer and transforms the training data."""
        if self.X_train_raw is None:
            raise ValueError("Training data must be loaded first! Call createTrainDataset().")

        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=5) 
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train_raw)
        self.X_train_sparse = csr_matrix(X_train_tfidf)
        
        print(f"Features vectorized. X_train sparse shape: {self.X_train_sparse.shape}")

    # --- 3. Classifier Chain Training ---
    def train_model(self):
        """Initializes and trains the Logistic Regression Classifier Chain."""
        if self.X_train_sparse is None:
            raise ValueError("Data must be vectorized first! Call fit_vectorizer().")
        
        # 1. Define the base estimator (Logistic Regression)
        base_lr = LogisticRegression(
            C=self.C, 
            penalty=self.penalty, 
            solver=self.solver, 
            random_state=42, 
            max_iter=500 # Increased max_iter for convergence
        )

        # 2. Wrap the base estimator in the Classifier Chain strategy
        self.classifier = ClassifierChain(
            base_lr, 
            order=np.arange(self.Y_train.shape[1]), # Use default label order [0, 1, 2, ...]
        )
        
        print(f"\nStarting Classifier Chain Training (Base Estimator: Logistic Regression)...")
        
        # Train the Classifier Chain
        # The chain handles training a separate LR for each label, sequentially.
        self.classifier.fit(self.X_train_sparse, self.Y_train)
        
        print("Classifier Chain Training Complete.")

    # --- 4. Prediction and Evaluation (Modified for sparse prediction handling) ---
    def predict_and_evaluate(self, test_path, outPath):
        """Loads test data, predicts, and reports metrics."""
        
        X_test_raw, Y_test = self.load_data(test_path)
        
        if X_test_raw is None or self.classifier is None:
            print("Cannot run prediction: Test data or trained model missing.")
            return

        # Transform test data using the FITTED vectorizer
        X_test_tfidf = self.vectorizer.transform(X_test_raw)
        X_test_sparse = csr_matrix(X_test_tfidf)
        
        # Make predictions (returns a sparse matrix)
        predictions_sparse = self.classifier.predict(X_test_sparse)
        # Convert sparse predictions to dense array for metrics calculation
        predictions = predictions_sparse.toarray()
        
        # 1. Calculate Comprehensive Metrics 
        s_accuracy = accuracy_score(Y_test, predictions) 
        h_loss = hamming_loss(Y_test, predictions)
        j_score_macro = jaccard_score(Y_test, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(Y_test, predictions, average='macro', zero_division=0)
        f1_micro = f1_score(Y_test, predictions, average='micro', zero_division=0)

        print("\n--- Logistic Regression (Classifier Chain) Performance ---")
        print(f"Test Samples: {Y_test.shape[0]}")
        print(f"Hyperparameters (LR): C={self.C}, penalty={self.penalty}")
        print("-" * 50)
        print(f"1. **Subset Accuracy (Exact Match)**: {s_accuracy:.4f}")
        print(f"2. **Hamming Loss**: {h_loss:.4f}")
        print(f"3. **Macro F1-Score**: {f1_macro:.4f}")
        print(f"4. **Micro F1-Score**: {f1_micro:.4f}")
        print("-" * 50)

        # 2. Save Results to JSON 
        results = []
        
        for i in range(predictions.shape[0]):
            
            # --- CONVERT INDICES TO HUMAN-READABLE TOPIC NAMES ---
            # Get the indices of the predicted positive labels (where prediction == 1)
            predicted_topic_indices = np.where(predictions[i] == 1)[0]
            # Map indices to names
            predicted_topic_names = [num_topic_map[idx] for idx in predicted_topic_indices if idx in num_topic_map]
            
            # Get the indices of the true positive labels
            true_topic_indices = np.where(Y_test[i] == 1)[0]
            true_topic_names = [num_topic_map[idx] for idx in true_topic_indices if idx in num_topic_map]
            # ----------------------------------------------------
            
            results.append({
                "index": i,
                "prediction": predicted_topic_names, 
                "true_labels": true_topic_names, 
                "test_document_text_start": X_test_raw[i][:100] + "..." 
            })

        with open(outPath, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✅ Results saved successfully to {outPath}")
        
        if results:
            first_result = results[0]
            print("\nExample Prediction (First Sample with Topic Names):")
            print(f"True Labels: {first_result['true_labels']}")
            print(f"Predicted Labels: {first_result['prediction']}")


# --- EXECUTION ---

if __name__ == "__main__":
    
    # 1. Initialize the predictor with LR hyperparameters
    # C=1.0 is the default regularization inverse strength. 
    # Use 'liblinear' solver for L2 penalty on sparse data (or 'lbfgs' if you prefer, but requires dense).
    model = LogisticRegression_Predictor(C=1.0, penalty='l2', solver='liblinear') 
    
    # 2. Load Training Data
    model.createTrainDataset(TRAIN_PATH)
    
    # 3. Vectorize Features
    model.fit_vectorizer()
    
    # 4. Train the Logistic Regression Classifier Chain
    model.train_model()
    
    # 5. Predict and Evaluate
    # NOTE: Ensure the output directory '../output/' exists before running.
    model.predict_and_evaluate(TEST_PATH, "../output/lr_dev.json")