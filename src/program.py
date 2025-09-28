import os
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# 1. Load dataset
# -------------------------------
dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "rows.csv")

if not os.path.exists(dataset_path):
    print("❌ File not found:", dataset_path)
    exit()

df = pd.read_csv(dataset_path, low_memory=False)
print("\n✅ Dataset loaded successfully!")
print("Columns available:", df.columns)

# -------------------------------
# 2. Check available categories
# -------------------------------
print("\nAvailable Products:", df['Product'].unique())

if 'Consumer complaint narrative' in df.columns and 'Product' in df.columns:
    df = df[['Consumer complaint narrative', 'Product']].dropna()
else:
    print("❌ Required columns not found.")
    exit()

# -------------------------------
# 3. Map categories
# -------------------------------
category_map = {
    'Debt collection': 1,
    'Consumer Loan': 2,
    'Mortgage': 3,
    'Credit reporting, repair, or other': 0
}

df['label'] = df['Product'].map(category_map)
df = df.dropna(subset=['label'])

# -------------------------------
# 4. Exploratory Data Analysis
# -------------------------------
print("\nDataset Info:")
print(df['label'].value_counts())

sns.countplot(x=df['label'])
plt.title("Category Distribution")
plt.show()

# -------------------------------
# 5. Advanced Text Preprocessing
# -------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['Consumer complaint narrative'].apply(clean_text)

# -------------------------------
# 6. Train/Test Split
# -------------------------------
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 7. Feature Engineering (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------------
# 8. Handle Class Imbalance (SMOTE)
# -------------------------------
sm = SMOTE(random_state=42)
X_train_tfidf_res, y_train_res = sm.fit_resample(X_train_tfidf, y_train)

print("\n✅ After SMOTE:")
print(pd.Series(y_train_res).value_counts())

# -------------------------------
# 9. Model Training & Comparison
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(class_weight="balanced")
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf_res, y_train_res)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# -------------------------------
# 10. Model Performance Summary
# -------------------------------
print("\n📊 Model Comparison:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# -------------------------------
# 11. Save Best Model
# -------------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print(f"\n✅ Best model saved: {best_model_name}")

# -------------------------------
# 12. Prediction Example
# -------------------------------
example = ["I am unable to pay my loan, and debt collectors are calling me daily."]
example_clean = [clean_text(text) for text in example]
example_vec = vectorizer.transform(example_clean)
pred = best_model.predict(example_vec)[0]

reverse_map = {v: k for k, v in category_map.items()}
print("Prediction for example text:", reverse_map[pred])
