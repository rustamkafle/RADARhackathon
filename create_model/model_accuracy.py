import pickle
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load data
with open("data/features.pkl", "rb") as f:
    X = pickle.load(f)

with open("data/labels.pkl", "rb") as f:
    y = pickle.load(f)

# Reshape flattened images
X = X.reshape(-1, 50, 50, 3)

# Extract HOG features
hog_features = []
for img in X:
    gray = np.mean(img, axis=2)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    hog_features.append(hog_feat)

X_hog = np.array(hog_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_hog,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Same model as training
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced"
    ))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

print("🎯 Model Accuracy:", round(accuracy * 100, 2), "%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
