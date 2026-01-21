import pickle
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load features (raw images flattened)
with open("data/features.pkl", "rb") as f:
    X = pickle.load(f)

# Load labels
with open("data/labels.pkl", "rb") as f:
    y = pickle.load(f)

print("Original feature shape:", X.shape)

# Convert flattened RGB → grayscale images
X = X.reshape(-1, 50, 50, 3)

hog_features = []

for img in X:
    gray = np.mean(img, axis=2)  # RGB → Gray

    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    hog_features.append(hog_feat)

X_hog = np.array(hog_features)

print("HOG feature shape:", X_hog.shape)

# SVM pipeline (scaling + classifier)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced"
    ))
])

model.fit(X_hog, y)

print("✅ SVM trained successfully")

# ✅ FIXED: Use joblib.dump (consistent with loading)
joblib.dump(model, "data/face_svm_model.pkl")
print("💾 Model saved")
