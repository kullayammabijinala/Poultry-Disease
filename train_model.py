from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import pickle

data = []
labels = []

# dataset/
# ├── fowlpox/
# │   ├── img1.jpg
# ├── coccidiosis/
# │   ├── img2.jpg

base_path = 'poultry_dataset'

for class_name in os.listdir(base_path):
    class_path = os.path.join(base_path, class_name)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                img = img.astype("float32") / 255.0
                img_flat = img.flatten()
                data.append(img_flat)
                labels.append(class_name.lower())

X = np.array(data)
y = np.array(labels)

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    print("model is succufully ")
