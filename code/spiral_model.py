import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


class SpiralModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = RandomForestClassifier(n_estimators=100)

    def load_images(self):
        data = []
        labels = []

        for label in ["healthy", "parkinson"]:
            path = os.path.join(self.dataset_path, label)

            for file in os.listdir(path):
                img_path = os.path.join(path, file)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))

                data.append(img.flatten())
                labels.append(0 if label == "healthy" else 1)

        return np.array(data), np.array(labels)

    def train(self):
        X, y = self.load_images()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        print("\n--- Spiral Model Results ---")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
