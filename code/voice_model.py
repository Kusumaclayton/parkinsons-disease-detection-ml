import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier


class VoiceModel:
    def __init__(self, path):
        self.path = path
        self.model = XGBClassifier()
        self.scaler = StandardScaler()

    def load_data(self):
        data = pd.read_csv(self.path)
        X = data.drop(["status", "name"], axis=1)
        y = data["status"]
        return X, y

    def preprocess(self, X_train, X_test):
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    def train(self):
        X, y = self.load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        X_train, X_test = self.preprocess(X_train, X_test)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        print("\n--- Voice Model Results ---")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Report:\n", classification_report(y_test, y_pred))
