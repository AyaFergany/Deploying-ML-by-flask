import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def load_data(file_path):
    try:
        df = pd.read_csv("iris.csv")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def preprocess_data(df):
    features = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    target = df["species"]
    return features, target

def train_model(features, target):
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=50)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    return classifier

def save_model(model, file_path):
    try:
        pickle.dump(model, open(file_path, "wb"))
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    file_path = "iris.csv"
    df = load_data(file_path)
    if df is None:
        return
    features, target = preprocess_data(df)
    model = train_model(features, target)
    save_model(model, "model.pkl")

if __name__ == "__main__":
    main()