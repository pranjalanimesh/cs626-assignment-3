import streamlit as st
import numpy as np
from datasets import load_dataset
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
import pandas as pd

# Load the dataset
dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)

# NER Tag Mapping
ner_tag_map = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


# Function to map numerical NER tags to BIO labels
def map_ner_tags(dataset_split):
    for example in dataset_split:
        example["ner_tags"] = [ner_tag_map[tag] for tag in example["ner_tags"]]
    return dataset_split


# Apply tag mapping
dataset["train"] = map_ner_tags(dataset["train"])
dataset["validation"] = map_ner_tags(dataset["validation"])
dataset["test"] = map_ner_tags(dataset["test"])


# Feature extraction
def extract_features(tokens, pos_tags, chunk_tags, idx):
    token = tokens[idx]
    pos = pos_tags[idx]
    chunk = chunk_tags[idx]
    features = {
        "token": token,
        "pos": pos,
        "chunk": chunk,
        "prev_token": tokens[idx - 1] if idx > 0 else "BOS",
        "next_token": tokens[idx + 1] if idx < len(tokens) - 1 else "EOS",
    }
    return features


def prepare_data(dataset_split):
    X, y = [], []
    for example in dataset_split:
        tokens = example["tokens"]
        pos_tags = example["pos_tags"]
        chunk_tags = example["chunk_tags"]
        ner_tags = example["ner_tags"]
        for idx in range(len(tokens)):
            X.append(extract_features(tokens, pos_tags, chunk_tags, idx))
            y.append(ner_tags[idx])
    return X, y


# Prepare training data
X_train, y_train = prepare_data(dataset["train"])
X_test, y_test = prepare_data(dataset["test"])

# Encode labels and vectorize features
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
vectorizer = DictVectorizer(sparse=True)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the classifier
classifier = svm.LinearSVC()
classifier.fit(X_train_vectorized, y_train_encoded)


# Prediction and evaluation function
def predict_and_format(sentence):
    tokens = sentence.split()
    features = [
        extract_features(tokens, ["NOUN"] * len(tokens), ["O"] * len(tokens), i)
        for i in range(len(tokens))
    ]
    vectorized_features = vectorizer.transform(features)
    predictions = classifier.predict(vectorized_features)
    labels = label_encoder.inverse_transform(predictions)
    return " ".join([f"{token}_{label}" for token, label in zip(tokens, labels)])


# Streamlit layout
st.title("Named Entity Recognition Demo")

# Input area
st.header("Enter a sentence to perform Named Entity Recognition:")
user_input = st.text_input("Sentence", "Enter your sentence here")

if user_input:
    result = predict_and_format(user_input)
    st.subheader("Predicted Output:")
    st.write(result)

# Model evaluation
st.header("Model Performance on Test Set")
y_test_pred = classifier.predict(X_test_vectorized)
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
y_test_true_labels = label_encoder.inverse_transform(y_test_encoded)
report = classification_report(y_test_true_labels, y_test_pred_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.write("### Classification Report:")
st.write(report_df)
