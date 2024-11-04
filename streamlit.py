import streamlit as st
import numpy as np
from datasets import load_dataset
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import nltk
from nltk import pos_tag, word_tokenize
import re
import string
from typing import List, Tuple, Dict

# Assuming the functions and model training code from the original script are available here
results = {}

def get_pos_tags(sentence: str) -> List[str]:
    """Get POS tags for a sentence using NLTK."""
    if isinstance(sentence, list):
        tokens = sentence
    else:
        tokens = word_tokenize(sentence)
    pos_tagged = pos_tag(tokens)
    return [tag for _, tag in pos_tagged]


def extract_features_for_token(
    token: str,
    tokens: List[str],
    pos_tags: List[str],
    pos_tag_dict: Dict[str, int],
    index: int,
    window_size: int = 2,
) -> np.ndarray:
    """Extract features for a single token."""
    features = []

    # 1. Case-based features (5 features)
    features.extend(
        [
            1.0 if token.isupper() else 0.0,
            1.0 if token.istitle() else 0.0,
            1.0 if token.islower() else 0.0,
            1.0 if any(c.isdigit() for c in token) else 0.0,
            1.0 if re.match(r".*[A-Z].*[A-Z].*", token) else 0.0,  # 2 uppercase letters
        ]
    )

    # 2. Token length feature (1 feature)
    features.append(len(token) / 20.0)

    # 3. Position-based features (2 features)
    features.extend(
        [1.0 if index == 0 else 0.0, 1.0 if index == len(tokens) - 1 else 0.0]
    )

    # 4. Character-based features (4 features)
    features.extend(
        [
            1.0 if "." in token else 0.0,
            1.0 if "," in token else 0.0,
            1.0 if "-" in token else 0.0,
            1.0 if "'" in token else 0.0,
            1.0 if "''" in token else 0.0,
            1.0 if "(" in token else 0.0,
            1.0 if ")" in token else 0.0,
            1.0 if any(c in string.punctuation for c in token) else 0.0,
        ]
    )

    # 5. POS tag feature (one-hot encoded)
    pos_features = np.zeros(len(pos_tag_dict))
    if pos_tags[index] in pos_tag_dict:
        pos_features[pos_tag_dict[pos_tags[index]]] = 1.0
    features.extend(pos_features)

    # 6. Context window features
    for i in range(-window_size, window_size + 1):
        if i == 0:
            continue
        pos = index + i
        if pos < 0 or pos >= len(tokens):
            # Padding features for out-of-bounds positions (5 features per position)
            features.extend([0.0] * 5)
        else:
            context_token = tokens[pos]
            features.extend(
                [
                    1.0 if context_token.isupper() else 0.0,
                    1.0 if context_token.istitle() else 0.0,
                    1.0 if context_token.islower() else 0.0,
                    len(context_token) / 20.0,
                    1.0 if pos_tags[pos] in pos_tag_dict else 0.0,
                ]
            )

    return np.array(features)

def extract_features_for_sentence(
    tokens: List[str], pos_tag_dict: Dict[str, int]
) -> np.ndarray:
    """Extract features for all tokens in a sentence."""
    pos_tags = get_pos_tags(tokens)
    return np.array(
        [
            extract_features_for_token(tokens[i], tokens, pos_tags, pos_tag_dict, i)
            for i in range(len(tokens))
        ]
    )


def predict_sentence(
    sentence: str, svm_model, scaler, pos_tag_dict
) -> List[Tuple[str, int]]:
    """Predict NER tags for a new sentence."""
    tokens = word_tokenize(sentence)

    # Extract features
    features = extract_features_for_sentence(tokens, pos_tag_dict)

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict
    predictions = svm_model.predict(scaled_features)

    return list(zip(tokens, predictions))


def setup_nltk():
    """Download required NLTK resources."""
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")
        nltk.download("punkt")

setup_nltk()

# Streamlit GUI
def main():
    st.title("Named Entity Recognition Demo with SVM and ChatGPT Comparison")
    
    # Sidebar options
    st.sidebar.header("Options")
    show_chatgpt_comparison = st.sidebar.checkbox("Show ChatGPT Comparison", value=True)

    st.write("This demo uses an SVM-based Named Entity Recognition (NER) model trained on the CoNLL-2003 dataset. You can enter a sentence to see which words are identified as named entities.")
    
    # User input
    sentence = st.text_input("Enter a sentence to analyze:", "Washington DC is the capital of United States of America")
    
    if st.button("Analyze Sentence"):
        if sentence:
            # Perform NER with the trained SVM model
            st.write("### SVM Model Prediction")
            predictions = predict_sentence(sentence, results["model"], results["scaler"], results["pos_tag_dict"])
            formatted_output = " ".join([f"{token}_{pred}" for token, pred in predictions])
            st.write(f"**Input:** {sentence}")
            st.write(f"**Output:** {formatted_output}")
            
            # Optionally compare with ChatGPT
            if show_chatgpt_comparison:
                st.write("### ChatGPT-like Response")
                # Mocking a ChatGPT-like response for comparison
                chatgpt_response = mock_chatgpt_ner(sentence)
                st.write(chatgpt_response)

# Mock function to simulate ChatGPT-like NER response
def mock_chatgpt_ner(sentence: str) -> str:
    # In a real application, this function would call the actual ChatGPT API.
    # For demonstration purposes, we'll provide a simple mocked output.
    named_entities = ["Washington DC", "United States of America"]
    tokens = word_tokenize(sentence)
    output = []
    
    for token in tokens:
        if any(token in entity for entity in named_entities):
            output.append(f"{token}_1")  # 1 indicates named entity
        else:
            output.append(f"{token}_0")  # 0 indicates non-named entity
    
    return " ".join(output)

if __name__ == "__main__":
    main()
