from datasets import load_dataset
dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

# Mapping of NER tags as per CoNLL-2003
# These tags are typically:
# 0: O
# 1: B-PER
# 2: I-PER
# 3: B-ORG
# 4: I-ORG
# 5: B-LOC
# 6: I-LOC
# 7: B-MISC
# 8: I-MISC
ner_tag_map = {
    0: 'O',
    1: 'B-PER',
    2: 'I-PER',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-LOC',
    6: 'I-LOC',
    7: 'B-MISC',
    8: 'I-MISC'
}

# Function to map numerical NER tags to BIO labels
def map_ner_tags(dataset_split):
    for example in dataset_split:
        example['ner_tags'] = [ner_tag_map[tag] for tag in example['ner_tags']]
    return dataset_split

# Apply mapping to all splits
dataset['train'] = map_ner_tags(dataset['train'])
dataset['validation'] = map_ner_tags(dataset['validation'])
dataset['test'] = map_ner_tags(dataset['test'])

# Function to extract features from a token
def extract_features(tokens, pos_tags, chunk_tags, idx):
    token = tokens[idx]
    pos = pos_tags[idx]
    chunk = chunk_tags[idx]
    
    features = {
        'token': token,
        'pos': pos,
        'chunk': chunk,
    }
    
    # Previous token features
    if idx > 0:
        features['prev_token'] = tokens[idx - 1]
        features['prev_pos'] = pos_tags[idx - 1]
    else:
        features['prev_token'] = 'BOS'  # Beginning of sentence
        features['prev_pos'] = 'BOS'
        
    # Next token features
    if idx < len(tokens) - 1:
        features['next_token'] = tokens[idx + 1]
        features['next_pos'] = pos_tags[idx + 1]
    else:
        features['next_token'] = 'EOS'  # End of sentence
        features['next_pos'] = 'EOS'
    
    return features

# Prepare the dataset for training
def prepare_data(dataset_split):
    X = []
    y = []
    for example in dataset_split:
        tokens = example['tokens']
        pos_tags = example['pos_tags']
        chunk_tags = example['chunk_tags']
        ner_tags = example['ner_tags']
        for idx in range(len(tokens)):
            X.append(extract_features(tokens, pos_tags, chunk_tags, idx))
            y.append(ner_tags[idx])
    return X, y


# Prepare training data
X_train, y_train = prepare_data(dataset['train'])
X_val, y_val = prepare_data(dataset['validation'])
X_test, y_test = prepare_data(dataset['test'])

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Vectorize the features
vectorizer = DictVectorizer(sparse=True)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize the SVM classifier
# Using a linear kernel for efficiency; you can experiment with other kernels
classifier = svm.LinearSVC()

# Train the classifier
print("Training the SVM classifier...")
classifier.fit(X_train_vectorized, y_train_encoded)
print("Training completed.")

# Predict on the validation set
print("Predicting on the validation set...")
y_val_pred = classifier.predict(X_val_vectorized)

# Decode the labels
y_val_pred_labels = label_encoder.inverse_transform(y_val_pred)
y_val_true_labels = label_encoder.inverse_transform(y_val_encoded)

# Evaluate the model
print("Evaluation on Validation Set:")
print(classification_report(y_val_true_labels, y_val_pred_labels))

# Predict on the test set
print("Predicting on the test set...")
y_test_pred = classifier.predict(X_test_vectorized)
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
y_test_true_labels = label_encoder.inverse_transform(y_test_encoded)


# Evaluate the model
print("Evaluation on Test Set:")
print(classification_report(y_test_true_labels, y_test_pred_labels))


# Corrected format_output function
def format_output(dataset_split, predictions):
    formatted_sentences = []
    current_index = 0  # Pointer to track the position in predictions
    
    for example in dataset_split:
        tokens = example['tokens']
        num_tokens = len(tokens)

        # Extract the predictions for the current sentence
        pred = predictions[current_index:current_index + num_tokens]
        current_index += num_tokens  # Move the pointer forward
        
        tagged_tokens = []
        for token, p in zip(tokens, pred):
            tag = ner_tag_map[p]
            if tag == 'O':
                tagged_tokens.append(token)
            else:
                # Split the tag into BIO and entity type
                try:
                    bio, entity = tag.split('-')
                    tagged_tokens.append(f"{token}_{bio}")
                except ValueError:
                    # Handle cases where the tag might not follow the expected format
                    tagged_tokens.append(token)
        
        formatted_sentence = ' '.join(tagged_tokens)
        formatted_sentences.append(formatted_sentence)
    
    return formatted_sentences

# Format the test set predictions using the corrected function
formatted_test_output = format_output(dataset['test'], y_test_pred_labels)

# Display some examples
for i in range(5):
    print(f"Input: {' '.join(dataset['test'][i]['tokens'])}")
    print(f"Output: {formatted_test_output[i]}")
    print()
            