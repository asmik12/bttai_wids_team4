import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix


def import_processed_data(test=False):
    """
    Imports and merges processed data with functional connectome matrices.

    Args:
        test (bool, optional): If True, loads test data instead of training data. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - data (DataFrame): Merged dataset excluding 'participant_id'.
            - participant_ids (Series): 'participant_id' column for submission.
    """
    file = "./processed_data/train_processed_data.csv"
    connectome_matrices = "./data/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"
    
    if test:
        print("Testing data file names updated")
        file = "./processed_data/test_processed_data.csv"
        connectome_matrices = "./data/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"
    
    processed_data = pd.read_csv(file)
    matrices = pd.read_csv(connectome_matrices)

    # Merge the two
    data = processed_data.merge(matrices, on='participant_id', how='inner')

    # Drop participant_id
    participant_ids = data['participant_id']  # Save for submission
    data = data.drop(columns=['participant_id'])

    print("Data imported successfully.")
    return data, participant_ids

def train_test_split_data(training_data, seed=42, split=0.2):
    """
    Splits the data into training and test sets for ADHD and Sex_F targets.

    Args:
        training_data (DataFrame): Input data containing the features and targets.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        split (float, optional): Fraction of data to be used as test set. Defaults to 0.2.

    Returns:
        tuple: A tuple containing:
            - adhd (list): Split data for ADHD prediction (X_train, X_test, y_train_adhd, y_test_adhd).
            - sex (list): Split data for Sex_F prediction (X_train_sex, X_test_sex, y_train_sex, y_test_sex).
    """

    X = training_data.drop(columns=['ADHD_Outcome', 'Sex_F'])
    y_adhd = training_data['ADHD_Outcome']  # Target 1
    y_sex = training_data['Sex_F']          # Target 2

    X_train, X_test, y_train_adhd, y_test_adhd = train_test_split(X, y_adhd, test_size=split, random_state=seed)
    X_train_sex, X_test_sex, y_train_sex, y_test_sex = train_test_split(X, y_sex, test_size=split, random_state=seed)

    adhd = (X_train, X_test, y_train_adhd, y_test_adhd)
    sex = (X_train_sex, X_test_sex, y_train_sex, y_test_sex)

    return adhd, sex

def decision_tree_training_and_validation(data):

    #   Unpacking the data
    adhd, sex = train_test_split_data(data)
    X_train, X_test, y_train_adhd, y_test_adhd = adhd
    X_train_sex, X_test_sex, y_train_sex, y_test_sex = sex

    # Train ADHD Outcome
    adhd_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    adhd_model.fit(X_train, y_train_adhd)
    y_pred_adhd = adhd_model.predict(X_test)

    # Train Sex Classification
    sex_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    sex_model.fit(X_train_sex, y_train_sex)
    y_pred_sex = sex_model.predict(X_test_sex)

    # ADHD Model
    print("ADHD Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test_adhd, y_pred_adhd))
    print("Classification Report:\n", classification_report(y_test_adhd, y_pred_adhd))

    # Sex Prediction Model
    print("\nSex Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test_sex, y_pred_sex))
    print("Classification Report:\n", classification_report(y_test_sex, y_pred_sex))

#================= MAIN FUNCTION (TESTING ONLY)

def main(test=False):
    # Step 1: Import processed data
    data, participant_ids = import_processed_data(test)
    
    # Step 2: Train decision tree models and evaluate their performance
    decision_tree_training_and_validation(data)

if __name__ == "__main__":
    # Run the main function, with the option to use test data
    main(test=False)



