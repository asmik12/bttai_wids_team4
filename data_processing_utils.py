#Importing the required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def import_data(test=False):
    """
    Imports and loads metadata.

    This function reads categorical metadata and quantitative metadata 
    from Excel and CSV files. If `test` is set to True, it loads test data; otherwise, it 
    loads training data including solutions.

    Args:
        test (bool, optional): If True, loads test data. Defaults to False.

    Returns:
        dict: A dictionary containing imported data with keys:
            - 'test': List of test datasets (categorical metadata, quantitative metadata).
            - 'train': List of training datasets (categorical metadata, quantitative metadata, solutions).
    """
    test_cat_metadata, test_quant_metadata = None, None
    data = {'test':[], 'train':[]}

    if test:
        test_cat_metadata = pd.read_excel('data/TEST/TEST_CATEGORICAL.xlsx') # Categorical metadata
        test_quant_metadata = pd.read_excel('data/TEST/TEST_QUANTITATIVE_METADATA.xlsx') # Quantitative metadata
        data['test'] = [test_cat_metadata, test_quant_metadata]
    else:
        # Importing and preprocessing the dataset
        cat_metadata = pd.read_excel('data/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx') # Categorical metadata
        quant_metadata = pd.read_excel('data/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx') # Quantitative metadata
        sol = pd.read_excel('data/TRAIN/TRAINING_SOLUTIONS.xlsx') # Solutions to the training data examples
        
        data['train'] = [cat_metadata, quant_metadata, sol]

    return test, data

def pack_data(cat_metadata, quant_metadata, sol, test_cat_metadata=None, test_quant_metadata=None):
    """
    Packages training and test datasets into a structured dictionary.

    Args:
        cat_metadata: Categorical metadata for training data.
        quant_metadata: Quantitative metadata for training.
        sol: Training solutions.
        test_cat_metadata (optional): Categorical metadata for test data.
        test_quant_metadata (optional): Quantitative metadata for test data.

    Returns:
        dict: A dictionary with 'train' and 'test' keys containing respective datasets.
    """
    data = {'test':[], 'train':[]}
    data['test'].append(test_cat_metadata, test_quant_metadata)
    data['train'].append(cat_metadata,quant_metadata, sol)
    return data

def unpack_data(data, test):
    """
    Extracts training and test datasets from a structured dictionary.

    Args:
        data (dict): A dictionary containing 'train' and 'test' datasets.

    Returns:
        tuple: Training datasets (cat_metadata, quant_metadata, sol)
               and test datasets (test_cat_metadata, test_quant_metadata).
    """
    cat_metadata, quant_metadata, sol = data['train']
    if test:
        test_cat_metadata, test_quant_metadata = data['test']
    else:
        test_cat_metadata, test_quant_metadata = None, None

    return cat_metadata, quant_metadata, sol, test_cat_metadata, test_quant_metadata


def process_nan_data(cat_metadata, quant_metadata, test_cat_metadata, test_quant_metadata, test=False):
    """
    Handles missing values in categorical and quantitative metadata.

    Args:
        cat_metadata (DataFrame): Categorical metadata for training.
        quant_metadata (DataFrame): Quantitative metadata for training.
        test_cat_metadata (DataFrame): Categorical metadata for testing.
        test_quant_metadata (DataFrame): Quantitative metadata for testing.
        test (bool, optional): Whether to process test data. Defaults to False.

    Returns:
        list: Processed [cat_metadata, quant_metadata, test_cat_metadata, test_quant_metadata].
    """
        
    # Replacing nan in Cateogrical Data with the most common value in the dataset
    col = 'PreInt_Demos_Fam_Child_Ethnicity'
    cat_metadata[col] = cat_metadata[col].fillna(cat_metadata[col].mode()[0])
    print(f"Imputed NAN values for train categorical data. Current NAN values:{cat_metadata.isna().sum().sum()}")

    # Discarding the nan values for quant metadata
    col = 'MRI_Track_Age_at_Scan'
    quant_metadata = quant_metadata.dropna(subset=col)
    print(f"Discarded NAN values for train quant metadata: {col}. Current NAN values:{quant_metadata.isna().sum().sum()}")
    

    # Doing the same for test data
    if test:
        col = 'PreInt_Demos_Fam_Child_Ethnicity'
        test_cat_metadata[col] = test_cat_metadata[col].fillna(test_cat_metadata[col].mode()[0])
        print(f"Imputed NAN values for test categorical data. Current NAN values:{test_cat_metadata.isna().sum().sum()}")

        col = 'MRI_Track_Age_at_Scan'
        test_quant_metadata = test_quant_metadata.dropna(subset=col)
        print(f"Dropped NAN values for test quantitative data. Current NAN values:{test_quant_metadata.isna().sum().sum()}")

    return [cat_metadata, quant_metadata, test_cat_metadata, test_quant_metadata] 

def normalize_cols(quant_metadata, test_quant_metadata, test=False):
    """
    Normalizes numerical columns in quantitative metadata using StandardScaler.

    Args:
        quant_metadata (DataFrame): Quantitative metadata for training.
        test_quant_metadata (DataFrame): Quantitative metadata for testing.
        test (bool, optional): Whether to normalize test data. Defaults to False.

    Returns:
        tuple: Normalized (quant_metadata, test_quant_metadata).
    """

    scaler = StandardScaler()
    exclude = ['participant_id', 'MRI_Track_Age_at_Scan', 'EHQ_EHQ_Total']
    numeric_columns = quant_metadata.select_dtypes(include=['number']).columns.difference(exclude)
    quant_metadata[numeric_columns] = quant_metadata[numeric_columns].astype('float')
    quant_metadata.loc[:,numeric_columns] = scaler.fit_transform(quant_metadata[numeric_columns])



    print(f"Normalized columns for training data.")

    if test:
        numeric_columns = test_quant_metadata.select_dtypes(include=['number']).columns.difference(exclude)
        test_quant_metadata[numeric_columns] = scaler.fit_transform(test_quant_metadata[numeric_columns])
        print(f"Normalized columns for testing data.")
    
    return quant_metadata, test_quant_metadata


def one_hot_encode(cat_metadata, test_cat_metadata, test=False):
    """
    One-hot encodes selected categorical columns in the metadata.

    Args:
        cat_metadata (DataFrame): Categorical metadata for training.
        test_cat_metadata (DataFrame): Categorical metadata for testing.
        test (bool, optional): Whether to one-hot encode test data. Defaults to False.

    Returns:
        tuple: One-hot encoded (cat_metadata, test_cat_metadata).
    """
        
    one_hot_encode_cols = ['Basic_Demos_Study_Site', 'PreInt_Demos_Fam_Child_Ethnicity', 
                       'PreInt_Demos_Fam_Child_Race', 'MRI_Track_Scan_Location']
    # Initialize OneHotEncoder
    encoder = OneHotEncoder()

    # Fit and transform categorical columns
    encoded_cat_data = encoder.fit_transform(cat_metadata[one_hot_encode_cols]).toarray()  # Convert to dense array

    # Get feature names for encoded columns
    encoded_cols = encoder.get_feature_names_out(one_hot_encode_cols)

    # Create a DataFrame for the encoded data
    cat_encoded = pd.DataFrame(encoded_cat_data, columns=encoded_cols, index=cat_metadata.index)

    # Add the encoded columns back to the original DataFrame
    cat_metadata = pd.concat([cat_metadata, cat_encoded], axis=1)
    print("One hot encoded categorical columns.")
    
    if test:
        encoded_test = one_hot_encode(test_cat_metadata)
    
    return cat_metadata, test_cat_metadata
    

def preprocess_data_pipeline(test=False):
    """
    Complete data preprocessing pipeline for both training and test datasets.

    The pipeline includes:
        - Importing the data.
        - Packing the data into a structured format.
        - Handling missing values in categorical and quantitative metadata.
        - Normalizing numerical columns.
        - One-hot encoding categorical columns.

    Args:
        test (bool, optional): Whether to process test data. Defaults to False.

    Returns:
        tuple: Processed data, including:
            - Normalized and encoded training data.
            - Normalized and encoded test data (if `test=True`).
    """
    # Step 1: Import data
    test, data = import_data(test)
    print("Data has been imported.")

    # Step 2: Pack data into a structured format
    cat_metadata, quant_metadata, sol, test_cat_metadata, test_quant_metadata = unpack_data(data, test)

    # Step 3: Handle missing values
    cat_metadata, quant_metadata, test_cat_metadata, test_quant_metadata = process_nan_data(cat_metadata, quant_metadata, test_cat_metadata, test_quant_metadata, test)

    # Step 4: Normalize numerical columns
    quant_metadata, test_quant_metadata = normalize_cols(quant_metadata, test_quant_metadata, test)

    # Step 5: One-hot encode categorical columns
    cat_metadata, test_cat_metadata = one_hot_encode(cat_metadata, test_cat_metadata, test)
    
    # Return processed data
    return cat_metadata, quant_metadata, sol, test_cat_metadata, test_quant_metadata

def convert_to_csv(cat_metadata, quant_metadata, sol, test_cat_metadata, test_quant_metadata, test=False):

    if test:
        metadata_file = 'test_processed_data.csv'
    else:
        test_metadata_file = 'train_processed_data.csv'

    combined_data = cat_metadata.merge(quant_metadata, on='participant_id', how='inner')
    combined_data = combined_data.merge(sol, on='participant_id', how='inner')
    quant_metadata.to_csv(metadata_file, index=False)
    print(f"Training quantitative metadata saved to {metadata_file}")

    if test:
        test_combined_data = test_cat_metadata.merge(test_quant_metadata, on='participant_id', how='inner')
        test_combined_data.to_csv(test_metadata_file, index=False)
        print(f"Test metadata saved to {test_metadata_file}")

    return


def main(test=False):
    """
    Main function to execute the entire data preprocessing pipeline, followed by 
    further steps such as model building or evaluation.

    Args:
        test (bool, optional): If True, processes test data. Defaults to False.
    """
    # Step 1: Preprocess the data (import, clean, normalize, encode)
    cat_metadata, quant_metadata, sol, test_cat_metadata, test_quant_metadata = preprocess_data_pipeline(test)
    
    # Step 2: Further steps - Add code here to build and evaluate a model
    # Example: Train a model using the processed data
    if not test:
        print("Training data is ready. Proceeding with model training...")
        # Here, you can call functions for model training, e.g., a machine learning model or deep learning model
    
    else:
        print("Test data is ready. Proceeding with model evaluation...")
        # Here, you can call functions for model evaluation using the processed test data

if __name__ == "__main__":
    main(test=False)
