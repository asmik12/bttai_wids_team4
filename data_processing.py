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
    Imports and loads functional connectome data along with associated metadata.

    This function reads categorical metadata, connectome matrices, and quantitative metadata 
    from Excel and CSV files. If `test` is set to True, it loads test data; otherwise, it 
    loads training data including solutions.

    Args:
        test (bool, optional): If True, loads test data. Defaults to False.

    Returns:
        dict: A dictionary containing imported data with keys:
            - 'test': List of test datasets (categorical metadata, connectome matrices, quantitative metadata).
            - 'train': List of training datasets (categorical metadata, connectome matrices, quantitative metadata, solutions).
    """
    test_cat_metadata, test_connectome_matrices, test_quant_metadata = None, None, None
    data = {'test':[], 'train':[]}

    if test:
        test_cat_metadata = pd.read_excel('data/TEST/TEST_CATEGORICAL.xlsx') # Categorical metadata
        test_connectome_matrices = pd.read_csv('data/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv') #Connectome matrices
        test_quant_metadata = pd.read_excel('data/TEST/TEST_QUANTITATIVE_METADATA.xlsx') # Quantitative metadata
        data['test'].append(test_cat_metadata, test_connectome_matrices, test_quant_metadata)
    else:
        # Importing and preprocessing the dataset
        cat_metadata = pd.read_excel('data/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx') # Categorical metadata
        connectome_matrices = pd.read_csv('data/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv') #Connectome matrices
        quant_metadata = pd.read_excel('data/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx') # Quantitative metadata
        sol = pd.read_excel('data/TRAIN/TRAINING_SOLUTIONS.xlsx') # Solutions to the training data examples
        
        data['train'].append(cat_metadata, connectome_matrices, quant_metadata, sol)

    return data
    

# F1 score for female/sex - female will be given more weights - try to assign more weights to the female category 
# sample_weights parameter - assign a sample weight of ADHD and Female/Sex - play around the female cateogyr weights - x2, x2.5, x3 etc
# Check recall value of classifiaction report on gender
# Increase in recall value & accuracy - check for F1 score. 
# Gender classification - XGBoost!! LightBoost
# Neural nets will outperform
