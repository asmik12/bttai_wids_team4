
from data_processing_utils import preprocess_data_pipeline, convert_to_csv

def process_data(test=False):
    """
    Processes and saves data to CSV files. Prepares training or test data 
    based on the `test` flag.
    
    Args:
        test (bool, optional): If True, processes test data; otherwise, processes training data. Defaults to False.
    
    Returns:
        None
    """
    cat_metadata, quant_metadata, sol, test_cat_metadata, test_quant_metadata = preprocess_data_pipeline(test)
    convert_to_csv(cat_metadata, quant_metadata, sol, test_cat_metadata, test_quant_metadata, test)
    print("Data Processing Complete.")


def main(process=False):
    """
    Main function to execute the entire data preprocessing pipeline, followed by 
    further steps such as model building or evaluation.

    Args:
        test (bool, optional): If True, processes test data. Defaults to False.
    """
    # Step 1: Preprocess the data (import, clean, normalize, encode)
    if not process:
        test=True
        process_data(test)

    # Step 2: If data is already preprocessed, read it in from the relevant files.
      
    # Step 2: Further steps - Add code here to build and evaluate a model
    # Example: Train a model using the processed data
    if not test:
        pass
        #print("Training data is ready. Proceeding with model training...")
        # Here, you can call functions for model training, e.g., a machine learning model or deep learning model
    
    else:
        pass
        #print("Test data is ready. Proceeding with model evaluation...")
        # Here, you can call functions for model evaluation using the processed test data

if __name__ == "__main__":
    main(process=False)


# F1 score for female/sex - female will be given more weights - try to assign more weights to the female category 
# sample_weights parameter - assign a sample weight of ADHD and Female/Sex - play around the female cateogyr weights - x2, x2.5, x3 etc
# Check recall value of classifiaction report on gender
# Increase in recall value & accuracy - check for F1 score. 
# Gender classification - XGBoost!! LightBoost
# Neural nets will outperform
