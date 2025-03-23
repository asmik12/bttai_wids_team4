import pandas as pd
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score
from model import import_processed_data

def load_model(model_path):
    """Load the saved XGBoost model."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

def save_predictions_to_csv(predictions, participant_ids, filename="predictions.csv"):
    """Save the predictions along with participant IDs to a CSV file."""
    output_df = pd.DataFrame({
        'participant_id': participant_ids,
        'prediction': predictions
    })
    output_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def predict_with_model(model, test_data):
    """Make predictions using the loaded model on the test data."""
    predictions = model.predict(test_data)
    return predictions

# Main function to load models, make predictions, and save the results
def main():
    # Load the testing data
    test_data, participant_ids = import_processed_data(test=True)
    
    # Load the trained models (ADHD and Sex models)
    adhd_model_path = 'xgb_best_adhd_model.json' 
    sex_model_path = 'xgb_best_model.json'
    
    # Load the models
    adhd_model = load_model(adhd_model_path)
    sex_model = load_model(sex_model_path)

    # Predict using both models (you can choose to use one or both)
    adhd_predictions = predict_with_model(adhd_model, test_data)
    sex_predictions = predict_with_model(sex_model, test_data)

    # Save the predictions to CSV files
    save_predictions_to_csv(adhd_predictions, participant_ids, filename="adhd_predictions.csv")
    save_predictions_to_csv(sex_predictions, participant_ids, filename="sex_predictions.csv")

if __name__ == "__main__":
    main()
