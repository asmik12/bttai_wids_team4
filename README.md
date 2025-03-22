# bttai_wids_team4

This project was created as part of the **Break Through Tech AI Program** at UCLA, **Spring 2025 AI Studio Cohort**.

### Contributors:
- Asmi Kawatkar
- Arda Hoke
- Hearty Parrenas
- Heidi Yu

## Setup Instructions

To run this project on your machine with the original, unprocessed dataset, follow these steps:

1. Download the publicly available dataset from Kaggle:
   - [WIDS Datathon 2025 Dataset](https://www.kaggle.com/competitions/widsdatathon2025)
   
2. Add the contents of the `widsdatathon2025` folder to a folder named `data` in the home directory of this repository.

3. Run the `data_preprocessing.ipynb` notebook first to set up and create the merged dataset. This notebook will generate a file called `merged_data.csv` that will be used in subsequent files.

## Preprocessing Steps

The `data_preprocessing.ipynb` notebook contains detailed documentation about the preprocessing steps taken on the data. In brief, the following preprocessing operations are performed:

1. **Imports all the data into DataFrames**.
2. **Encodes categorical features** from `TRAIN_CATEGORICAL_METADATA.XLSX` appropriately.
3. **Label encodes the `ADHD_Outcome` feature** from `TRAINING_SOLUTIONS.XLSX` (previously a string).
4. **Checks for NaN values** and imputes them with the most frequent feature value or discards the example as appropriate.
5. **ADD** – (provide more details about step 5)
6. **ADD** – (provide more details about step 6)
7. **ADD** – (provide more details about step 7)
8. **Merges the data into one single DataFrame** and saves it to a CSV file called `merged_data.csv`.

---

## Notes to update future readme:
- Run the code using commands like `python data_processing_utils.py` and not `python3`  

For more information, please refer to the `data_preprocessing.ipynb` notebook, where each step is explained in detail.
