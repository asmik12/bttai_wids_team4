# bttai_wids_team4

To run this project on your own machine with the original, unprocessed dataset, download the publicly available dataset from Kaggle at this link, and add the contents of the `widsdatathon2025` folder to a folder named `data` in the home directory of this repository.

`https://www.kaggle.com/competitions/widsdatathon2025`

Run the `data_preprocessing.ipynb` notebook first to setup and create the merged dataset. Subsequent files in this project will refer to this dataset that will be stored as `merged_data.csv` on your local machine.

### Preprocessing Steps
`data_preprocessing.ipynb` contains detailed documentation about the preprocessing steps taken on this data. In short, it performs the following preprocessing operations (not in order):
1. Imports all the data into dataframes
2. Encodes the categorical features from `TRAIN_CATEGORICAL_METADATA.XLSX` appropriately
3. Label encodes the `ADHD_Outcome` feature from `TRAINING_SOLUTIONS.XLSX` (previously str)
4. Checks for nan values and imputes with most frequent feature value/discards the example as appropriate.
5. ADD
6. ADD
7. ADD
8. Merges the data into one single dataframe and saves it to a csv file called `merged_data.csv`
