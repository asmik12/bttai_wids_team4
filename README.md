# 🚀 bttai_wids_team4
Welcome to our WIDS Datathon 2025 project! 🎉 This repository contains our team's journey through data preprocessing, exploratory data analysis (EDA), and machine learning model development.

Developed as part of the Break Through Tech AI Program at UCLA, this project showcases our approach to solving a real-world data challenge.

👥 Team Members
* Asmi Kawatkar
* Arda Hoke
* Hearty Parrenas
* Heidi Yu

## 📌 What's Inside?
Here's a quick breakdown of what you'll find in this repo:

📂 Notebooks
* 📊 data_preprocessing.ipynb → Data cleaning & transformation
* 🔍 eda.ipynb → Exploratory data analysis
* 🌲 dec_tree.ipynb → Decision tree model implementation

📂 Python Scripts
* ⚙️ main.py → The main script to run everything!
* 🤖 model.py → Defines machine learning models
* 🛠️ model_utils.py → Helper functions for training & evaluation
* 🏗️ data_processing_utils.py → Utility functions for data handling

📂 Model Outputs
* 📈 predictions.csv → Model-generated predictions
* 🏆 xgb_best_model.json → Our best XGBoost model
* 🎯 xgb_best_adhd_model.json → Specialized ADHD-focused XGBoost model
* 📊 benchmark.json → Model performance metrics
* 🧠 adhd_benchmark.json → Benchmarking for ADHD-specific predictions

## 🚀 Getting Started
Want to run this project yourself? Follow these steps:

1️. Download the Dataset
Get the data from Kaggle

2️. Organize Your Files
Move the downloaded widsdatathon2025 folder into a new directory:
```
mkdir data
mv path_to_downloaded_folder/* data/
```

3. Set Up Your Environment
Install the necessary dependencies:

```
pip install -r requirements.txt
```
4. Run the Full Pipeline
Want to execute everything in one go? Run:

```
python main.py
```

## Notes:
- Run the code using commands like `python data_processing_utils.py` and not `python3`  

For more information, please refer to the `data_preprocessing.ipynb` notebook, where each step is explained in detail.
