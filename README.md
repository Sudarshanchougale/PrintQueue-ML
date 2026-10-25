
# PrintQueue
## Photocopy and Printing Job Completion-Time Prediction

### Project Overview
PrintQueue is a Machine Learning mini-project that estimates the
time required for a photocopy or printing job to be ready for collection.

The project uses Multiple Linear Regression and a Streamlit web
interface to accept job details and display an estimated completion time.

### Objectives
- Estimate total job completion time.
- Study how job size and queue workload relate to completion time.
- Apply a supervised machine learning regression algorithm.
- Evaluate predictions using standard regression metrics.
- Develop a simple interactive prediction interface.

### Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

### Machine Learning Method
Algorithm: Multiple Linear Regression

Input features:
- Page count
- Number of jobs ahead
- Printing type
- Binding requirement
- Available printers
- Pages waiting ahead in the queue

Target variable: completion_time_minutes

The dataset is split into 80% training data and 20% testing data.
Categorical features are encoded, numeric features are scaled,
and preprocessing is fitted using training data only.

### Dataset
The current development dataset contains 300 records and 11 columns.
It is synthetically generated for development and testing.
It must not be presented as real photocopy-shop data.

### Evaluation Results
- Mean Absolute Error (MAE): 1.93 minutes
- Root Mean Squared Error (RMSE): 2.89 minutes
- R-squared (R²): 0.820

These results are based on a held-out test set from the synthetic
dataset and do not establish real-world accuracy.

### Project Structure
- data/raw/printqueue_data.csv
- models/printqueue_model.joblib
- reports/actual_vs_predicted.png
- reports/residual_plot.png
- reports/correlation_heatmap.png
- reports/model_metrics.csv
- reports/test_predictions.csv
- train.py
- evaluate.py
- app.py
- requirements.txt
- README.md

### Installation
1. Install Python.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment.
4. Install dependencies:

   pip install -r requirements.txt

### Run the Project
Train the model:

   python train.py

Generate evaluation outputs:

   python evaluate.py

Launch the web application:

   streamlit run app.py

### Limitations
- Current evaluation uses synthetic data.
- Real shop performance may differ from synthetic patterns.
- Predictions depend on the accuracy of the input values.
- Unexpected delays and operational changes may affect completion time.

### Future Scope
- Collect anonymized real-world shop data.
- Validate the model on real observations.
- Compare regression algorithms.
- Add prediction history and improved reporting.
