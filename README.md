# Student Exam Performance Indicator

A complete end-to-end machine learning project that predicts a student’s maths score using academic and demographic information. The project combines data preprocessing, model training, artifact saving, and a simple Flask web interface for making predictions.

## Project Overview

This application uses a regression model to estimate the target variable `math_score` from features such as:

- Gender
- Race or ethnicity
- Parental level of education
- Lunch type
- Test preparation course
- Reading score
- Writing score

The workflow includes:

1. Data ingestion from the dataset in the notebook folder
2. Data preprocessing and feature engineering
3. Training multiple regression models
4. Selecting the best-performing model
5. Saving trained artifacts
6. Deploying the model through a Flask web app

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib
- Seaborn
- Dill

## Project Structure

```text
mlproject/
├── app.py
├── requirements.txt
├── setup.py
├── artifacts/
├── logs/
├── notebook/
│   ├── data/
│   └── *.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
└── templates/
    ├── home.html
    └── index.html
```

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install the required dependencies

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Running the Application

Start the Flask app with:

```bash
python app.py
```

Then open your browser and visit:

- http://127.0.0.1:5000/
- http://127.0.0.1:5000/predictdata

## Training the Model

The training pipeline is implemented in the data ingestion and model training components. To run the full training workflow:

```bash
python src/components/data_ingestion.py
```

This will:

- load the dataset
- split it into train and test sets
- transform features
- train and evaluate several regression models
- save the best model and preprocessing object in the artifacts folder

## Model Evaluation

The project evaluates models using metrics such as:

- R² score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The best-performing model is selected based on the highest test R² score.

## Usage

1. Open the prediction page
2. Enter the required student details
3. Submit the form
4. View the predicted maths score

## Notes

- The trained model artifacts are stored in the artifacts directory.
- The web interface is simple and designed for demonstration and prediction purposes.
- You can further improve the project by adding model explainability, better UI styling, or deployment to cloud platforms.

## Author

Menghak Phal
