import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Splitting training and testing arrays...")

            X_train = train_array[:, :-1].astype(np.float32)
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1].astype(np.float32)
            y_test = test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0,),
                "CatBoost": CatBoostRegressor(verbose=False,random_seed=42,),
            }

            params = {
                "Linear Regression": {},
                "Decision Tree": {
                    "criterion": ["squared_error","absolute_error","poisson"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200],
                    "subsample": [0.8, 1.0],
                },

                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                },

                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                },

                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },

                "CatBoost": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200],
                },
            }

            model_report = {}
            trained_models = {}

            for model_name, model in models.items():

                logging.info(f"Training {model_name}")

                param_grid = params.get(model_name, {})

                if len(param_grid) > 0:

                    gs = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=3,
                        scoring="r2",
                        n_jobs=-1,
                    )

                    gs.fit(X_train, y_train)

                    model = gs.best_estimator_

                    logging.info(
                        f"Best Parameters ({model_name}) : {gs.best_params_}"
                    )

                else:

                    model.fit(X_train, y_train)

                trained_models[model_name] = model

                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)

                model_report[model_name] = test_r2

                logging.info(
                    f"{model_name} | Train R² = {train_r2:.4f} | Test R² = {test_r2:.4f}"
                )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            logging.info(f"Best Model : {best_model_name}")
            logging.info(f"Best R² Score : {best_model_score:.4f}")

            if best_model_score < 0.60:
                raise CustomException("No suitable model found with R² >= 0.60", sys,)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predictions = best_model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))

            return {
                "best_model_name": best_model_name,
                "r2_score": r2,
                "mae": mae,
                "rmse": rmse,
            }

        except Exception as e:
            raise CustomException(e, sys)