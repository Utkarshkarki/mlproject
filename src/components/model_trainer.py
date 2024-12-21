import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging


from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
         self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors classifier":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting classifier":CatBoostRegressor(),
                "AdaBoost Classifier":AdaBoostRegressor(),
                
                }
            params={
                "Decision Tree":{
                    "criterion":["squared_error","friedman_mse","absolute_error","poisson"],
                    #"splitter":["best","random"],
                    #max_features:["auto","sqrt","log2"],
                },
                "Random Forest":{
                    "n_estimators":[8,16,32,64,128,256],
                    #"criterion":["squared_error","friedman_mse","absolute_error","poisson"],
                    #"max_features":["auto","sqrt","log2",none],
                },
                "Gradient Boosting":{
                    #"loss":["squared_error","absolute_error","huber","quantile"],
                    "n_estimators":[8,16,32,64,128,256],
                    'learning_rate': [0.01, 0.1, .001,.05],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    #"criterion":["squared_error","friedman_mse","absolute_error","poisson"],
                    #"max_features":["auto","sqrt","log2",none],
                },
                "Linear Regression":{},
                "K-Neighbors classifier":{
                    'n_neighbors': [5,7,9,11,19],
                    #'weights': ['uniform', 'distance'],
                    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                },
                "XGBClassifier":{
                    'n_estimators': [100, 500, 900, 1100, 1500],
                   # #'max_depth': [2, 3, 5, 10, 15],
                    'booster': ['gbtree', 'gblinear'],
                    #'learning_rate': [0.05, 0.1, 0.15, 0.20],
                    'min_child_weight': [1, 2, 3, 4],
                },
                "CatBoosting classifier":{
                    'depth': [4, 7, 10],
                    #'learning_rate' : [0.03, 0.1, 0.15],
                    #'l2_leaf_reg': [1,4,9],
                    #'iterations': [300],
                },
                "AdaBoost Classifier":{
                    'n_estimators': [50, 100, 250, 500],
                   # 'learning_rate': [0.01, 0.1, 1, 10],
                   # 'loss': ['linear', 'square', 'exponential'],
                }
            }
            
            model_report:dict=evaluate_models( X_train=X_train,y_train=y_train,X_test=X_test,
                                             y_test=y_test,models=models,param=params)
            
            ## To get best model score from dict 
            best_model_score=max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name=list(model_report.keys())[
               list(model_report.values()).index(best_model_score) 
                 ]                     
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found modelonboth training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square





        except Exception as e:
             raise CustomException(e,sys)
  
