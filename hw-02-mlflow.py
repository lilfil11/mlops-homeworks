import os
import mlflow
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from mlflow.models import infer_signature


def do_experiment(models):
    # Создаём эксперимент (если не существует)
    exp_name = 'kirill_filatov'
    try:
        exp_id = mlflow.create_experiment(name=exp_name)
    except:
        mlflow.set_experiment(exp_name)
        exp_id = dict(mlflow.get_experiment_by_name(exp_name))['experiment_id']

    # Работаем с датасетом
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    cols = list(X.columns)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=cols)
    
    df = X.copy()
    df['target'] = y

    # Создаём один Parent Run для всех моделей
    with mlflow.start_run(run_name='Lilfil11', experiment_id=exp_id) as parent_run:
        for model_name in models.keys():
            # Создаём Child Run для каждой модели
            with mlflow.start_run(run_name=model_name, experiment_id=exp_id, nested=True) as child_run:
                model = models[model_name]             
                model.fit(X, y)
                preds = model.predict(X)
            
                # Сохраняем метрики и регистрируем модель
                signature = infer_signature(X, preds)
                model_info = mlflow.sklearn.log_model(model, model_name, signature=signature,
                                                      registered_model_name=f'sklearn-{model_name}-reg-model')
        
                mlflow.evaluate(model=model_info.model_uri, data=df, targets='target', model_type='regressor', evaluators=['default'],)
    

models = dict(zip(['LinearRegression', 'DesicionTree', 'RandomForest'], 
                  [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]))    

do_experiment(models)
