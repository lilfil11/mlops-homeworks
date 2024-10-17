import mlflow
import os

from airflow.models import DAG, Variable
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Dict, Literal

# YOUR IMPORTS HERE
import io
import time
import json
import logging
import pandas as pd
from datetime import timedelta

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago

from mlflow.models import infer_signature


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

DIR_NAME = 'KirillFilatov'
BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    'owner': 'Kirill Filatov',
    'retry': 3,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    dag_id='Filatov-Kirill',
    schedule = '0 1 * * *',
    start_date = days_ago(2),
    catchup = False,
    tags = ['mlops'],
    default_args = DEFAULT_ARGS
)

model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))


def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)


def init() -> Dict[str, Any]:
    # YOUR CODE HERE
    
    metrics = {}
    metrics['time_pipeline_start'] = time.time()

    exp_name = 'kirill_filatov'
    try:
        exp_id = mlflow.create_experiment(name=exp_name)
    except:
        mlflow.set_experiment(exp_name)
        exp_id = dict(mlflow.get_experiment_by_name(exp_name))['experiment_id']
    
    with mlflow.start_run(run_name='Lilfil11', experiment_id=exp_id) as parent_run:
        metrics['exp_id'] = exp_id
        metrics['run_id'] = parent_run.info.run_id
        
    _LOG.info(f'Initialized!')

    return metrics 
    
def get_data(**kwargs) -> Dict[str, Any]:
    # YOUR CODE HERE
    
    # Принимаем параметры
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='init')
    
    # Скачиваем датасет из SKLearn
    time_start = time.time()
    housing = fetch_california_housing(as_frame=True)
    time_finish = time.time()

    metrics['time_dataset_start'] = time_start
    metrics['time_dataset_finish'] = time_finish
    metrics['dataset_size'] = len(housing.data)

    # Кидаем датасет на S3
    s3_hook = S3Hook('s3_connection') 
    
    buffer = io.BytesIO()
    housing.data.to_pickle(buffer)
    buffer.seek(0)
    s3_hook.load_file_obj(file_obj=buffer, key=f'{DIR_NAME}/datasets/housing_data.pkl', bucket_name=BUCKET, replace=True)

    buffer = io.BytesIO()
    housing.target.to_pickle(buffer)
    buffer.seek(0)
    s3_hook.load_file_obj(file_obj=buffer, key=f'{DIR_NAME}/datasets/housing_target.pkl', bucket_name=BUCKET, replace=True)

    _LOG.info(f'Data is read!')
    
    return metrics

def prepare_data(**kwargs) -> Dict[str, Any]:
    # YOUR CODE HERE

    # Принимаем параметры
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='get_data')

    # Считываем датасет с S3
    s3_hook = S3Hook('s3_connection')
    
    housing_data = s3_hook.download_file(key=f'{DIR_NAME}/datasets/housing_data.pkl', bucket_name=BUCKET)
    X = pd.read_pickle(housing_data)
    cols = list(X.columns)

    time_start = time.time()
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=cols)
    time_finish = time.time()

    metrics['time_prepare_start'] = time_start
    metrics['time_prepare_finish'] = time_finish
    metrics['features'] = list(X.columns)

    # Кидаем нормализованный датасет на S3       
    buffer = io.BytesIO()
    X.to_pickle(buffer)
    buffer.seek(0)
    s3_hook.load_file_obj(file_obj=buffer, key=f'{DIR_NAME}/datasets/housing_data_normalized.pkl', bucket_name=BUCKET, replace=True)

    _LOG.info(f'Data is prepared!')
    
    return metrics

def train_model(model_name: Literal["random_forest", "linear_regression", "desicion_tree"], **kwargs) -> Dict[str, Any]:
    # YOUR CODE HERE

    # Принимаем параметры
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='prepare_data')

    # Считываем датасет с S3
    s3_hook = S3Hook('s3_connection')
    
    housing_data = s3_hook.download_file(key=f'{DIR_NAME}/datasets/housing_data_normalized.pkl', bucket_name=BUCKET)
    housing_target = s3_hook.download_file(key=f'{DIR_NAME}/datasets/housing_target.pkl', bucket_name=BUCKET)
    X = pd.read_pickle(housing_data)
    y = pd.read_pickle(housing_target) 

    df = X.copy()
    df['target'] = y

    time_start = time.time()
    model = {}
    if model_name == 'random_forest':
        model = RandomForestRegressor()
    elif model_name == 'linear_regression':
        model = LinearRegression()
    else:
        model = DecisionTreeRegressor()

    with mlflow.start_run(run_name=model_name, experiment_id=metrics['exp_id'], parent_run_id=metrics['run_id']) as child_run:
        model.fit(X, y)
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)   

        signature = infer_signature(X, preds)
        model_info = mlflow.sklearn.log_model(model, model_name, signature=signature,
                                              registered_model_name=f'sklearn-{model_name}-reg-model')
    
        mlflow.evaluate(model=model_info.model_uri, data=df, targets='target', model_type='regressor', evaluators=['default'])    
        
    time_finish = time.time() 

    metrics['time_train_start'] = time_start
    metrics['time_train_finish'] = time_finish
    metrics['model_name'] = model_name
    metrics['mse'] = mse

    _LOG.info(f'Model is trained!')
    
    return metrics

def save_results(**kwargs) -> None:
    # YOUR CODE HERE

    s3_hook = S3Hook('s3_connection')
    
    ti = kwargs['ti']
    for model_name in models.keys():
        metrics = ti.xcom_pull(task_ids=f'train_{model_name}')
    
        buffer = io.BytesIO()
        buffer.write(json.dumps(metrics).encode())
        buffer.seek(0)
        s3_hook.load_file_obj(file_obj=buffer, key=f'{DIR_NAME}/{model_name}/metrics.json', bucket_name=BUCKET, replace=True)

    _LOG.info(f'Results saved!')
    
task_init = PythonOperator(task_id='init', python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id='get_data', python_callable=get_data, dag=dag, provide_context=True)

task_prepare_data = PythonOperator(task_id='prepare_data', python_callable=prepare_data, dag=dag, provide_context=True)

training_model_tasks = [PythonOperator(task_id=f'train_{model_name}', python_callable=train_model, dag=dag, op_kwargs={'model_name': model_name}, 
                                       provide_context=True) for model_name in models.keys()]

task_save_results = PythonOperator(task_id='save_results', python_callable=save_results, dag=dag, provide_context=True)

configure_mlflow()

task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results
