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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

DIR_NAME = 'KirillFilatov'
BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    'owner': 'Kirill Filatov',
    'retry': 3,
    'retry_delay': timedelta(minutes=1)
}
model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))


def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):

    ####### DAG STEPS #######

    def init(m_name: Literal["random_forest", "linear_regression", "desicion_tree"]) -> Dict[str, Any]:
        # YOUR CODE HERE
        
        metrics = {}
        metrics['time_pipeline_start'] = time.time()
        metrics['model_name'] = m_name

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
        s3_hook.load_file_obj(file_obj=buffer, key=f'{DIR_NAME}/{m_name}/datasets/housing_data.pkl', bucket_name=BUCKET, replace=True)

        buffer = io.BytesIO()
        housing.target.to_pickle(buffer)
        buffer.seek(0)
        s3_hook.load_file_obj(file_obj=buffer, key=f'{DIR_NAME}/{m_name}/datasets/housing_target.pkl', bucket_name=BUCKET, replace=True)

        _LOG.info(f'Data is read!')
        
        return metrics

    def prepare_data(**kwargs) -> Dict[str, Any]:
        # YOUR CODE HERE

        # Принимаем параметры
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='get_data')

        # Считываем датасет с S3
        s3_hook = S3Hook('s3_connection')
        
        housing_data = s3_hook.download_file(key=f'{DIR_NAME}/{m_name}/datasets/housing_data.pkl', bucket_name=BUCKET)
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
        s3_hook.load_file_obj(file_obj=buffer, key=f'{DIR_NAME}/{m_name}/datasets/housing_data_normalized.pkl', bucket_name=BUCKET, replace=True)

        _LOG.info(f'Data is prepared!')
        
        return metrics
        

    def train_model(**kwargs) -> Dict[str, Any]:
        # YOUR CODE HERE

        # Принимаем параметры
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='prepare_data')

        # Считываем датасет с S3
        s3_hook = S3Hook('s3_connection')
        
        housing_data = s3_hook.download_file(key=f'{DIR_NAME}/{m_name}/datasets/housing_data_normalized.pkl', bucket_name=BUCKET)
        housing_target = s3_hook.download_file(key=f'{DIR_NAME}/{m_name}/datasets/housing_target.pkl', bucket_name=BUCKET)
        X = pd.read_pickle(housing_data)
        y = pd.read_pickle(housing_target) 

        time_start = time.time()
        model = {}
        if metrics['model_name'] == 'random_forest':
            model = RandomForestRegressor()
        elif metrics['model_name'] == 'linear_regression':
            model = LinearRegression()
        else:
            model = DecisionTreeRegressor()
            
        model.fit(X, y)
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)
        time_finish = time.time()      

        metrics['time_train_start'] = time_start
        metrics['time_train_finish'] = time_finish
        metrics['mse'] = mse
        metrics['mae'] = mae
        metrics['r2'] = r2

        _LOG.info(f'Model is trained!')
        
        return metrics
    

    def save_results(**kwargs) -> None:
        # YOUR CODE HERE

        # Принимаем параметры
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='train_model')

        # Кидаем метрики на S3
        s3_hook = S3Hook('s3_connection')
        
        buffer = io.BytesIO()
        buffer.write(json.dumps(metrics).encode())
        buffer.seek(0)
        s3_hook.load_file_obj(file_obj=buffer, key=f'{DIR_NAME}/{m_name}/results/metrics.json', bucket_name=BUCKET, replace=True)

        _LOG.info(f'Results saved!')

    ####### INIT DAG #######

    dag = DAG(# YOUR CODE HERE
        dag_id=dag_id,
        schedule = '0 1 * * *',
        start_date = days_ago(2),
        catchup = False,
        tags = ['mlops'],
        default_args = DEFAULT_ARGS
    )

    with dag:
        # YOUR TASKS HERE
        task_init = PythonOperator(task_id='init', python_callable=init, dag=dag, op_kwargs={'m_name': m_name})
        task_get_data = PythonOperator(task_id='get_data', python_callable=get_data, dag=dag, provide_context=True)
        task_prepare_data = PythonOperator(task_id='prepare_data', python_callable=prepare_data, dag=dag, provide_context=True)
        task_train_model = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag, provide_context=True)
        task_save_results = PythonOperator(task_id='save_results', python_callable=save_results, dag=dag, provide_context=True)
        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"kirill_filatov_{model_name}", model_name)
    