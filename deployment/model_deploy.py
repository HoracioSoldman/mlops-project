import pandas as pd
from datetime import datetime
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import LinearSVR

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

@task
def read_df(file_path, dataset_size_in_million):
    df_chuncks = pd.read_csv(file_path, chunksize=1000000)
    df = pd.DataFrame({"columns":[]})
    index = 0
    for ck in df_chuncks:
        if index < dataset_size_in_million:
            df = pd.concat([df, ck], axis=0, ignore_index=True)
        index+=1
    return df

@task
def extract_dep_hour(dep_time):
    dep_time = str(int(dep_time))
    
    if len(dep_time) == 3:
        dep_time = f'0{dep_time}'
    return dep_time

@task
def get_train_df(df_source, target):
    df= df_source[['FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST', 'DEP_TIME', 'DISTANCE', 'CRS_DEP_TIME', 'DEP_DELAY']]
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df['FL_DAY'] = df['FL_DATE'].dt.day_name()
    df['DEP_HOUR_MIN'] = df['CRS_DEP_TIME'].apply(lambda x: extract_dep_hour(x))
    df['DEP_HOUR'] = df['DEP_HOUR_MIN'].apply(lambda x: x[:2])
    df['DEP_MIN'] = df['DEP_HOUR_MIN'].apply(lambda x: x[2:])
    df['DEP_DELAY'] = df['DEP_DELAY'].apply(lambda x: abs(x))
    df= df[['FL_DAY','OP_CARRIER', 'ORIGIN', 'DEST', 'DISTANCE', 'DEP_HOUR', 'DEP_MIN', 'DEP_DELAY']]
    
    categorical = ['FL_DAY','OP_CARRIER', 'ORIGIN', 'DEST', 'DEP_HOUR', 'DEP_MIN']

    df[categorical] = df[categorical].astype(str)
    
    train_df = df.drop(columns = target).copy()
    
    return train_df


@task
def get_trainable_data(data_path, target):
    df = read_df(data_path, 2)
    train_df = get_train_df(df, target)

    dv = DictVectorizer()

    train_dict = train_df.to_dict(orient='records')
    X = dv.fit_transform(train_dict)
    y = df[target]

    # missing values -> 0
    y.fillna(0, inplace=True)

    # we are going to split our dataset into 80:10:10 as training:test:validation respectively
    train_size=0.8
    # split the data in training and other dataset
    X_train, X_oth, y_train, y_oth = train_test_split(X, y, train_size=train_size)

    # for the other data which is the remaining one, we split it into test and validation
    test_size = 0.5
    X_valid, X_test, y_valid, y_test = train_test_split(X_oth, y_oth, test_size=0.5)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, dv


# train xgboost model
@task
def best_model_search(train, valid, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=800,
                evals=[(valid, 'validation')],
                early_stopping_rounds=30
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials()
    )
    return

@task
def train_model_search(X_train, y_train, X_valid, y_valid, data_path):
    mlflow.sklearn.autolog()
    model_files = ['lin_reg.bin', 'lasso.bin', 'lvr.bin', 'xgb.bin']

    for index, model_class in enumerate([LinearRegression, Lasso, LinearSVR, xgb]):

        if model_files[index] == 'xgb.bin':
            train = xgb.DMatrix(X_train, label=y_train.values)
            valid = xgb.DMatrix(X_valid, label=y_valid.values)
            best_model_search(train, valid, y_valid.values)
            

        else:
            with mlflow.start_run():

                mlflow.set_tag("model", model_files[index])
                mlflow.log_param("data-path", data_path)

                mlmodel = model_class()
                mlmodel.fit(X_train, y_train.values.ravel())


                with open(f'models/{model_files[index]}', 'wb') as f_out:
                    pickle.dump((dv, mlmodel), f_out)

                y_pred = mlmodel.predict(X_valid)
                rmse = mean_squared_error(y_valid, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_artifact(local_path=f"models/{model_files[index]}", artifact_path="models_pickle")


@task
def best_xgboost_model(train, valid, y_valid, dv):
    with mlflow.start_run():

        best_params = {
            'learning_rate': 0.05761444959082953,
            'max_depth': 5,
            'min_child_weight': 2.263346156223211,
            'objective': 'reg:linear',
            'reg_alpha': 0.040130067007024615,
            'reg_lambda': 0.05964721843226435,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=800,
            evals=[(valid, 'validation')],
            early_stopping_rounds=30
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        with open("models/xgboost.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/xgboost.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")



@flow(task_runner=SequentialTaskRunner())
def main(data_path = '../data/2018.csv'):
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("flight-delay-experiment")

    target = ['DEP_DELAY']
    
    
    X_train, y_train, X_val, y_val, X_test, y_test, dv = get_trainable_data(data_path, target).result()
    
    train_model_search(X_train, y_train, X_val, y_val,data_path)
    train = xgb.DMatrix(X_train, label=y_train.values)
    valid = xgb.DMatrix(X_val, label=y_val.values)
    best_xgboost_model(train, valid, y_val, dv)

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

Deployment(
    flow=main,
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(days=3)),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml", "flight", "delay"]
)
