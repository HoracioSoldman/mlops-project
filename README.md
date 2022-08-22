## mlops-project
This project aims to predict the delay of flight departures on a dataset from [Airline delay and cancellation data 2009-2018](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018)


1. Running the project
    - Start the environment
    ```bash
         pipenv shell
    ```
    - Install the required modules
    ```bash
         pipenv install
    ```
    - Launch __mlflow server__ by executing the following command:
    ```bash
         mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root=artifacts
    ```
    - Execute all the cells in [eperimentation.ipynb](notebook/experimentation.ipynb)

    - Head over to the [MLFlow UI: http://127.0.0.1:5000](http://127.0.0.1:5000)

