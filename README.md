## mlops-project
This project aims to predict the delay of flight departures on a dataset from [Airline delay and cancellation data 2009-2018](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018)


### 1. Installing the project
- Start the environment with pipenv
```bash
     pipenv shell
```
- Install the required modules
```bash
     pipenv install
```

Alternatively, if the modules installation with pipenv is too slow, the [requirements.txt](requirements.txt) file can be used in a new virtual environment. 
```bash
     python3 -m venv venv
```
Then activate the environment with:
```bash
     source venv/bin/activate
```
Install the modules
```bash
     pip install -r requirements.txt
```

### 2. Running the project    
### MLFlow
- Launch __mlflow server__ by executing the following command:
```bash
     mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root=artifacts
```
- Execute all the cells in [eperimentation.ipynb](notebook/experimentation.ipynb)

- Head over to the MLFlow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000)


### Prefect
- Start the Orion Server
```bash
     prefect orion start
```
- Run the `model_deploy.py` to check if the script is working fine.
```bash
     python orchestration/model_deploy.py
```

Then head over to the Prefect UI at [http://127.0.0.1:4200](http://127.0.0.1:4200/) 

- Create a deployment:
```bash
     prefect deployment create orchestration/model_deploy.py 
```

### Deployment
- With Docker and flask

```bash
docker build -t delay-prediction-service:v1 .
```

```bash
docker run -it --rm -p 7200:7200  delay-prediction-service:v1
```

### Test the model

```bash
python flask-deployment/tester.py
```

