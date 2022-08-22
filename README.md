## mlops-project
This project aims to predict the delay of flight departures in the dataset [Airline delay and cancellation data 2009-2018](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018).


By providing several necessary flight information, the final model is able to predict the delay of the given flight departure. 

The project was developed and can be hosted on a Google VM hosted on GCP. 
Terraform is used to provision the infrastructure.


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

### 3. Using terraform

1. Move to the `terraform directory`

```bash
cd terraform
```


2. Specify the path to the `.json` file containing the Google credentials before calling Terraform.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/.google/credentials/google_credentials.json

```

3. Enable IAM API by visiting [IAM API](https://console.cloud.google.com/apis/api/iam.googleapis.com) then click on: __ENABLE API__
#### Terraform Execution steps
- `terraform init`: 
    * Initializes & configures the backend, installs plugins/providers, & checks out an existing configuration from a version control. 
- `terraform plan`:
    * Matches/previews local changes against a remote state, and proposes an Execution Plan.
- `terraform apply`: 
    * Asks for approval to the proposed plan, and applies changes to cloud.


