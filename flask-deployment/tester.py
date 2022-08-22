import requests

flight_info = {
    "FL_DAY": 'Tuesday',
    "OP_CARRIER": 'UA',
    "ORIGIN": 'ORD',
    "DISTANCE": 762.0,
    "DEP_HOUR": '10',
    "DEP_MIN": '25'
}

url = 'http://localhost:7200/predict-delay'
response = requests.post(url, json=flight_info)
print(response.json())
