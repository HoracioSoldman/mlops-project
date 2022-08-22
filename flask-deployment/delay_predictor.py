import pickle

from flask import Flask, request, jsonify

with open('flask-deployment/lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])


app = Flask('delay-prediction')


@app.route('/predict-delay', methods=['POST'])
def predict_endpoint():
    info = request.get_json()

    pred = predict(info)

    result = {
        'delay': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=7200)