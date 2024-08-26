from flask import Flask,request,jsonify
import pickle

app = Flask(__name__)

def get_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@app.route('/')
def home():
    return "hello"

@app.route('/predict', methods=['POST'])
def predict():
    data=request.get_json()
    model=get_model()
    print('model')
    prediction=model.predict([data['input']])
    return jsonify({'prediction': int(prediction[0])})

if __name__=="__main__":
    app.run(debug=True)