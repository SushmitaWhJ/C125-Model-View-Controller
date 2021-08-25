from flask import Flask, jsonify, request
from classifier import  get_prediction

app = Flask(__name__)

@app.route('/')
def index():
  return "Welcome to home page,This is our API"

#Now will be starting a route 
#we need a post request to send the image to the prediction model 
# and our route name would be ‘Predict digit’

@app.route("/predict-digit", methods=["POST"])

def predict_data():
  # image = cv2.imdecode(np.fromstring(request.files.get("digit").read(), np.uint8), cv2.IMREAD_UNCHANGED)
  #requesting the API to get data as digit .Digit is the key
  image = request.files.get("digit")
  # using get prediction methos from classifier.py
  prediction = get_prediction(image)
  return jsonify({
    "prediction": prediction
  }), 200

if __name__ == "__main__":
  app.run(debug=True)