from flask import Flask, render_template, request, jsonify
from botChat import get_response

app = Flask(__name__)

@app.get("/")#.route("/", methods=["GET"])#
def index_get():
    print("Flask1 application running")
    return render_template("index.html")

@app.post("/predict")
def predict():
    print("Flask application running")
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True,port=8080)#app.run(debug=True, port=8080)

#run command: FLASK_RUN_PORT=8080 python3 app.py
