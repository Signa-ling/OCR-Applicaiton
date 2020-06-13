from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    message = "Hello"
    return message

@app.route('/welcome')
def welcome():
    message = "Welcome!"
    return message

if __name__ == "__main__":
    app.run(debug=True)
