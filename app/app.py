from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from inference import predict_image


app = Flask(__name__)
CORS(app) # ローカルへAjaxでPOSTするため

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        result = predict_image(request.form['img'])
        return jsonify({'ans': result})

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run()
