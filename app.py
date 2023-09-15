from flask import Flask, render_template, request, jsonify
import json
from function import *
from beamcalc import *

app = Flask(__name__)

# prevent caching
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# auto reload templates
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        req = json.loads(json.dumps(request.get_json()))  
        print(req)

        # process JSON request
        req_data = request.get_json()
        print(req_data)
        result = caller(req_data)  
        return jsonify(result), 201

    # render HTML for GET request
    return render_template('index.html') 

if __name__ == '__main__':
    app.run(debug = True)
    # app.run()