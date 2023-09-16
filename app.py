from flask import Flask, render_template, request, jsonify
from beamcalc import *
from elements import *

app = Flask(__name__)

# prevent caching
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# auto reload templates
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/', methods=['GET', 'POST'])
def index():

    # process JSON request
    if request.method == 'POST':
        req_data = request.get_json()
        result = generate_graphs(req_data)  
        return jsonify(result), 201

    # render HTML for GET request
    return render_template('index.html') 

if __name__ == '__main__':
    app.run()