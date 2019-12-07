from flask import Flask, render_template, request, jsonify
import json
from function import *
from BeamCalc import *

app= Flask(__name__)


app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/',methods =['GET','POST'])
def index():
    req = json.loads(json.dumps(request.get_json()))

    if request.method == 'POST':
        result = caller(req)
        return jsonify(result)

    return render_template('index.html')
   
if __name__ == '__main__':
    app.run()


 