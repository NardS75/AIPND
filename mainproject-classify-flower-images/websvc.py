
from flask import Flask, request, redirect

import model_helper_keras as mhk
import model_helper_pytorch as mhp
import helper
import os

import tensorflow as tf
import json
import uuid

### Configuration start here
category_names = 'cat_to_name.json'
checkpoint_file = 'checkpoints/cp_densenet201_e3_lr0.05.pth'
upload_folder = '_tmp/'
top_k = 5

###

def setup():
    #Load categories
    cat_to_name = helper.load_category_names(category_names)    
    #Define backend
    backend = helper.get_backend(checkpoint_file)

    #Load model checkpoint
    if backend == 'keras':
        model = mhk.create_model_from_checkpoint(checkpoint_file)
    elif backend == 'pytorch':
        model = mhp.create_model_from_checkpoint(checkpoint_file)
    else:
        raise Exception("Unknow backend: {}".format(backend))

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    return model, backend, cat_to_name

app = Flask(__name__)
model, backend, cat_to_name = setup()
if backend == 'keras':
    graph = tf.get_default_graph()

@app.route('/')
def homepage():
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data action="predict">
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method != 'POST':
        return redirect('/')
    else:
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect('/')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect('/')

        # save uploaded image as unique name
        filename = os.path.join(upload_folder, str(uuid.uuid4())) 
        file.save(filename)    

        # predict
        try:
            if backend == 'keras':
                with graph.as_default():
                    probs, classes = mhk.predict(filename, model, top_k)        
            elif backend == 'pytorch':
                probs, classes = mhp.predict(filename, model, top_k)       
        except:
            probs, classes = [], []

        # delete temp file
        os.remove(filename)

        # format results
        result = []
        for c,p in zip(classes, probs):
            if c in cat_to_name:
                c_name = cat_to_name[c].title()
            else:
                c_name = ''
            result.append({'Class' : c, 'Category' : c_name, 'Prob' : float(p)})    

        return json.dumps(result)


# run the app.
if __name__ == "__main__":
    app.run(host='0.0.0.0')