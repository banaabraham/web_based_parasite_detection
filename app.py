import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, render_template, request
from werkzeug import secure_filename
from os.path import join, dirname, realpath
import cv2
import numpy as np
from keras import models
from sklearn.cluster import MeanShift
from scipy import ndimage
from nms import non_max_suppression_fast
from keras import models
import tensorflow as tf
from plasmodium_detector import *


global graph
graph = tf.get_default_graph()
global model
model = models.load_model("weights_dir")
model._make_predict_function()


UPLOADS_PATH = join(dirname(realpath(__file__)), 'static/uploads/')

application = Flask(__name__)

application.config['UPLOAD_FOLDER'] = UPLOADS_PATH

application.config['MAX_CONTENT_PATH'] = 1000000000

@application.route("/",methods=["GET","POST"])
def index():
    if request.method == "POST":
        f = request.files['file']
        filename =  application.config['UPLOAD_FOLDER'] + \
                   '\\' + secure_filename(f.filename)
        try:
            f.save(filename)
            img = cv2.imread(filename)
            generatePrediction(img,filename)
            image_name = ('static/uploads/' + f.filename)
            return render_template("image.html",image_name=image_name)
        except Exception as e:
            print(e)
            pass
    return render_template("form.html")

if __name__ == '__main__':
    application.run(debug=True)
