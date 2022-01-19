from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import base64
import uuid
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = None
def AiApp():
    app = Flask(__name__)
    def init():
        global model
        model = load_model("model.h5", compile=False)
    init()
    return app

app = AiApp()

@app.route('/')
def home():
  return jsonify({"Onwer": "Created by Muhammad Hanan Asghar"})

@app.route("/predict", methods=['POST'])
def predict():
    pic_data = request.form['URI']
    unique_filename = str(uuid.uuid4())
    pic_data = pic_data.replace("data:image/jpeg;base64,", "")
    pic_data = pic_data.replace("data:image/png;base64,", "")
    pic_data = pic_data.replace("data:image/jpg;base64,", "")
    imgdata = base64.b64decode(pic_data)
    filename = f'{unique_filename}.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = tf.expand_dims(image, 0)
    scene = model.predict(image)
    os.remove(filename)
    prediction = np.argmax(scene)
    cond = ""
    if prediction == 0:
      cond="Buildings"
    if prediction == 1:
      cond="Forest"
    if prediction == 2:
      cond="Glacier"
    if prediction == 3:
      cond="Mountain"
    if prediction == 4:
      cond="Sea"
    if prediction == 5:
      cond="Street"

    return jsonify({"value":cond})

if __name__ == '__main__':
  app.run()