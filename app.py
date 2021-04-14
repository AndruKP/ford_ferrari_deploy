import os
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.keras import EfficientNetB7

IMG_SIZE = 224  # какого размера подаем изображения в сеть
IMG_CHANNELS = 3  # у RGB 3 канала
input_shape = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
class_names = [
    'Приора',  # 0
    'Ford Focus',  # 1
    'Самара',  # 2
    'ВАЗ-2110',  # 3
    'Жигули',  # 4
    'Нива',  # 5
    'Калина',  # 6
    'ВАЗ-2109',  # 7
    'Volkswagen Passat',  # 8
    'ВАЗ-21099'  # 9
]

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.trainable = True
loaded_model.load_weights("model/best_model.hdf5")
print("Loaded model from disk")
datagen = ImageDataGenerator(rescale=1. / 255)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(filename):
    img = image.load_img(filename, target_size=(IMG_SIZE, IMG_SIZE))
    input_arr = image.img_to_array(img)
    return np.array([input_arr])


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', result='no file apart')

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', result='no selected file')

        if file and not allowed_file(file.filename):
            return render_template('index.html', result='not allowed extension')

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filename)

            sub_generator = datagen.flow(preprocess_image(filename), shuffle=False)
            sub_generator.reset()

            predictions = loaded_model.predict_generator(sub_generator, steps=len(sub_generator), verbose=1)
            predictions = np.argmax(predictions, axis=-1)  # multiple categories
            result = class_names[predictions[0]]

            os.remove(filename)

            return render_template('index.html', result=result, response=1)
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
