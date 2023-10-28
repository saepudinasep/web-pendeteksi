from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0: 'Normal', 1: 'Pneumonia'}

# Load the Xception model (3C_Plus_Xception.h5)
model_xception = load_model('3C_Plus_Xception.h5')
model_xception.make_predict_function()

# Load the 3C model (3C.h5)
model_3c = load_model('3C.h5')
model_3c.make_predict_function()

def predict_label(img_path, model):
    # i = image.load_img(img_path, target_size=(224, 224))
    i = image.load_img(img_path, target_size=(128, 128))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 128, 128, 3)
    p = model.predict(i)
    return dic[np.argmax(p)]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        # Use the Xception model for predictions
        p_xception = predict_label(img_path, model_xception)

        # Use the 3C model for predictions
        p_3c = predict_label(img_path, model_3c)

    return render_template("classification.html", prediction_xception=p_xception, prediction_3c=p_3c, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
