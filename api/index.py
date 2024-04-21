import pytesseract
import numpy as np
import cv2
import json
import base64
import io
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

import tensorflow as tf

app = Flask(__name__)

classes = {0: 'a',
 1: 'b',
 2: 'c',
 3: 'd',
 4: 'e',
 5: 'f',
 6: 'g',
 7: 'h',
 8: 'i',
 9: 'j',
 10: 'k',
 11: 'l',
 12: 'm',
 13: 'n',
 14: 'o',
 15: 'p',
 16: 'q',
 17: 'r',
 18: 's',
 19: 't',
 20: 'u',
 21: 'v',
 22: 'w',
 23: 'x',
 24: 'y',
 25: 'z'}

model = load_model('codenotes.keras')

@app.route('/api/ocr', methods=["POST"])
def ocr():
    print("hello world")
    base64_image = request.data.decode("utf-8").split(",")[1]
    decoded = base64.b64decode(base64_image)
    img_array = np.fromstring(decoded, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5,5), 0)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    characters = ""

    for index in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[index])
        letter = image[y: y + h, x: x + w]
        letter = cv2.resize(letter, (28, 28))

        letter = cv2.bitwise_not(letter)

        letter = tf.expand_dims(letter, 0)

        predictions = model.predict(letter, verbose = 0)
        predicted_class_name = classes[np.argmax(predictions, axis=1)[0]]

        characters += predicted_class_name

        print(predicted_class_name)


    return jsonify(characters)
