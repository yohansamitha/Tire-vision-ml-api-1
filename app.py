import base64
import io

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, make_response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

model = load_model('best_model.h5')

IMAGE_SIZE = (150, 150)


def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'imageData' not in data:
            response = make_response(jsonify({'error': 'No image provided 1'}), 400)
            response.headers['Content-Type'] = 'application/json'
            return response

        image_data = data['imageData']

        if image_data.startswith('data:image/jpeg;base64,'):
            image_data = image_data.split('data:image/jpeg;base64,')[-1]
        elif image_data.startswith('data:image/png;base64,'):
            image_data = image_data.split('data:image/png;base64,')[-1]

        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes))

        processed_image = preprocess_image(image, IMAGE_SIZE)

        prediction = model.predict(processed_image)

        percentage = prediction[0][0] * 100
        rounded_percentage = round(percentage)
        formatted_percentage = f"{rounded_percentage:02d}"
        print("formatted_percentage : ", formatted_percentage, " | ")

        response = make_response(jsonify({'result': formatted_percentage}), 200)
        response.headers['Content-Type'] = 'application/json'
        return response

    except Exception as e:
        response = make_response(jsonify({'error': str(e)}), 400)
        response.headers['Content-Type'] = 'application/json'
        return response


if __name__ == '__main__':
    app.run(debug=True)
