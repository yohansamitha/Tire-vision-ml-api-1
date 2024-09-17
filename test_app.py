import unittest
import base64
import json
from app import app


class FlaskTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = app.test_client()
        cls.app.testing = True

    def encode_image(self, image_path):
        """ Helper method to encode an image file to base64. """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

    def test_predict_valid_image(self):
        """ Test prediction with a valid image. """
        image_base64 = self.encode_image("badtire.jpg")  # Replace with actual path
        response = self.app.post('/predict', json={'imageData': image_base64})
        self.assertEqual(response.status_code, 200)
        response_json = json.loads(response.data)
        self.assertIn('result', response_json)
        self.assertIsInstance(response_json['result'], str)

    def test_predict_no_image(self):
        """ Test prediction with no image data. """
        response = self.app.post('/predict', json={})
        self.assertEqual(response.status_code, 400)
        response_json = json.loads(response.data)
        self.assertIn('error', response_json)

    def test_predict_invalid_image_data(self):
        """ Test prediction with invalid image data. """
        response = self.app.post('/predict', json={'imageData': 'invalid_base64_data'})
        self.assertEqual(response.status_code, 400)
        response_json = json.loads(response.data)
        self.assertIn('error', response_json)


if __name__ == '__main__':
    unittest.main()
