import src.processing as processing
from flask import Flask
import flask
from PIL import Image
import sys
import numpy as np
import sys
import json
import os
import time
import onnxruntime
import base64
from io import BytesIO
import argparse

parser = argparse.ArgumentParser("Performs car recognition on vehicle images.")

parser.add_argument(
    "-n",
    "--model-name",
    dest="model_name",
    type=str,
    help="The name of the model to import.",
)

args = parser.parse_args()

__root__ = os.path.dirname(os.path.realpath(__file__))
model = args.model_name

# init inference session
session = onnxruntime.InferenceSession(model, None)
inputs = [inp.name for inp in session.get_inputs()]
outputs = [out.name for out in session.get_outputs()]

# definition of Flask app for serving predictions
app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    health = session is not None
    status = 200 if health else 404
    msg = "Model server is running!" if health else "Model server down."
    return flask.Response(response= f'{msg}\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def main():
    # acquire image and preprocess
    input_json = flask.request.get_json()
    serial_img = input_json['input_image']
    img_bytes = base64.b64decode(serial_img.encode('utf-8'))
    image = Image.open(BytesIO(img_bytes))

    # data preparation
    image_data = processing.preprocess(image)
    image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)

    boxes, scores, indices = session.run(outputs, {inputs[0]: image_data, inputs[1]: image_size})
    out_boxes, out_scores, out_classes = processing.postprocess(boxes, scores, indices)

    # Transform predictions to JSON
    result = {
        'boxes': [list(el) for el in out_boxes],
        'scores': out_scores,
        'classes': out_classes
        }

    resultjson = json.dumps(str(result))
    return flask.Response(response=resultjson, status=200, mimetype='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
