import requests
import json
from io import BytesIO
import base64
from PIL import Image
import cv2
import os
import numpy as np

__testdir__ = os.path.dirname(os.path.realpath(__file__))
url = 'http://52.50.7.104:80/invocations'

# variables
with open(f'{__testdir__}/classes', 'r') as f:
    rawcat = f.read()
    class_dict = {idx_ : cat for idx_, cat in enumerate(rawcat.split('\n'))}

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, COLORS):
    label = str(class_dict[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_class_img(img_in, out_boxes, out_scores, out_classes):
    COLORS = np.random.uniform(0, 255, size=(len(class_dict.keys()), 3))

    # image_data = cv2.imread(img_in)
    for box, score, cl in zip(out_boxes, out_scores, out_classes):
        x = box[1]
        y = box[0]
        w = box[3]
        h = box[2]
        
        draw_bounding_box(img_in, cl, score, round(x), round(y), round(w), round(h), COLORS)

    # cv2.imwrite(img_out, image_data)
    
    return img_in


def getprediction():
    # Set the API endpoint URL
    
    input_img = f"{__testdir__}/data/2012-chevrolt.jpg"
    with open(input_img, 'rb') as f:
        img_bytes = f.read()

    # Encode image 
    img_str = base64.b64encode(img_bytes).decode('utf-8')

    # Define the JSON payload to send in the POST request
    payload = {
        "input_image": img_str
    }

    # Convert the payload to a JSON string
    json_payload = json.dumps(payload)

    # Set the headers for the request
    headers = {'Content-Type': 'application/json'}

    # Make the POST request with the JSON payload and headers
    response = requests.post(url, data=json_payload, headers=headers)

    # Print the response from the API
    result = eval(json.loads(response.content))
    print(response.text)

    # load image from byte string
    jpg_as_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img_in = cv2.imdecode(jpg_as_np, flags=1)

    # draw bounding boxes and class names in the original image
    img_out = get_class_img(img_in, result['boxes'], result['scores'], result['classes'])
    cv2.imshow('image', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    getprediction()

