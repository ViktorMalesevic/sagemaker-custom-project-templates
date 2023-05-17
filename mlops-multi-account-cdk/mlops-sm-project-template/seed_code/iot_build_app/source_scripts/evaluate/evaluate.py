import json
import subprocess
import sys
import argparse
import tarfile

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path


def main():
   
    parser = argparse.ArgumentParser(description='evaluate TF model')

    parser.add_argument('--input', default="/opt/ml/processing/input/model/", help="input model")
    parser.add_argument('--data', default="/opt/ml/processing/input/data/", help="input data")
    parser.add_argument('--output', default="/opt/ml/processing/output/", help="output metrics")
    
    args = parser.parse_args()
    
    # Folder structure creation
    input_dir = Path(args.input)
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    input_tar_path = input_dir/'model.tar.gz'
    input_extract_dir = Path('/tmp/extracted')
    input_extract_dir.mkdir(exist_ok=True)
    input_tar = tarfile.open(input_tar_path, "r:gz")
    input_tar.extractall(input_extract_dir)
    
    input_model_path = input_extract_dir/'model.h5'
    
    # assuming  that model is a Keras H5 file with graph, weights, and compilation flags
    model = tf.keras.models.load_model(input_model_path)
    
    data_generator = ImageDataGenerator()
    dataset = data_generator.flow_from_directory(directory=str(data_dir),
                                                 target_size=(model.input.shape[1], model.input.shape[2]),
                                                 class_mode='sparse',
                                                 batch_size=32,
                                                 shuffle=False,
                                                 interpolation='bicubic',
                                                 keep_aspect_ratio=True
                                                )

    # check model metrics at training time to get them here
    loss, accuracy = model.evaluate(dataset)
    
    
    print(f'Generating metrics.json.')
    hp = {'metrics': {
              'loss': {
                  'value': loss
                  },
              'accuracy':{
                  'value': accuracy
                  }
            }
         }
    json.dump(hp, open(output_dir/'metrics.json', 'w'))

    
if __name__ == '__main__':
    main()
    
