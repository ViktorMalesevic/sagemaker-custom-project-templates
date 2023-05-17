import json
import subprocess
import sys
import argparse
import tarfile

import tf2onnx
import tensorflow as tf

from pathlib import Path

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    # install('tf2onnx')
    
    parser = argparse.ArgumentParser(description='convert TF model to ONNX')

    parser.add_argument('--input', default="/opt/ml/processing/input/", help="input model")
    parser.add_argument('--output', default="/opt/ml/processing/output/", help="output onnx model")
    
    args = parser.parse_args()
    
    # Folder structure creation
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    input_tar_path = input_dir/'model.tar.gz'
    input_extract_dir = Path('/tmp/extracted')
    input_extract_dir.mkdir(exist_ok=True)
    input_tar = tarfile.open(input_tar_path, "r:gz")
    input_tar.extractall(input_extract_dir)
    
    input_model_path = input_extract_dir/'model.h5'
    
    # assuming for now that model is a Keras H5 file with at least graph and weights
    model = tf.keras.models.load_model(input_model_path)
    
    onnx_path = output_dir/input_model_path.with_suffix('.onnx').name
    
    input_spec = (tf.TensorSpec.from_spec(model.input),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec, opset=13, output_path=onnx_path)
    
    output_names = [n.name for n in model_proto.graph.output]
    input_names = [n.name for n in model_proto.graph.input]
    input_shape = input_spec[0].shape.as_list()[1:]
    
    signature = {'input_names': input_names,
                 'output_names': output_names,
                 'input_shape': input_shape
                }
    
    signature_path = output_dir/'signature.json'
    json.dump(signature, open(signature_path, 'w'))
    
    output_tar_path = output_dir/'model.tar.gz'
    output_tar = tarfile.open(output_tar_path, "w:gz")
    output_tar.add(onnx_path, arcname=onnx_path.name)
    output_tar.add(signature_path, arcname=signature_path.name)
    output_tar.close()
    
    # cleanup output folder
    onnx_path.unlink()
    signature_path.unlink()
    
    
    print(f'Finished.')

    
if __name__ == '__main__':
    main()
    
