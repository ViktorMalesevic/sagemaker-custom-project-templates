import argparse
import json
import boto3
import tarfile

import numpy as np
import tensorflow as tf

from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
       
'''
TODO: create a dictionary that maps backbone names to their submodules, 
so that the right preprocess_function can be retrieved.
'''

def main():
    parser = argparse.ArgumentParser(description='train car recognition model')

    parser.add_argument('--train-input', default="/opt/ml/input/data/train", help="input train dataset with one subfolder per class")
    parser.add_argument('--val-input', default="/opt/ml/input/data/val", help="input val dataset with one subfolder per class")
    parser.add_argument('--model_dir', default="/opt/ml/model/", help="location of model artifact")
    
    parser.add_argument('--input-shape', default=224, type=int, help="size of the input model layers")
    parser.add_argument('--epochs', default=256, type=int, help="number of epochs to train the model")
    parser.add_argument('--steps-per-epoch', default=None, type=int, help="Total number of steps (batches of samples) before declaring one epoch finished")
    parser.add_argument('--batch-size', default=32, type=int, help="number of images per iteration")
    parser.add_argument('--optimizer', default='SGD', type=str, help="classification optimizer")
    parser.add_argument('--learning-rate', default=1e-3, type=float, help="classification learning rate")
    parser.add_argument('--backbone', default='ResNet50', type=str, help="tf.keras.applications backbone choice. Ignored if checkpoint is an S3 URI.")
    parser.add_argument('--checkpoint', default='imagenet', type=str, help="Keras pre-trained weights, or S3 URI with the model checkpoint to start from (in .tar.gz format)")
    
    args = parser.parse_args()
    
    # prepare folders
    # input_dir = Path(args.input)
    train_dir = Path(args.train_input)
    val_dir = Path(args.val_input)
    output_dir = Path(args.model_dir)
    
    # declare datasets
    data_generator = ImageDataGenerator()
    train_dataset = data_generator.flow_from_directory(directory=str(train_dir),
                                                       target_size=(args.input_shape, args.input_shape),
                                                       class_mode='sparse',
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       interpolation='bicubic',
                                                       keep_aspect_ratio=True
                                                      )
    
    val_dataset = data_generator.flow_from_directory(directory=str(val_dir),
                                                     target_size=(args.input_shape, args.input_shape),
                                                     class_mode='sparse',
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     interpolation='bicubic',
                                                     keep_aspect_ratio=True
                                                    )
    
    # declare model, check for checkpoint, otherwise use backbone parameter
    if args.checkpoint.startswith('s3://'):
        print(f'Downloading model S3 checkpoint. Ignoring backbone network parameter.')
        
        if not args.checkpoint.endswith('model.tar.gz'):
            raise ValueError('Provided model uri seems to be no model.tar.gz!')
            
        checkpoint = args.checkpoint[5:]
        bucket, key = checkpoint.split('/', 1)
        checkpoint_tar_path = '/tmp/model.tar.gz'
        
        print(f'Bucket {bucket}; Key {key}')
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, str(checkpoint_tar_path))
        
        checkpoint_extract_dir = Path('/tmp/extracted')
        checkpoint_extract_dir.mkdir(exist_ok=True)
        checkpoint_tar = tarfile.open(checkpoint_tar_path, "r:gz")
        checkpoint_tar.extractall(checkpoint_extract_dir)
        
        checkpoint_model_path = checkpoint_extract_dir/'model.h5'
    
        # assuming for now that model is a Keras H5 file with at least graph and weights
        model = tf.keras.models.load_model(checkpoint_model_path)
        
        # check if num classes have changed!
        if train_dataset.num_classes != model.output.shape[-1]:
            print(f'Pretrained model has {model.output.shape[-1]} classes, while the dataset has {train_dataset.num_classes}. Re-building top layers.')
            
            model.layers.pop()
            model.layers.pop()
            
            x = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
            output = tf.keras.layers.Dense(train_dataset.num_classes)(x)
            
            model = tf.keras.Model(inputs=model.input, outputs=output)
    else:
        checkpoint = args.checkpoint if args.checkpoint in ['imagenet', None] else 'imagenet'
        print(f'Downloading pretrained {checkpoint} weights. Building new backbone network')
        
        backbone = args.backbone
        image_size = args.input_shape
        num_img_channels = 3
        backbone_model = getattr(tf.keras.applications, args.backbone)
        process_function = getattr(tf.keras.applications, args.backbone.lower())

        model_input = tf.keras.Input(shape=(image_size, image_size, num_img_channels))
        # scale_input = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)(model_input)
        processed_input = process_function.preprocess_input(model_input)

        backbone_network = backbone_model(weights=checkpoint,
                                          include_top=False,
                                          input_tensor=processed_input)
        x = tf.keras.layers.GlobalAveragePooling2D()(backbone_network.output)
        output = tf.keras.layers.Dense(train_dataset.num_classes)(x)

        model = tf.keras.Model(inputs=model_input, outputs=output)
    
    model.summary()

    # Compile model
    optimizer = getattr(tf.keras.optimizers, args.optimizer)
    optimizer = optimizer(learning_rate=args.learning_rate)

    loss = [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)]
    metrics = ['accuracy']

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    
    
    # Train model.
    checkpoint_path = str(output_dir/'model.h5')
    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        monitor='val_accuracy',
                                                        mode='max',
                                                        save_best_only=True))
    
    
    
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=5, 
                                                          verbose=True))
    

    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        callbacks=callbacks,
                        verbose=2
                       )
    
    # save with weights, topology AND compilation flags
    model.load_weights(checkpoint_path)
    model.save(checkpoint_path, 
               save_format='h5', 
               include_optimizer=True)
    
    json.dump(train_dataset.class_indices, open(output_dir/'class_indices.json', 'w'))
    
    
if __name__ == '__main__':
    main()
    
