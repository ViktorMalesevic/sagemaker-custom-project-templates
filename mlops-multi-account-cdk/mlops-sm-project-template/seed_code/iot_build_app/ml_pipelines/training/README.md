# Training SageMaker Pipeline

This SageMaker Pipeline definition creates a workflow that will:
- Prepare the an image classification dataset through a SageMaker Processing Job
- Train a Tensorflow image classification model on the train set
- Evaluate the performance of the trained model algorithm on the validation set
- The model is set for Manual Approval to SageMaker Model Registry.
