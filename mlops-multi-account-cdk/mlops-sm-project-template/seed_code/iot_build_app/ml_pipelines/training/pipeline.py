"""
Workflow pipeline script for Image classification pipeline.

"""
import os
from datetime import datetime

import boto3
import sagemaker
import sagemaker.session

# from sagemaker.estimator import Estimator
from sagemaker.tensorflow import TensorFlow, TensorFlowProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet, Join

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterFloat,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CacheConfig
)

from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import (
    LambdaOutput, 
    LambdaOutputTypeEnum, 
    LambdaStep
)

from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.image_uris import retrieve

from sagemaker.workflow.pipeline_context import PipelineSession

# General caching config - modify this to your needs
cache_config = CacheConfig(enable_caching=True, expire_after="P10d")


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="ImageClassificationPackageGroup",
    pipeline_name="ImageClassificationPipeline",
    base_job_prefix="ImageClassification",
    project_name="",
    ecr_repo_uri="",
):
    """Gets a SageMaker ML Pipeline instance working on Image Classification data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = PipelineSession() # get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    if default_bucket is None:
        default_bucket = sagemaker.Session().default_bucket()
        
        
    """
    Parameters of the pipeline
    """

    # Model hyperparameters
    training_num_epochs = ParameterInteger(
        name="TrainingNumEpochs", default_value=100
    )
    batch_size = ParameterInteger(
        name="BatchSize", default_value=32
    )
    learning_rate = ParameterFloat(
        name="LearningRate", default_value=4e-3
    )
    backbone = ParameterString(
        name="Backbone", default_value="ResNet50"
    )
    optimizer = ParameterString(
        name="Optimizer", default_value="SGD"
    )
    processing_num_augmentations = ParameterString(
        name="ProcessingNumAugmentations", default_value="0"
    )
    checkpoint_type = ParameterString(
        name="CheckpointType", default_value="registry", enum_values=["registry", "imagenet"]
    )
    
    # Ops parameters
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    s3_input_dir = ParameterString(
        name="s3_input_dir", default_value=f's3://{default_bucket}/dataset'
    )
    
    
    """
    Processor step to do train/val split and data augmentation
    """
    
    s3_processing_input_mode = 'File'
    s3_training_input_mode = 'File'
    s3_upload_mode = 'Continuous'
    
    
    processing_image_uri = f'{ecr_repo_uri}:preprocessing'
    
    # pre-processing step
    preprocessor = ScriptProcessor(
        command=["python3"],
        image_uri=processing_image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        base_job_name=f'{base_job_prefix}-preprocessing',
        # volume_size_in_gb=100
    )
    
    
    step_process = ProcessingStep(
        name='Preprocessing',
        processor=preprocessor,
        code=f'source_scripts/preprocessing/create_dataset.py',
        inputs=[
            ProcessingInput(source=s3_input_dir, 
                            destination="/opt/ml/processing/input", 
                            input_name="input_set",
                            s3_input_mode=s3_processing_input_mode
                           ),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output/train", 
                             output_name="train_set",
                             s3_upload_mode=s3_upload_mode
                            ),
            ProcessingOutput(source="/opt/ml/processing/output/val", 
                             output_name="val_set",
                             s3_upload_mode=s3_upload_mode
                            ),
            ProcessingOutput(source="/opt/ml/processing/output/test", 
                             output_name="test_set",
                             s3_upload_mode=s3_upload_mode
                            ),
        ],
        job_arguments=['--num-augmentations', processing_num_augmentations],
        cache_config=cache_config
    )
    
    
    
    """
    Lambda step to get last approved version of the model.
    Notice that this will create a new Lambda or overwrite an existing one with the same name.
    """

    model_uri = LambdaOutput(output_name="model_uri", output_type=LambdaOutputTypeEnum.String)
    
    # If an ARN is not specified, a Lambda is created from scratch
    # or the existing one with same name is updated
    last_approved_lambda_name = f"{project_name}-get-last-approved-model"
    last_approved_lambda = Lambda(
        function_name=last_approved_lambda_name,
        script=f'lambdas/get-last-approved-model/lambda_function.py',
        handler='lambda_function.lambda_handler',
        execution_role_arn=role
    )

    # If checkpoint type is not 'registry', model_uri = checkpoint_type (see Lambda code)
    step_get_last_approved_model = LambdaStep(
        name="GetLastApprovedModel",
        lambda_func=last_approved_lambda,
        inputs={
            "checkpoint_type": checkpoint_type,
            "model_package_group_name": model_package_group_name
        },
        outputs=[model_uri],
        cache_config=cache_config
    )
    
    
    """
    Training step to learn a standard image classifier
    """
    
    # Training step - hyperparameters  
    hyperparameters = {'backbone': backbone,
                       'epochs': training_num_epochs,
                       'batch-size': batch_size,
                       'learning-rate': learning_rate,
                       'optimizer': optimizer,
                       'checkpoint': step_get_last_approved_model.properties.Outputs["model_uri"],
                       # use this below only if the dataset is too large and epochs too long to iterate more often
                       # 'steps-per-epoch': 1000, 
                      }
    
    # DEFINE METRICS
    # This metric below captures floating points that include exponential representations for very small numbers
    regex_number = '[-+]?[0-9]+[.]?[0-9]*(?:[eE][-+]?[0-9]+)?' # '?:' avoids capturing group
    regex = f'({regex_number})'

    metric_definitions=[{'Name': 'epochs', 'Regex':  f'Epoch {regex}'},
                    
                    {'Name': 'train:loss', 'Regex':  f' loss: {regex}'},
                    {'Name': 'val:loss', 'Regex':  f' val_loss: {regex}'},
                    
                    {'Name': 'train:accuracy', 'Regex':  f' accuracy: {regex}'},
                    {'Name': 'val:accuracy', 'Regex':  f' val_accuracy: {regex}'},
                    
                    # This one is for HPO to preserve the metric of the best model checkpoint
                    {'Name': 'val:checkpoint_metric', 'Regex': f'(?:did not improve from|from (?:{regex_number}|-inf|inf) to) {regex}'}
                    
                    ]
    
    train_estimator = TensorFlow(
        py_version='py39',
        framework_version='2.11',
        role=role,
        instance_type="ml.p3.2xlarge",
        instance_count=1,
        entry_point='train_model.py',
        source_dir=f'source_scripts/training/',
        base_job_name=f'{base_job_prefix}-training',
        hyperparameters=hyperparameters,
        model_dir='/opt/ml/model',
        metric_definitions=metric_definitions
    )
    
    
    step_train = TrainingStep(
        name=f'Training',
        estimator=train_estimator,
        inputs={
            "train": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_set"].S3Output.S3Uri,
                                   input_mode=s3_training_input_mode,
            ),
            "val": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["val_set"].S3Output.S3Uri,
                                 input_mode=s3_training_input_mode,
            )
        },
        cache_config=cache_config
    )
    

    """
    Evaluation step to record the metrics of the model
    """
    
    evaluator = TensorFlowProcessor(
        # entry_point=["python3"],
        framework_version='2.11',
        role=role,
        instance_type='ml.c5.2xlarge',
        instance_count=1,
        base_job_name=f'{base_job_prefix}-evaluate',
        py_version='py39',
        sagemaker_session=sagemaker_session
    )
    
    
    step_args = evaluator.run(
        code='evaluate.py',
        source_dir='source_scripts/evaluate',
        inputs=[ProcessingInput(
                    source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/input/model",
                    s3_input_mode=s3_processing_input_mode
                ),
                ProcessingInput(
                    source=step_process.properties.ProcessingOutputConfig.Outputs["test_set"].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/data",
                    s3_input_mode=s3_processing_input_mode
                )],
        outputs=[
                ProcessingOutput(source="/opt/ml/processing/output", 
                                 output_name="model_metrics",
                                 s3_upload_mode=s3_upload_mode
                                ),
                ],
    )
    
    
    step_eval = ProcessingStep(
        name='Evaluation',
        step_args=step_args,
        cache_config=cache_config
    )
    
    
    """
    Metrics collection and model Registration
    """
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(on='/', 
                        values=[step_eval.properties.ProcessingOutputConfig.Outputs["model_metrics"].S3Output.S3Uri, 
                                'metrics.json']
                       ),
            content_type="application/json",
        )
    )
    
    step_register = RegisterModel(
        name='RegisterModel',
        estimator=train_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["image/jpeg"],
        response_types=["image/jpeg"],
        inference_instances=["ml.c5.2xlarge"],
        transform_instances=["ml.c5.2xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    
    
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            # Model Hyperparameters
            training_num_epochs,
            batch_size,
            learning_rate,
            backbone,
            optimizer,
            processing_num_augmentations,
            # Ops parameters
            checkpoint_type,
            model_approval_status,
            s3_input_dir,
        ],
        steps=[step_process, 
               step_get_last_approved_model, 
               step_train, 
               step_register,
               step_eval],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
