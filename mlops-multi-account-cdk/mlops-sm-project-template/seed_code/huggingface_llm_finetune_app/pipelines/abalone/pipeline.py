"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import json
import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
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
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
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
    CreateModelStep,
    TransformStep
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.step_collections import CreateModelStep, RegisterModel
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

# Added
from sagemaker.huggingface import HuggingFace, HuggingFaceModel, HuggingFaceProcessor, get_huggingface_llm_image_uri
from huggingface_hub import HfApi, HfFolder


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


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

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="LLMEcommercePackageGroup",
    pipeline_name="LLMEcommercePipeline",
    base_job_prefix="sagemaker/llm-chatbot-demo",
    processing_instance_count=1,
    processing_instance_type="ml.g4dn.12xlarge",
    training_instance_type="ml.g5.24xlarge",
    transformers_version= "4.28.1",
    pytorch_version="2.0.0",              
    py_version="py310",
    
):
    
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{default_bucket}/dataset/example_qa_dataset.json",
    )
    
    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/models"
    new_hf_model_id = ParameterString(name="HFModelId", default_value= f"{YOUR_HF_HUB}/falcon_7b_ecommerce_ai_chatbot")
    base_model_id = ParameterString(name="BaseModelId", default_value= "tiiuae/falcon-7b")
    # condition step for evaluating model quality and branching execution
    threshold_f1_score = ParameterFloat(name="ThresholdF1Score", default_value=0.8)

    
    # train_input_path = f"{base_job_prefix}/dataset/train/Ecommerce_FAQ_Chatbot_dataset.json"
    train_input_path = f's3://{sagemaker_session.default_bucket()}/{base_job_prefix}/dataset/train'
    eval_input_path= f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/dataset/eval/eval_dataset.json"
    batch_test_input_path= f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/dataset/test/test_inputs_with_params.jsonl"
    test_input_path= f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/dataset/test/test_dataset.json"
    print(train_input_path, eval_input_path)


    hyperparameters={
        "model_name":base_model_id,  # name of pretrained model
        "study_name":"falcon-chatbot",
        "n_trials": 1,
        "n_gpus":1,
        "push_to_hub": True,             # Defines if we want to push the model to the hub
        "hub_model_id": new_hf_model_id, # The model id of the model to push to the hub                 
        "hub_token": HfFolder.get_token() 
    }


    huggingface_estimator = HuggingFace(
        entry_point="train.py",                 # fine-tuning script to use in training job
        source_dir="./pipelines/abalone/scripts",                 # directory where fine-tuning script is stored
        instance_type=training_instance_type,          # instance type -> should be larger than 12xlarge otherwise provision instance error
        instance_count=processing_instance_count,                       # number of instances
        role=role,                              # IAM role used in training job to acccess AWS resources (S3)
        transformers_version=transformers_version,             # Transformers version
        pytorch_version=pytorch_version,                  # PyTorch version
        py_version=py_version,   
        output_path = model_path,
        hyperparameters=hyperparameters,         # hyperparameters to use in training job
        sagemaker_session=pipeline_session
    )
    
    step_args = huggingface_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=train_input_path
            ),
            "test": TrainingInput(
                s3_data=eval_input_path
            ),
        },
    )
    
    step_train = TrainingStep(
        name="TrainLLMEcommerceModel",
        step_args=step_args,
    )
   
    llm_image = get_huggingface_llm_image_uri(
          "huggingface",
          version="0.9.3"
    )
   
    number_of_gpu = 1
    health_check_timeout = 300
    trust_remote_code = True

    # Define Model and Endpoint configuration parameter
    config = {
      'HF_MODEL_ID': new_hf_model_id, # path to where sagemaker stores the model
      'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
      'MAX_INPUT_LENGTH': json.dumps(1024), # Max length of input text
      'MAX_TOTAL_TOKENS': json.dumps(2048), # Max length of the generation (including input text)
      'MAX_STOP_SEQUENCES':json.dumps(1),
      'HF_TRUST_REMOTE_CODE': json.dumps(trust_remote_code),
      'HF_MODEL_QUANTIZE': "bitsandbytes",# Comment in to quantize
    }
    
    output_dir = f's3://{sagemaker_session.default_bucket()}/{base_job_prefix}/outputs'
    model_dir = f's3://{sagemaker_session.default_bucket()}/{base_job_prefix}/models'
    testset_name = "test_inputs_with_params.jsonl"
    batch_test_input_url = f's3://{sagemaker_session.default_bucket()}/{base_job_prefix}/dataset/test/{testset_name}'
    test_input_url = f's3://{sagemaker_session.default_bucket()}/{base_job_prefix}/dataset/test/test_dataset.json'
    
    huggingface_model = HuggingFaceModel(
        name="LLMEcommerce",
        role=role, 
        image_uri=llm_image,
        env=config,
        sagemaker_session=pipeline_session,
    )

    step_create_model = ModelStep(
        name="LLMEcommerceCreateStep",
        step_args = huggingface_model.create(instance_type="ml.m5.large"),
    )

    transformer = sagemaker.transformer.Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        strategy='SingleRecord',
        output_path=output_dir,
        sagemaker_session=pipeline_session 
    )

    step_args = transformer.transform(
        data=batch_test_input_url,
        content_type='application/json',
        split_type='Line',
    )

    step_transform = TransformStep(
        name="LLMEcommerceTransform",
        step_args=step_args,
    )
    
    
    script_eval = HuggingFaceProcessor(
        role=role, 
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        transformers_version=transformers_version,
        pytorch_version=pytorch_version, 
        py_version=py_version,
        base_job_name=f"{base_job_prefix}/evaluation",
        sagemaker_session=pipeline_session,
    )

    evaluation_report = PropertyFile(
        name="LLMEcommerceEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateLLMEcommerceModel",
        processor=script_eval,
        inputs=[ 
            ProcessingInput(
                source=step_transform.properties.TransformOutput.S3OutputPath,
                destination="/opt/ml/input/data/test",
            ), 
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", 
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/evaluation_report",
            ),
        ],
        job_arguments=[
            "--hf_model_id", new_hf_model_id,
            "--testset_filename", testset_name, #
        ],
        code=os.path.join(BASE_DIR, "scripts/evaluate_llm.py"),
        property_files=[evaluation_report],
    )

    step_register = RegisterModel(
        name="LLMEcommerceModel-register",
        model=huggingface_model,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.g4dn.xlarge", "ml.g4dn.8xlarge"], # add the ones that fit for your account
        transform_instances=["ml.g4dn.xlarge", "ml.g4dn.8xlarge"], # add the ones that fit for your account
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
    )
    
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="Q&A_metrics.bert_score.f1_score",
        ),
        right=threshold_f1_score,
    )

    step_cond = ConditionStep(
        name="CheckHuggingfaceEvalF1Score",
        conditions=[cond_gte],
        if_steps=[step_register],
        # if_steps=[step_register, step_deployment],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            threshold_f1_score,
            new_hf_model_id,
            base_model_id,
            input_data,
        ],
       
        steps=[step_create_model, step_transform, step_eval, step_cond],
        # steps=[step_train, step_transform],
        sagemaker_session=pipeline_session,
    )
    return pipeline
