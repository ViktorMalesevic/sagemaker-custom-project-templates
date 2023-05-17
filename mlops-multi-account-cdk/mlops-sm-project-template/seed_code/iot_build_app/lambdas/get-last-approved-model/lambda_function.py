"""
This sample is non-production-ready template
© 2021 Amazon Web Services, Inc. or its affiliates. All Rights Reserved.
This AWS Content is provided subject to the terms of the AWS Customer Agreement available at
http://aws.amazon.com/agreement or other written agreement between Customer and either
Amazon Web Services, Inc. or Amazon Web Services EMEA SARL or both.
"""

import boto3
from botocore.exceptions import ClientError
import json
import logging
import traceback

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

sagemaker_client = boto3.client("sagemaker")

def describe_model_package(model_package_arn):
    try:
        model_package = sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )

        LOGGER.info("{}".format(model_package))

        if len(model_package) == 0:
            error_message = ("No ModelPackage found for: {}".format(model_package_arn))
            LOGGER.error("{}".format(error_message))

            raise Exception(error_message)

        return model_package
    except ClientError as e:
        stacktrace = traceback.format_exc()
        error_message = e.response["Error"]["Message"]
        LOGGER.error("{}".format(stacktrace))

        raise Exception(error_message)


def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            LOGGER.debug("Getting more packages for token: {}".format(response["NextToken"]))
            response = sagemaker_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = ("No approved ModelPackage found for ModelPackageGroup: {}".format(model_package_group_name))
            LOGGER.error("{}".format(error_message))

            return None
            # raise Exception(error_message)

        model_package = approved_packages[0]
        LOGGER.info("Identified the latest approved model package: {}".format(model_package))

        return model_package
    except ClientError as e:
        stacktrace = traceback.format_exc()
        error_message = e.response["Error"]["Message"]
        LOGGER.error("{}".format(stacktrace))

        raise Exception(error_message)

def lambda_handler(event, context):
    if isinstance(event, bytes):
        data = json.loads(event)

    if isinstance(event, dict):
        if "body" in event:
            if isinstance(event["body"], dict):
                data = event["body"]
            else:
                data = json.loads(event["body"])
        else:
            data = event

    LOGGER.info("{}".format(data))
    print("{}".format(data))
    
    # if we specify imagenet (or equivalent parameter) as checkpoint type and ignore the registry
    # ideally implement this in the pipeline itself, now it is not possible with conditionstep
    if "checkpoint_type" in data and data["checkpoint_type"] != "registry":
        return {
            "statusCode": "200",
            "model_version": '0',
            "model_uri": data["checkpoint_type"],
            "model_base_uri": "none",
        }

    model_package_group_name = data["model_package_group_name"]
    model_package_approved = get_approved_package(model_package_group_name)
    
    if not model_package_approved:
        return {
            "statusCode": "200",
            "model_version": '0',
            "model_uri": "none",
            "model_base_uri": "none",
        }
    
    model_package = describe_model_package(model_package_approved["ModelPackageArn"])
    model_uri = model_package['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    model_base_uri = '/'.join(model_uri.split('/')[:-2])
    
    return {
        "statusCode": "200",
        "model_version": model_package['ModelPackageVersion'],
        "model_uri": model_uri,
        "model_base_uri": model_base_uri,
    }
