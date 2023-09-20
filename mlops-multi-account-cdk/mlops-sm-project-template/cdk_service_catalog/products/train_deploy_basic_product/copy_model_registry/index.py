# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import os

import boto3

sm_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")

bucket = os.environ.get("artefact_bucket")
prefix = f"{os.environ.get('prefix')}/"


def lambda_handler(event, context):
    source_model_package_arn = event["detail"]["ModelPackageArn"]
    model_package = sm_client.describe_model_package(ModelPackageName=source_model_package_arn)

    for idx, container in enumerate(
            model_package["InferenceSpecification"]["Containers"]
    ):
        model_data = container["ModelDataUrl"].split("s3://")[1]
        print(model_data)

        key = prefix + model_data.split("/", 1)[1]
        s3_client.copy_object(Bucket=bucket, CopySource=model_data, Key=key)
        model_data_url = f"s3://{bucket}/{key}"
        model_package["InferenceSpecification"]["Containers"][idx][
            "ModelDataUrl"
        ] = model_data_url
        try:
            del model_package["InferenceSpecification"]["Containers"][idx][
                "ImageDigest"
            ]
        except:
            pass
        if idx == 0:
            if "ModelMetrics" in model_package.keys():
                try:
                    metric_source = model_package["ModelMetrics"]["ModelQuality"][
                        "Statistics"
                    ]["S3Uri"].split("s3://")[1]
                    key_metrics = prefix + str(metric_source.split("/", 1)[1])
                    s3_client.copy_object(
                        Bucket=bucket, CopySource=metric_source, Key=key_metrics
                    )
                    metric_source_url = f"s3://{bucket}/{key_metrics}"
                    model_package["ModelMetrics"]["ModelQuality"]["Statistics"][
                        "S3Uri"
                    ] = metric_source_url
                except:
                    pass

                try:
                    bias_source = model_package["ModelMetrics"]["Bias"]["Report"][
                        "S3Uri"
                    ].split("s3://")[1]
                    key_metrics = prefix + str(bias_source.split("/", 1)[1])
                    s3_client.copy_object(
                        Bucket=bucket, CopySource=bias_source, Key=key_metrics
                    )
                    bias_source_url = "s3://" + bucket + "/" + key_metrics
                    model_package["ModelMetrics"]["Bias"]["Report"][
                        "S3Uri"
                    ] = bias_source_url
                except:
                    pass

                try:
                    exp_source = model_package["Explainability"]["Report"][
                        "S3Uri"
                    ].split("s3://")[1]
                    key_metrics = prefix + str(exp_source.split("/", 1)[1])
                    s3_client.copy_object(
                        Bucket=bucket, CopySource=exp_source, Key=key_metrics
                    )
                    exp_source_url = "s3://" + bucket + "/" + key_metrics
                    model_package["ModelMetrics"]["Explainability"]["Report"][
                        "S3Uri"
                    ] = exp_source_url
                except:
                    pass

    sm_client.create_model_package(
        ModelPackageGroupName=os.environ.get("target_model_package_group"),
        InferenceSpecification=model_package["InferenceSpecification"],
        ModelApprovalStatus="PendingManualApproval",
        ModelMetrics=model_package["ModelMetrics"]
        if "ModelMetrics" in model_package.keys()
        else {},
        ModelPackageDescription="package version "
                                + event["detail"]["ModelPackageArn"].split("/")[-1],
    )
    return {"statusCode": 200, "body": json.dumps("Copied and Registered Model!")}
