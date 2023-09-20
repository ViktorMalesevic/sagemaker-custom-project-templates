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


import os

import aws_cdk
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    Stage,
    Tags,
    aws_codecommit as codecommit,
    pipelines as pipelines,
    aws_s3 as s3,
    aws_iam as iam,
    aws_kms as kms,
)
from constructs import Construct
from cdk_service_catalog.products.constructs.zip_utils import create_zip
from cdk_service_catalog.sm_service_catalog import SageMakerServiceCatalog


class SageMakerServiceCatalogStage(Stage):
    def __init__(
            self, scope: Construct, construct_id: str,
            **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        service = SageMakerServiceCatalog(
            self,
            "template",
            **kwargs,
        )


class CdkPipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        hub_account = self.node.try_get_context("hub_account")
        hub_region = self.node.try_get_context("hub_region")

        base_dir: str = os.path.abspath(f'{os.path.dirname(__file__)}{os.path.sep}..')
        repo_branch_name: str = "main"
        repo = codecommit.Repository(
            self,
            "mlops-sm-project-template-repo",
            repository_name="mlops-sm-project-template-repo",
            description="CDK Code with Sagemaker Projects Service Catalog products",
            code=codecommit.Code.from_zip_file(file_path=create_zip(base_dir), branch=repo_branch_name)
        )

        artifact_bucket = self.create_pipeline_artifact_bucket(hub_account=hub_account)

        pipeline = pipelines.CodePipeline(
            self,
            "Pipeline",
            pipeline_name="mlops-sm-project-template-pipeline",
            synth=pipelines.ShellStep(
                "Synth",
                input=pipelines.CodePipelineSource.code_commit(repo, repo_branch_name),
                install_commands=[
                    "npm install -g aws-cdk",
                    "pip install -r requirements.txt",
                ],
                commands=[
                    f"cdk synth --context hub_account={hub_account} --context hub_region={hub_region}",
                ],
            ),
            cross_account_keys=True,
            artifact_bucket=artifact_bucket,

        )

        pipeline.add_stage(
            SageMakerServiceCatalogStage(
                self,
                "mlops-sm-project",
                env={
                    "account": hub_account,
                    "region": hub_region,
                },
            )
        )

        # General tags applied to all resources created on this scope (self)
        Tags.of(self).add("key", "value")

    def create_pipeline_artifact_bucket(self, hub_account: str) -> s3.Bucket:
        # create kms key to be used by the assets bucket
        kms_key = kms.Key(
            self,
            "MLOpsPipelineArtifactsBucketKMSKey",
            description="key used for encryption of data in Amazon S3",
            enable_key_rotation=True,
            policy=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        actions=["kms:*"],
                        effect=iam.Effect.ALLOW,
                        resources=["*"],
                        principals=[iam.AccountRootPrincipal()],
                    )
                ]
            ),
        )

        # allow cross account access to the kms key
        kms_key.add_to_resource_policy(
            iam.PolicyStatement(
                actions=[
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:ReEncrypt*",
                    "kms:GenerateDataKey*",
                    "kms:DescribeKey",
                ],
                resources=[
                    "*",
                ],
                principals=[
                    iam.ArnPrincipal(f"arn:aws:iam::{hub_account}:root"),
                ],
            )
        )

        s3_artifact = s3.Bucket(
            self,
            "MLOpsSmTemplatePipelineArtifactBucket",
            bucket_name=f"mlops-sm-template-pipeline-artifact-bucket",  # Bucket name has a limit of 63 characters
            encryption_key=kms_key,
            versioned=True,
            auto_delete_objects=True,
            removal_policy=aws_cdk.RemovalPolicy.DESTROY,
        )

        # Block insecure requests to the bucket
        s3_artifact.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowSSLRequestsOnly",
                actions=["s3:*"],
                effect=iam.Effect.DENY,
                resources=[
                    s3_artifact.bucket_arn,
                    s3_artifact.arn_for_objects(key_pattern="*"),
                ],
                conditions={"Bool": {"aws:SecureTransport": "false"}},
                principals=[iam.AnyPrincipal()],
            )
        )

        # Tooling account access to objects in the bucket
        s3_artifact.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AddToolingPermissions",
                actions=["s3:*"],
                resources=[
                    s3_artifact.arn_for_objects(key_pattern="*"),
                    s3_artifact.bucket_arn,
                ],
                principals=[
                    iam.ArnPrincipal(f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:root"),
                ],
            )
        )

        # Hub account access to objects in the bucket
        s3_artifact.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AddCrossAccountPermissions",
                actions=["s3:List*", "s3:Get*", "s3:Put*"],
                resources=[
                    s3_artifact.arn_for_objects(key_pattern="*"),
                    s3_artifact.bucket_arn,
                ],
                principals=[
                    iam.ArnPrincipal(f"arn:aws:iam::{hub_account}:root"),
                ],
            )
        )

        return s3_artifact
