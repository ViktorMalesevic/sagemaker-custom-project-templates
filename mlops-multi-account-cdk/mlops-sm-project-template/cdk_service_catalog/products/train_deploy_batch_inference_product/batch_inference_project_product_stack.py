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
from aws_cdk import (
    Aws,
    Tags,
    aws_s3 as s3,
    aws_iam as iam,
    aws_kms as kms,
    aws_sagemaker as sagemaker,
    aws_servicecatalog as sc,
    aws_codecommit as codecommit,
    aws_lambda as _lambda,
)
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from constructs import Construct

from cdk_service_catalog.products.constructs.build_pipeline import BuildPipelineConstruct
from cdk_service_catalog.products.constructs.deploy_pipeline import DeployPipelineConstruct
from cdk_service_catalog.products.constructs.ssm import SSMConstruct

from cdk_service_catalog.products.constructs.zip_utils import create_zip

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class MLOpsStack(sc.ProductStack):
    DESCRIPTION: str = ("This template includes a build and a deploy code repository (CodeCommit) associated "
                        "to their respective CICD pipeline (CodePipeline). The build repository and CICD pipeline "
                        "are used to run SageMaker pipeline(s) in dev and promote the pipeline definition to an "
                        "artefact bucket. The deploy repository and CICD pipeline loads the artefact SageMaker "
                        "pipeline definition to create a Sagemaker pipeline in preprod and production as "
                        "infrastructure as code (eg for batch inference). The target PREPROD/PROD accounts are "
                        "provided as cloudformation parameters and must be provided during project creation. "
                        "The PREPROD/PROD accounts need to be cdk bootstraped in advance to have the right "
                        "CloudFormation execution cross account roles.")

    TEMPLATE_NAME: str = ("MLOps template to build and deploy SageMaker pipeline(s) cross-account "
                          "with parametrized accounts")

    SUPPORT_EMAIL: str = 'batch_inference_project@example.com'

    SUPPORT_URL: str = 'https://example.com/support/batch_inference_project'

    SUPPORT_DESCRIPTION: str = ('Example of support details for batch inference project'
                                )

    @classmethod
    def get_description(cls) -> str:
        return cls.DESCRIPTION

    @classmethod
    def get_support_email(cls) -> str:
        return cls.SUPPORT_EMAIL

    @classmethod
    def get_product_name(cls) -> str:
        return cls.TEMPLATE_NAME

    @classmethod
    def get_support_url(cls) -> str:
        return cls.SUPPORT_URL

    @classmethod
    def get_support_description(cls) -> str:
        return cls.SUPPORT_DESCRIPTION

    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            asset_bucket: s3.Bucket = None,
            **kwargs
    ) -> None:
        super().__init__(scope, construct_id, asset_bucket=asset_bucket, **kwargs)

        # Define required parameters
        project_name = aws_cdk.CfnParameter(
            self,
            "SageMakerProjectName",
            type="String",
            description="The name of the SageMaker project.",
            min_length=1,
            max_length=32,
        ).value_as_string

        project_id = aws_cdk.CfnParameter(
            self,
            "SageMakerProjectId",
            type="String",
            min_length=1,
            max_length=16,
            description="Service generated Id of the project.",
        ).value_as_string

        # TODO: derive account number from model package ARN
        source_account = aws_cdk.CfnParameter(
            self,
            "SourceDevAccount",
            type="String",
            min_length=11,
            max_length=13,
            description="Id of source - dev account.",
        ).value_as_string

        preprod_account = aws_cdk.CfnParameter(
            self,
            "PreProdAccount",
            type="String",
            min_length=11,
            max_length=13,
            description="Id of preprod account.",
        ).value_as_string

        prod_account = aws_cdk.CfnParameter(
            self,
            "ProdAccount",
            type="String",
            min_length=11,
            max_length=13,
            description="Id of prod account.",
        ).value_as_string

        deployment_region = aws_cdk.CfnParameter(
            self,
            "DeploymentRegion",
            type="String",
            min_length=8,
            max_length=10,
            description="Deployment region for preprod and prod account.",
        ).value_as_string

        Tags.of(self).add("sagemaker:project-id", project_id)
        Tags.of(self).add("sagemaker:project-name", project_name)

        source_artifact_buket = asset_bucket

        SSMConstruct(
            self,
            "MLOpsSSM",
            project_name=project_name,
            preprod_account=preprod_account,
            prod_account=prod_account,
            deployment_region=deployment_region,  # Modify when x-region is enabled
        )

        build_app_repository = codecommit.Repository(
            self,
            "BuildRepo",
            repository_name=f"{project_name}-{construct_id}-build",
            code=codecommit.Code.from_zip_file(
                create_zip(f"{BASE_DIR}/seed_code/build_app"),
                branch="main",
            ),
        )

        deploy_app_repository = codecommit.Repository(
            self,
            "DeployRepo",
            repository_name=f"{project_name}-{construct_id}-deploy",
            code=codecommit.Code.from_zip_file(
                create_zip(f"{BASE_DIR}/seed_code/deploy_app"),
                branch="main",
            ),
        )

        Tags.of(deploy_app_repository).add(key="sagemaker:project-id", value=project_id)
        Tags.of(deploy_app_repository).add(
            key="sagemaker:project-name", value=project_name
        )

        # create kms key to be used by the assets bucket
        kms_key = kms.Key(
            self,
            "ArtifactsBucketKMSKey",
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
                    iam.ArnPrincipal(f"arn:aws:iam::{preprod_account}:root"),
                    iam.ArnPrincipal(f"arn:aws:iam::{prod_account}:root"),
                ],
            )
        )

        s3_artifact = s3.Bucket(
            self,
            "S3Artifact",
            bucket_name=f"mlops-{project_name}-{Aws.ACCOUNT_ID}",  # Bucket name has a limit of 63 characters
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

        # DEV account access to objects in the bucket
        s3_artifact.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AddDevPermissions",
                actions=["s3:*"],
                resources=[
                    s3_artifact.arn_for_objects(key_pattern="*"),
                    s3_artifact.bucket_arn,
                ],
                principals=[
                    iam.ArnPrincipal(f"arn:aws:iam::{Aws.ACCOUNT_ID}:root"),
                ],
            )
        )

        # PROD account access to objects in the bucket
        s3_artifact.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AddCrossAccountPermissions",
                actions=["s3:List*", "s3:Get*", "s3:Put*"],
                resources=[
                    s3_artifact.arn_for_objects(key_pattern="*"),
                    s3_artifact.bucket_arn,
                ],
                principals=[
                    iam.ArnPrincipal(f"arn:aws:iam::{preprod_account}:root"),
                    iam.ArnPrincipal(f"arn:aws:iam::{prod_account}:root"),
                ],
            )
        )

        model_package_group_name = f"{project_name}-{project_id}"

        # cross account model registry resource policy
        model_package_group_policy = iam.PolicyDocument(
            statements=[
                iam.PolicyStatement(
                    sid="ModelPackageGroup",
                    actions=[
                        "sagemaker:DescribeModelPackageGroup",
                    ],
                    resources=[
                        f"arn:aws:sagemaker:{Aws.REGION}:{Aws.ACCOUNT_ID}:model-package-group/{model_package_group_name}"
                    ],
                    principals=[
                        iam.ArnPrincipal(f"arn:aws:iam::{preprod_account}:root"),
                        iam.ArnPrincipal(f"arn:aws:iam::{prod_account}:root"),
                    ],
                ),
                iam.PolicyStatement(
                    sid="ModelPackage",
                    actions=[
                        "sagemaker:DescribeModelPackage",
                        "sagemaker:ListModelPackages",
                        "sagemaker:UpdateModelPackage",
                        "sagemaker:CreateModel",
                    ],
                    resources=[
                        f"arn:aws:sagemaker:{Aws.REGION}:{Aws.ACCOUNT_ID}:model-package/{model_package_group_name}/*"
                    ],
                    principals=[
                        iam.ArnPrincipal(f"arn:aws:iam::{preprod_account}:root"),
                        iam.ArnPrincipal(f"arn:aws:iam::{prod_account}:root"),
                    ],
                ),
            ]
        ).to_json()

        model_package_group = sagemaker.CfnModelPackageGroup(
            self,
            "ModelPackageGroup",
            model_package_group_name=model_package_group_name,
            model_package_group_description=f"Model Package Group for {project_name}",
            model_package_group_policy=model_package_group_policy,
            tags=[
                aws_cdk.CfnTag(key="sagemaker:project-id", value=project_id),
                aws_cdk.CfnTag(key="sagemaker:project-name", value=project_name),
            ],
        )

        kms_key = kms.Key(
            self,
            "PipelineBucketKMSKey",
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

        pipeline_artifact_bucket = s3.Bucket(
            self,
            "PipelineBucket",
            bucket_name=f"pipeline-{project_name}-{Aws.ACCOUNT_ID}",  # Bucket name has a limit of 63 characters
            encryption_key=kms_key,
            versioned=True,
            auto_delete_objects=True,
            removal_policy=aws_cdk.RemovalPolicy.DESTROY,
        )

        # Create the Lambda function
        copy_model_function = _lambda.Function(
            self,
            "CopyModelFunction",
            code=_lambda.Code.from_asset(f"{BASE_DIR}/copy_model_registry"),
            environment={
                "artefact_bucket": s3_artifact.bucket_name,
                "target_model_package_group": model_package_group_name,  # model_package_group,  # TODO: Test if working
                "prefix": model_package_group_name,
            },
            handler="index.lambda_handler",
            runtime=_lambda.Runtime.PYTHON_3_8,
            timeout=aws_cdk.Duration.seconds(360),
            initial_policy=[
                iam.PolicyStatement(
                    actions=[
                        "sagemaker:DescribeModelPackage",
                        "sagemaker:DescribeModelPackageGroup",
                        "sagemaker:ListModelPackages",
                    ],
                    effect=iam.Effect.ALLOW,
                    resources=[
                        f"arn:aws:sagemaker:{Aws.REGION}:{source_account}:model-package-group/{model_package_group_name}",
                        f"arn:aws:sagemaker:{Aws.REGION}:{source_account}:model-package/{model_package_group_name}/*",
                        f"arn:aws:sagemaker:{Aws.REGION}:{Aws.ACCOUNT_ID}:model-package-group/{model_package_group_name}",
                        f"arn:aws:sagemaker:{Aws.REGION}:{Aws.ACCOUNT_ID}:model-package/{model_package_group_name}/*",
                    ],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sagemaker:CreateModelPackageGroup",
                        "sagemaker:CreateModelPackage",
                    ],
                    resources=[
                        f"arn:aws:sagemaker:{Aws.REGION}:{Aws.ACCOUNT_ID}:model-package-group/{model_package_group_name}",
                        f"arn:aws:sagemaker:{Aws.REGION}:{Aws.ACCOUNT_ID}:model-package/{model_package_group_name}/*",
                    ],
                ),
                iam.PolicyStatement(
                    actions=[
                        "kms:Decrypt",
                        "kms:Encrypt",
                    ],
                    effect=iam.Effect.ALLOW,
                    resources=[
                        "*"  # We are permissive for now because we do not know in advance the kms key from the Dev
                        # account
                    ],
                )
            ],
        )
        # For KMS, we will need to re-encrypt in the KMS key of the central model registry/central model s3 (the one
        # that this SC product creates)
        source_artifact_buket.grant_read(copy_model_function)
        s3_artifact.grant_read_write(copy_model_function)

        # Create the EventBridge rule
        # PRE-REQUISITES: Source account should already be writing
        # to target account event bus
        # For setup, see: https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-cross-account.html
        event_pattern = {
            "source": ["aws.sagemaker"],
            "detail_type": ["SageMaker Model Package State Change"],
            "detail": {
                "ModelPackageGroupName": [model_package_group_name],
                "ModelApprovalStatus": ["Approved"],
            },
        }

        copy_target = targets.LambdaFunction(
            handler=copy_model_function,
            # dead_letter_queue_enabled=True,
            retry_attempts=2,
        )

        copy_rule = events.Rule(  # noqa: F841
            self,
            "CopyEventBridgeRule",
            description="Trigger Copy Lambda function when source SageMaker Model Package Group version state changes",
            enabled=True,
            event_pattern=event_pattern,
            targets=[copy_target],
        )

        # Create SageMaker Model Cards lambda
        # Create the IAM role for Lambda function
        model_card_role = iam.Role(
            self,
            "ModelCardRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ServicePrincipal("lambda.amazonaws.com"),
            ),
            description="Allows Lambda function to access S3 and SageMaker resources",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
            ],
        )

        # Add s3 permissions to role
        model_card_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:PutObject",
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                    "s3:ListBucket",
                ],
                resources=[
                    s3_artifact.arn_for_objects(key_pattern="*"),
                    s3_artifact.bucket_arn,
                ],
            )
        )

        # Create the Lambda function
        model_card_function = _lambda.Function(
            self,
            "ModelCardFunction",
            code=_lambda.Code.from_asset(f"{BASE_DIR}/create_model_card"),
            environment={
                "role": model_card_role.role_arn,
            },
            handler="index.lambda_handler",
            role=model_card_role,
            runtime=_lambda.Runtime.PYTHON_3_8,
            timeout=aws_cdk.Duration.seconds(360),
        )

        # Create the EventBridge rule pattern to listen to Model Package Group
        event_pattern = {
            "source": ["aws.sagemaker"],
            "detail_type": ["SageMaker Model Package State Change"],
            "detail": {
                "ModelPackageGroupName": [model_package_group.model_package_group_name],
                "ModelApprovalStatus": ["Approved"],
            },
        }

        # Create Rule Target to Lambda
        model_card_target = targets.LambdaFunction(
            handler=model_card_function,
            # dead_letter_queue_enabled=True,
            retry_attempts=2,
        )

        # Create Event Rule and add Target
        model_card_rule = events.Rule(  # noqa: F841
            self,
            "ModelCardLambdaRule",
            description=(
                "Trigger Model Card Lambda function when target SageMaker "
                "Model Package Group version state changes"
            ),
            enabled=True,
            event_pattern=event_pattern,
            targets=[model_card_target],
        )

        BuildPipelineConstruct(
            self,
            "build",
            project_name=project_name,
            project_id=project_id,
            pipeline_artifact_bucket=pipeline_artifact_bucket,
            model_package_group_name=model_package_group_name,
            repository=build_app_repository,
            s3_artifact=s3_artifact
        )

        DeployPipelineConstruct(
            self,
            "deploy",
            project_name=project_name,
            project_id=project_id,
            pipeline_artifact_bucket=pipeline_artifact_bucket,
            model_package_group_name=model_package_group_name,
            repository=deploy_app_repository,
            preprod_account=preprod_account,
            prod_account=prod_account,
            deployment_region=deployment_region,
            create_model_event_rule=True,
        )
