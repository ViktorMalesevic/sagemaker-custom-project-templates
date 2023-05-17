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

from aws_cdk import (
    Aws,
    CfnDynamicReference,
    CfnDynamicReferenceService,
    Stack,
    Tags,
    aws_s3 as s3,
    aws_iam as iam,
    aws_kms as kms,
    aws_ecr as ecr,
    aws_sagemaker as sagemaker,
)

import aws_cdk

from constructs import Construct

from mlops_sm_project_template.templates.ssm_construct import SSMConstruct

from mlops_sm_project_template.templates.byoc_pipeline_constructs.iot_build_pipeline_construct import (
    BuildPipelineConstruct,
)
from mlops_sm_project_template.templates.byoc_pipeline_constructs.iot_deploy_pipeline_construct import (
    DeployPipelineConstruct,
)


class MLOpsStack(Stack):
    DESCRIPTION: str = "This template includes a model building pipeline that includes a workflow to pre-process, train, evaluate and register a model. The deploy pipeline creates a dev, preprod and production greengrass/iot application for inference. The target PREPROD/PROD accounts are provided as cloudformation paramters and must be provided during project creation."
    TEMPLATE_NAME: str = "Dynamic Accounts MLOps template for iot deployment"

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Define required parmeters
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
        
        preprod_iotthinggroup_arn = aws_cdk.CfnParameter(
            self,
            "PreprodIotThinggroupArn",
            type="String",
            min_length=8,
            description="ARN of the Iot Thinggroup in preprod account.",
        ).value_as_string
        
        preprod_iotrole_arn = aws_cdk.CfnParameter(
            self,
            "PreprodIotRoleArn",
            type="String",
            min_length=8,
            description="ARN of the Iot Role in preprod account.",
        ).value_as_string
        
        prod_iotthinggroup_arn = aws_cdk.CfnParameter(
            self,
            "ProdIotThinggroupArn",
            type="String",
            min_length=8,
            description="ARN of the Iot Thinggroup in prod account.",
        ).value_as_string
        
        prod_iotrole_arn = aws_cdk.CfnParameter(
            self,
            "ProdIotRoleArn",
            type="String",
            min_length=8,
            description="ARN of the Iot Role in prod account.",
        ).value_as_string

        Tags.of(self).add("sagemaker:project-id", project_id)
        Tags.of(self).add("sagemaker:project-name", project_name)

        SSMConstruct(
            self,
            "MLOpsSSM",
            project_name=project_name,
            preprod_account=preprod_account,
            prod_account=prod_account,
            deployment_region=deployment_region,
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
            bucket_name=f"mlops-{project_name}-{Aws.ACCOUNT_ID}",
            encryption_key=kms_key,
            versioned=True,
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
        
        # create ECR repository
        ml_models_ecr_repo = ecr.Repository(
            self,
            "MLModelsECRRepository",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.MUTABLE,
            repository_name=f"{project_name}",
        )

        # add cross account resource policies
        ml_models_ecr_repo.add_to_resource_policy(
            iam.PolicyStatement(
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:BatchGetImage",
                    "ecr:CompleteLayerUpload",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:InitiateLayerUpload",
                    "ecr:PutImage",
                    "ecr:UploadLayerPart",
                ],
                principals=[
                    iam.ArnPrincipal(f"arn:aws:iam::{Aws.ACCOUNT_ID}:root"),
                ],
            )
        )

        ml_models_ecr_repo.add_to_resource_policy(
            iam.PolicyStatement(
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                principals=[
                    iam.ArnPrincipal(f"arn:aws:iam::{preprod_account}:root"),
                    iam.ArnPrincipal(f"arn:aws:iam::{prod_account}:root"),
                ],
            )
        )

        seed_bucket = CfnDynamicReference(CfnDynamicReferenceService.SSM, "/mlops/code/seed_bucket").to_string()
        build_app_key = CfnDynamicReference(CfnDynamicReferenceService.SSM, "/mlops/code/iot_build").to_string()
        deploy_app_key = CfnDynamicReference(CfnDynamicReferenceService.SSM, "/mlops/code/iot_deploy").to_string()

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
            bucket_name=f"pipeline-{project_name}-{Aws.ACCOUNT_ID}",
            encryption_key=kms_key,
            versioned=True,
            removal_policy=aws_cdk.RemovalPolicy.DESTROY,
        )

        BuildPipelineConstruct(
            self,
            "build",
            project_name,
            project_id,
            s3_artifact,
            pipeline_artifact_bucket,
            model_package_group_name,
            ml_models_ecr_repo.repository_name,
            seed_bucket,
            build_app_key,
        )

        DeployPipelineConstruct(
            self,
            "deploy",
            project_name,
            project_id,
            pipeline_artifact_bucket,
            model_package_group_name,
            ml_models_ecr_repo.repository_arn,
            s3_artifact,
            seed_bucket,
            deploy_app_key,
            preprod_account,
            prod_account,
            deployment_region,
            preprod_iotthinggroup_arn,
            preprod_iotrole_arn,
            prod_iotthinggroup_arn,
            prod_iotrole_arn,
            create_model_event_rule=True,
        )
