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
from typing import Optional
from pathlib import Path
import aws_cdk
from aws_cdk import (
    Aws,
    Tags,
    aws_s3 as s3,
    aws_iam as iam,
    aws_kms as kms,
    aws_sagemaker as sagemaker,
    aws_codecommit as codecommit,
)
from constructs import Construct

from cdk_service_catalog.products.constructs.base_product_metadata import BaseProductMetadata
from cdk_service_catalog.products.constructs.build_pipeline import BuildPipelineConstruct
from cdk_service_catalog.products.constructs.deploy_pipeline import DeployPipelineConstruct
from cdk_service_catalog.products.constructs.ssm import SSMConstruct
from cdk_utilities.zip_utils import ZipUtility


class MLOpsBaseProductStack(BaseProductMetadata):

    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            asset_bucket: s3.Bucket = None,
            **kwargs
    ) -> None:
        app_prefix: str = kwargs["app_prefix"]
        kwargs.pop('app_prefix')
        super().__init__(scope, construct_id, asset_bucket=asset_bucket, **kwargs)

        self.base_dir: str = os.path.abspath(f'{os.path.dirname(__file__)}{os.path.sep}..')

        self.subclass_path: str = self.get_subclass_realpath(base_dir=self.base_dir)

        self._default_build_app_seed_code_relative_path: str = './seed_code/build_app'
        self._default_deploy_app_seed_code_relative_path: str = './seed_code/deploy_app'
        self.project_name: str = ''
        self.project_id: str = ''
        self.preprod_account: str = ''
        self.prod_account: str = ''
        self.deployment_region: str = ''
        self.build_app_repository: Optional[codecommit.Repository] = None
        self.deploy_app_repository: Optional[codecommit.Repository] = None
        self.pipeline_artifact_bucket: Optional[s3.Bucket] = None
        self.BOOL_TRUE: bool = True
        self.BOOL_FALSE: bool = False
        # Define required parameters
        self.define_input_parameters()

        Tags.of(self).add("sagemaker:project-id", self.project_id)
        Tags.of(self).add("sagemaker:project-name", self.project_name)

        SSMConstruct(
            self,
            "MLOpsSSM",
            project_name=self.project_name,
            preprod_account=self.preprod_account,
            prod_account=self.prod_account,
            deployment_region=self.deployment_region,  # Modify when x-region is enabled
        )

        self.setup_seed_code_repositories(app_prefix, construct_id)

        Tags.of(self.deploy_app_repository).add(key="sagemaker:project-id", value=self.project_id)
        Tags.of(self.deploy_app_repository).add(
            key="sagemaker:project-name", value=self.project_name
        )

        self.s3_artifact: s3.Bucket = self.setup_seed_code_pipeline_artifact_bucket()

        self.model_package_group_name = f"{self.project_name}-{self.project_id}"

        # cross account model registry resource policy
        self.setup_sagemaker_model_package_group_policies()

        self.setup_pipeline_artifact_bucket()
        self.setup_resource()

        self.setup_pipeline()

    def get_seed_code_base_path(self) -> str:
        return self.get_subclass_realpath(base_dir=self.base_dir)

    def setup_sagemaker_model_package_group_policies(self):
        model_package_group_policy = iam.PolicyDocument(
            statements=[
                iam.PolicyStatement(
                    sid="ModelPackageGroup",
                    actions=[
                        "sagemaker:DescribeModelPackageGroup",
                    ],
                    resources=[
                        f"arn:aws:sagemaker:{Aws.REGION}:{Aws.ACCOUNT_ID}:"
                        f"model-package-group/{self.model_package_group_name}"
                    ],
                    principals=[
                        iam.ArnPrincipal(f"arn:aws:iam::{self.preprod_account}:root"),
                        iam.ArnPrincipal(f"arn:aws:iam::{self.prod_account}:root"),
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
                        f"arn:aws:sagemaker:{Aws.REGION}:{Aws.ACCOUNT_ID}:"
                        f"model-package/{self.model_package_group_name}/*"
                    ],
                    principals=[
                        iam.ArnPrincipal(f"arn:aws:iam::{self.preprod_account}:root"),
                        iam.ArnPrincipal(f"arn:aws:iam::{self.prod_account}:root"),
                    ],
                ),
            ]
        ).to_json()
        model_package_group = sagemaker.CfnModelPackageGroup(
            self,
            "ModelPackageGroup",
            model_package_group_name=self.model_package_group_name,
            model_package_group_description=f"Model Package Group for {self.project_name}",
            model_package_group_policy=model_package_group_policy,
            tags=[
                aws_cdk.CfnTag(key="sagemaker:project-id", value=self.project_id),
                aws_cdk.CfnTag(key="sagemaker:project-name", value=self.project_name),
            ],
        )

    def setup_pipeline_artifact_bucket(self):
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
        self.pipeline_artifact_bucket = s3.Bucket(
            self,
            "PipelineBucket",
            bucket_name=f"pipeline-{self.project_name}-{Aws.ACCOUNT_ID}",  # Bucket name has a limit of 63 characters
            encryption_key=kms_key,
            versioned=True,
            auto_delete_objects=True,
            removal_policy=aws_cdk.RemovalPolicy.DESTROY,
        )

    def setup_seed_code_pipeline_artifact_bucket(self) -> s3.Bucket:
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
                    iam.ArnPrincipal(f"arn:aws:iam::{self.preprod_account}:root"),
                    iam.ArnPrincipal(f"arn:aws:iam::{self.prod_account}:root"),
                ],
            )
        )
        s3_artifact: s3.Bucket = s3.Bucket(
            self,
            "S3Artifact",
            bucket_name=f"mlops-{self.project_name}-{Aws.ACCOUNT_ID}",  # Bucket name has a limit of 63 characters
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
                    iam.ArnPrincipal(f"arn:aws:iam::{self.preprod_account}:root"),
                    iam.ArnPrincipal(f"arn:aws:iam::{self.prod_account}:root"),
                ],
            )
        )

        return s3_artifact

    def setup_seed_code_repositories(self, app_prefix, construct_id):
        zip_out_path: Path = Path(f'.zip_archives{os.path.sep}{app_prefix}')
        self.build_app_repository = codecommit.Repository(
            self,
            "BuildRepo",
            repository_name=f"{self.project_name}-{construct_id}-build",
            code=codecommit.Code.from_zip_file(
                ZipUtility.create_zip(
                    local_path=f"{self.get_seed_code_base_path()}{os.path.sep}"
                               f"{self.get_build_app_seed_code_relative_path()}",
                    out_path=zip_out_path
                ),
                branch="main",
            ),
        )
        self.deploy_app_repository = codecommit.Repository(
            self,
            "DeployRepo",
            repository_name=f"{self.project_name}-{construct_id}-deploy",
            code=codecommit.Code.from_zip_file(
                ZipUtility.create_zip(
                    out_path=zip_out_path,
                    local_path=f"{self.get_seed_code_base_path()}{os.path.sep}"
                               f"{self.get_deploy_app_seed_code_relative_path()}"
                ),
                branch="main",
            ),
        )

    def define_input_parameters(self):
        self.project_name = aws_cdk.CfnParameter(
            self,
            "SageMakerProjectName",
            type="String",
            description="The name of the SageMaker project.",
            min_length=1,
            max_length=32,
        ).value_as_string
        self.project_id = aws_cdk.CfnParameter(
            self,
            "SageMakerProjectId",
            type="String",
            min_length=1,
            max_length=16,
            description="Service generated Id of the project.",
        ).value_as_string
        self.preprod_account = aws_cdk.CfnParameter(
            self,
            "PreProdAccount",
            type="String",
            min_length=11,
            max_length=13,
            description="Id of preprod account.",
        ).value_as_string
        self.prod_account = aws_cdk.CfnParameter(
            self,
            "ProdAccount",
            type="String",
            min_length=11,
            max_length=13,
            description="Id of prod account.",
        ).value_as_string
        self.deployment_region = aws_cdk.CfnParameter(
            self,
            "DeploymentRegion",
            type="String",
            min_length=8,
            max_length=10,
            description="Deployment region for preprod and prod account.",
        ).value_as_string

    def setup_resource(self):
        # nothing to do here
        pass
        # print(f'{self.project_name}')

    def setup_pipeline(self):
        BuildPipelineConstruct(
            self,
            "build",
            project_name=self.project_name,
            project_id=self.project_id,
            pipeline_artifact_bucket=self.pipeline_artifact_bucket,
            model_package_group_name=self.model_package_group_name,
            repository=self.build_app_repository,
            s3_artifact=self.s3_artifact
        )

        DeployPipelineConstruct(
            self,
            "deploy",
            project_name=self.project_name,
            project_id=self.project_id,
            pipeline_artifact_bucket=self.pipeline_artifact_bucket,
            model_package_group_name=self.model_package_group_name,
            repository=self.deploy_app_repository,
            s3_artifact=self.s3_artifact,
            preprod_account=self.preprod_account,
            prod_account=self.prod_account,
            deployment_region=self.deployment_region,
            create_model_event_rule=self.get_create_model_event_rule(),
        )

    def get_create_model_event_rule(self) -> bool:
        return self.BOOL_TRUE

    def get_build_app_seed_code_relative_path(self) -> str:
        return self._default_build_app_seed_code_relative_path

    def get_deploy_app_seed_code_relative_path(self) -> str:
        return self._default_deploy_app_seed_code_relative_path

    def set_product_metadata(self):
        # ################ Product Metadata #############################################################

        self.description: str = (
            "This template includes a model building pipeline that includes a workflow to pre-process, "
            "train, evaluate and register a model. The deploy pipeline creates a dev,preprod and "
            "production endpoint. The target DEV/PREPROD/PROD accounts are parameterized in this "
            "template."
        )
        self.product_name: str = ("Build & Deploy MLOps parameterized "
                                  "template for real-time deployment"
                                  )

        self.support_email: str = 'base_product@example.com'

        self.support_url: str = 'https://example.com/support/base_product'

        self.support_description: str = 'Example of support details for base product'

        # ###############################################################################################
