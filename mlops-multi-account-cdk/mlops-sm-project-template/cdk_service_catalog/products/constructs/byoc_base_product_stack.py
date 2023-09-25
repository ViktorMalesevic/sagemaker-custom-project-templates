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
    aws_s3 as s3,
    aws_iam as iam,
    aws_ecr as ecr,
)
from constructs import Construct
from typing import Optional
from cdk_service_catalog.products.constructs.base_product_stack import MLOpsBaseProductStack
from cdk_service_catalog.products.constructs.build_pipeline import BuildPipelineConstruct
from cdk_service_catalog.products.constructs.deploy_pipeline import DeployPipelineConstruct


class MLOpsBYOCBaseProductStack(MLOpsBaseProductStack):

    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            asset_bucket: s3.Bucket = None,
            **kwargs
    ) -> None:
        super().__init__(scope, construct_id, asset_bucket=asset_bucket, **kwargs)

        self.ml_models_ecr_repo: Optional[ecr.Repository] = None

    def setup_resource(self):

        # create ECR repository
        self.ml_models_ecr_repo = ecr.Repository(
            self,
            "MLModelsECRRepository",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.MUTABLE,
            repository_name=f"{self.project_name}",
        )

        # add cross account resource policies
        self.ml_models_ecr_repo.add_to_resource_policy(
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

        self.ml_models_ecr_repo.add_to_resource_policy(
            iam.PolicyStatement(
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                principals=[
                    iam.ArnPrincipal(f"arn:aws:iam::{self.preprod_account}:root"),
                    iam.ArnPrincipal(f"arn:aws:iam::{self.prod_account}:root"),
                ],
            )
        )

    def setup_pipeline(self):

        BuildPipelineConstruct(
            self,
            "build",
            project_name=self.project_name,
            project_id=self.project_id,
            pipeline_artifact_bucket=self.pipeline_artifact_bucket,
            model_package_group_name=self.model_package_group_name,
            repository=self.build_app_repository,
            s3_artifact=self.s3_artifact,
            ecr_repository_name=self.ml_models_ecr_repo.repository_name,
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
            model_bucket_arn=self.s3_artifact.bucket_arn,
            ecr_repo_arn=self.ml_models_ecr_repo.repository_arn,
            deployment_region=self.deployment_region,
            create_model_event_rule=True,
        )

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
                                  "template for real-time deployment using your own container"
                                  )

        self.support_email: str = 'byoc_base_product@example.com'

        self.support_url: str = 'https://example.com/support/byoc_base_product'

        self.support_description: str = 'Example of support details for byoc base product'

        # ###############################################################################################
