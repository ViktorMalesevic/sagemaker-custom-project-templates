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
from logging import Logger

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

from cdk_service_catalog.sm_service_catalog import SageMakerServiceCatalog
from mlops_commons.utilities.cdk_app_config import (
    DeploymentStage,
    PipelineConfig, CodeCommitConfig
)
from mlops_commons.utilities.log_helper import LogHelper
from mlops_commons.utilities.s3_utils import S3Utils


class SageMakerServiceCatalogStage(Stage):
    def __init__(
            self, scope: Construct, construct_id: str, app_prefix: str,
            **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        service = SageMakerServiceCatalog(
            self,
            "template",
            app_prefix=app_prefix,
            **kwargs,
        )


class CdkPipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, set_name: str,
                 app_prefix: str, deploy_conf: DeploymentStage,
                 pipeline_conf: PipelineConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.logger: Logger = LogHelper.get_logger(self)

        dev_account: str = str(deploy_conf.account)
        dev_region: str = deploy_conf.region

        code_commit_conf: CodeCommitConfig = pipeline_conf.code_commit.project_template
        repo: codecommit.IRepository = codecommit.Repository.from_repository_name(
            self, "ProjectTemplateRepo", repository_name=code_commit_conf.repo_name
        )

        artifact_bucket = self.create_pipeline_artifact_bucket(
            dev_account=dev_account,
            dev_region=dev_region,
            app_prefix=app_prefix,
            set_name=set_name
        )

        pipeline = pipelines.CodePipeline(
            self,
            "Pipeline",
            pipeline_name=f"{app_prefix}-service-catalog-"
                          f"{code_commit_conf.branch_name}-{set_name}",
            synth=pipelines.ShellStep(
                "Synth",
                input=pipelines.CodePipelineSource.code_commit(repo, code_commit_conf.branch_name),
                install_commands=[
                    "npm install -g aws-cdk@2.100.0",
                    "pip install -r requirements.txt",
                ],
                commands=[
                    f"cdk synth --no-lookup",
                ],
            ),
            docker_enabled_for_synth=True,
            docker_enabled_for_self_mutation=True,
            cross_account_keys=True,
            self_mutation=True,
            artifact_bucket=artifact_bucket,

        )

        # using as environment variable to pass the set name to concrete implementation of product, this
        # will be accessed in the deploy_pipeline.py file
        os.environ["infra_set_name"] = set_name

        pipeline.add_stage(
            SageMakerServiceCatalogStage(
                self,
                "mlops-sm-project",
                app_prefix=app_prefix,
                env={
                    "account": dev_account,
                    "region": dev_region,
                },
            )
        )

        # General tags applied to all resources created on this scope (self)
        Tags.of(self).add("cdk-app", f"{app_prefix}-sm-template")

    def create_pipeline_artifact_bucket(self, dev_account: str, dev_region: str, app_prefix: str, set_name: str) -> s3.Bucket:
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
                    iam.ArnPrincipal(f"arn:aws:iam::{dev_account}:root"),
                ],
            )
        )

        bucket_name: str = S3Utils.create_bucket_name(
            prefix=app_prefix,
            name_part1='sm-template-pipeline',
            name_part2=set_name,
            suffix_part1=dev_account,
            suffix_part2=dev_region,
            convert_region_to_short_code=True
        )
        self.logger.info(f'Creating sm-project-template pipeline bucket : {bucket_name}')
        s3_artifact = s3.Bucket(
            self,
            "MLOpsSmTemplatePipelineArtifactBucket",
            bucket_name=bucket_name,
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
                    iam.ArnPrincipal(f"arn:aws:iam::{dev_account}:root"),
                ],
            )
        )

        return s3_artifact