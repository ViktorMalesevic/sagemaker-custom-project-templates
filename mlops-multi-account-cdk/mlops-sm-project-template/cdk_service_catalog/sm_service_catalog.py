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


from typing import List, Any
import logging
from logging import Logger
import aws_cdk
import aws_cdk as cdk
from aws_cdk import CfnParameter
from aws_cdk import Stack, Tags
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_servicecatalog as servicecatalog
from constructs import Construct

from cdk_service_catalog.products.constructs.base_product_stack import MLOpsBaseProductStack
from cdk_utilities.class_utilities import ClassUtilities


class SageMakerServiceCatalog(Stack):
    logging.basicConfig(level=logging.INFO)

    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            app_prefix: str,
            **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.logger: Logger = logging.getLogger(self.__class__.__name__)

        execution_role_arn = CfnParameter(
            self,
            "ExecutionRoleArn",
            type="AWS::SSM::Parameter::Value<String>",
            description="The SageMaker Studio execution role",
            min_length=1,
            default="/mlops/role/lead",
        ).value_as_string

        default_file_assets_bucket_name: str = (f'cdk-{aws_cdk.DefaultStackSynthesizer.DEFAULT_QUALIFIER}'
                                                f'-assets-{aws_cdk.Aws.ACCOUNT_ID}-{aws_cdk.Aws.REGION}')

        sc_product_artifact_bucket = s3.Bucket.from_bucket_name(
            scope=self,
            id='DEFAULT_FILE_ASSETS_BUCKET',
            bucket_name=default_file_assets_bucket_name,
        )

        # Service Catalog Portfolio
        portfolio = servicecatalog.Portfolio(
            self,
            "SM_Projects_Portfolio",
            display_name="SM Projects Portfolio",
            provider_name="ML Admin Team",
            description="Products for SM Projects",
        )

        execute_role = iam.Role.from_role_arn(self,
                                              'PortfolioExecutionRoleArn',
                                              execution_role_arn,
                                              mutable=False
                                              )
        portfolio.give_access_to_role(execute_role)
        launch_role: iam.Role = self.create_launch_role()

        # Adding sagemaker projects products
        self.add_all_products(
            portfolio=portfolio,
            launch_role=launch_role,
            sc_product_artifact_bucket=sc_product_artifact_bucket,
            app_prefix=app_prefix,
        )

    def add_all_products(
            self,
            portfolio: servicecatalog.Portfolio,
            launch_role: iam.Role,
            base_package: str = 'cdk_service_catalog.products',
            exclude_packages: List[str] = ('constructs', 'seed_code'),
            **kwargs
    ):
        product_classes: List[Any] = ClassUtilities.find_subclasses(
            base_class=MLOpsBaseProductStack,
            base_package=base_package,
            exclude_packages=exclude_packages
        )

        for product_class in product_classes:
            self.logger.info(f'product_class : {product_class.__name__}')
            SageMakerServiceCatalogProduct(
                self,
                product_class.__name__,
                portfolio=portfolio,
                product_class=product_class,
                launch_role=launch_role,
                **kwargs,
            )

    def create_launch_role(self) -> iam.Role:
        # Create the launch role
        products_launch_role = iam.Role(
            self,
            "ProductLaunchRole",
            assumed_by=iam.ServicePrincipal("servicecatalog.amazonaws.com"),
            path="/service-role/",
        )

        products_launch_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSageMakerAdmin-ServiceCatalogProductsServiceRolePolicy"
            )
        )

        products_launch_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEventBridgeFullAccess")
        )

        products_launch_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AWSKeyManagementServicePowerUser")
        )

        products_launch_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("IAMFullAccess"))

        products_launch_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AWSCodeCommitFullAccess")
        )

        products_launch_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AWSCodePipeline_FullAccess")
        )

        products_launch_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AWSCodeBuildAdminAccess")
        )

        products_launch_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambda_FullAccess"))
        products_launch_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMReadOnlyAccess")
        )
        products_launch_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryFullAccess")
        )

        products_launch_role.add_to_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                effect=iam.Effect.ALLOW,
                resources=[
                    "*"
                    # TODO lock this policy to only certain roles from the other account that are used for deploying the solution as defined in templates/pipeline_constructs/deploy_pipeline_stack.py
                ],
            ),
        )

        products_launch_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:*",
                    "s3-object-lambda:*",
                ],
                effect=iam.Effect.ALLOW,
                resources=["*"],
            ),
        )

        products_launch_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "kms:Create*",
                    "kms:Describe*",
                    "kms:Enable*",
                    "kms:List*",
                    "kms:Put*",
                    "kms:Update*",
                    "kms:Revoke*",
                    "kms:Disable*",
                    "kms:Get*",
                    "kms:Delete*",
                    "kms:ScheduleKeyDeletion",
                    "kms:CancelKeyDeletion",
                ],
                effect=iam.Effect.ALLOW,
                resources=["*"],
            ),
        )

        products_launch_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ssm:*",
                ],
                effect=iam.Effect.ALLOW,
                resources=[
                    f"arn:aws:ssm:*:{cdk.Aws.ACCOUNT_ID}:parameter/mlops/*",
                ],
            ),
        )

        products_launch_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:*",
                ],
                effect=iam.Effect.ALLOW,
                resources=[f"arn:aws:sagemaker:*:{cdk.Aws.ACCOUNT_ID}:model-package-group/*"],
            ),
        )

        return products_launch_role


class SageMakerServiceCatalogProduct(cdk.NestedStack):
    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            portfolio: servicecatalog.Portfolio,
            product_class: Any,
            launch_role: iam.Role,
            sc_product_artifact_bucket: s3.Bucket,
            app_prefix: str,
            **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        product: MLOpsBaseProductStack = product_class(self, "project", asset_bucket=sc_product_artifact_bucket,
                                                       app_prefix=app_prefix)

        sm_projects_product = servicecatalog.CloudFormationProduct(
            self,
            product.__class__.__name__,
            product_name=product.product_name,
            owner="Global ML Team",
            product_versions=[
                servicecatalog.CloudFormationProductVersion(
                    cloud_formation_template=servicecatalog.CloudFormationTemplate.from_product_stack(
                        product
                    ),
                    product_version_name="v1",
                    validate_template=True,
                )
            ],
            description=product.description,
            support_email=product.support_email,
            support_description=product.support_description,
            support_url=product.support_url,
        )
        portfolio.add_product(sm_projects_product)
        portfolio.set_launch_role(sm_projects_product, launch_role)
        Tags.of(sm_projects_product).add(
            key="sagemaker:studio-visibility", value="true"
        )
