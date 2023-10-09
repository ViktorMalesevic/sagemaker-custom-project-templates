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
from typing import List, Any

import aws_cdk as core
from aws_cdk import (
    CfnParameter,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_sagemaker as sagemaker,
    aws_ec2 as ec2,
)
from aws_cdk.aws_lambda_python_alpha import PythonFunction
from aws_cdk.custom_resources import Provider
from constructs import Construct

from cdk_sm_infra.constructs.sm_roles import SMRoles
from cdk_utilities.cdk_infra_app_config import SagemakerConfig, SagemakerUserProfileConfig


class SMStudio(Construct):
    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            app_prefix: str,
            sagemaker_conf: SagemakerConfig,
            vpc: ec2.IVpc = None,
            subnets: List[ec2.Subnet] = []
    ) -> None:
        super().__init__(scope, construct_id)

        self.base_dir: str = os.path.abspath(f'{os.path.dirname(__file__)}')

        domain_name = CfnParameter(
            self,
            "StudioDomainName",
            type="String",
            description="Name to assign to the SageMaker Studio domain",
            default="studio-domain",
        ).value_as_string

        s3_bucket_prefix = CfnParameter(
            self,
            "S3BucketName",
            type="String",
            description="S3 bucket where data are stored",
            default=app_prefix,
        ).value_as_string

        # create roles to be used for sagemaker user profiles and attached to sagemaker studio domain
        sm_roles = SMRoles(self, "sm-roles", s3_bucket_prefix)

        # setup security group to be used for sagemaker studio domain
        sagemaker_sg = ec2.SecurityGroup(
            self,
            "SecurityGroup",
            vpc=vpc,
            description="Security Group for SageMaker Studio Notebook, Training Job and Hosting Endpoint",
        )

        sagemaker_sg.add_ingress_rule(sagemaker_sg, ec2.Port.all_traffic())

        # create sagemaker studio domain
        self.studio_domain = self.sagemaker_studio_domain(
            domain_name,
            sm_roles.sagemaker_studio_role,
            vpc_id=vpc.vpc_id,
            security_group_ids=[sagemaker_sg.security_group_id],
            subnet_ids=[subnet.subnet_id for subnet in subnets],
        )

        self.enable_sagemaker_projects(
            [
                sm_roles.sagemaker_studio_role.role_arn,
                sm_roles.data_scientist_role.role_arn,
                sm_roles.lead_data_scientist_role.role_arn,
            ]
        )

        # Configure sagemaker studio profiles

        # data scientist profiles
        self.sagemaker_studio_profiles(
            self.studio_domain.attr_domain_id,
            sm_roles.data_scientist_role.role_arn,
            sagemaker_conf.profiles.data_scientists
        )

        # lead data scientist profiles
        self.sagemaker_studio_profiles(
            self.studio_domain.attr_domain_id,
            sm_roles.lead_data_scientist_role.role_arn,
            sagemaker_conf.profiles.lead_data_scientists
        )

    """
        Create the Custom Resource to enable sagemaker projects for the different personas

        :param roles: - roles to be attached to service catalog portfolio
    """

    def enable_sagemaker_projects(self, roles):
        event_handler = PythonFunction(
            self,
            "sg-project-function",
            runtime=lambda_.Runtime.PYTHON_3_11,
            entry="cdk_sm_infra/functions/sm_studio/enable_sm_projects",
            timeout=core.Duration.seconds(120),
        )

        event_handler.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:EnableSagemakerServicecatalogPortfolio",
                    "servicecatalog:ListAcceptedPortfolioShares",
                    "servicecatalog:AssociatePrincipalWithPortfolio",
                    "servicecatalog:AcceptPortfolioShare",
                    "iam:GetRole",
                ],
                resources=["*"],
            ),
        )

        provider = Provider(self, "sg-project-lead-provider", on_event_handler=event_handler)

        core.CustomResource(
            self,
            "cs-sg-project",
            service_token=provider.service_token,
            removal_policy=core.RemovalPolicy.DESTROY,
            resource_type="Custom::EnableSageMakerProjects",
            properties={
                "iteration": 1,
                "ExecutionRoles": roles,
            },
        )

    """
        Create the SageMaker Studio Domain

        :param domain_name: - name to assign to the SageMaker Studio Domain
        :param s3_bucket: - S3 bucket used for sharing notebooks between users
        :param sagemaker_studio_role: - IAM Execution Role for the domain
        :param security_group_ids: - list of comma separated security group ids
        :param subnet_ids: - list of comma separated subnet ids
        :param vpc_id: - VPC Id for the domain
    """

    def sagemaker_studio_domain(
            self,
            domain_name,
            sagemaker_studio_role,
            security_group_ids,
            subnet_ids,
            vpc_id,
    ):
        domain = sagemaker.CfnDomain(
            self,
            "sagemaker-domain",
            auth_mode="IAM",
            app_network_access_type="VpcOnly",
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=sagemaker_studio_role.role_arn,
                security_groups=security_group_ids,
                sharing_settings=sagemaker.CfnDomain.SharingSettingsProperty(),  # disable notebook output sharing
            ),
            domain_name=domain_name,
            subnet_ids=subnet_ids,
            vpc_id=vpc_id,
        )

        return domain

    """
        Create SageMaker User Profiles

        :param studio_domain_id: - SageMaker Studio Domain id from object created. See method sagemaker_studio_domain
        :param role_arn: - IAM Execution Role Arn for the user profiles
        :param file_name: - Name of yaml file to load
    """

    def sagemaker_studio_profiles(self, studio_domain_id, role_arn, profile_conf: SagemakerUserProfileConfig):

        sm_user_profiles: List[Any] = list()

        for user in profile_conf.users:
            sm_user_profiles.append(
                sagemaker.CfnUserProfile(
                    self,
                    f"{profile_conf.prefix}-{user.user_profile_name}",
                    domain_id=studio_domain_id,
                    user_profile_name=user.user_profile_name,
                    user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(execution_role=role_arn),
                )
            )

        return sm_user_profiles
