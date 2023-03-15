"""Sagemaker studio domain service catalog product"""
import base64
from typing import List

from aws_cdk import (
    Fn,
    CfnTag,
    Duration,
    CfnParameter,
    CustomResource,
    RemovalPolicy,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_ssm as ssm,
    aws_lambda as lambda_,
    aws_sagemaker as sagemaker,
    aws_servicecatalog as servicecatalog,
)
from constructs import Construct

from mlops_sm_project_template.templates.helper_scripts.sm_roles import (
    SMRoles,
)
from mlops_sm_project_template.templates.helper_scripts.sagemaker_images import (
    get_sagemaker_image_arn,
)

JUPYTER_SERVER_APP_IMAGE_NAME = "jupyter-server-3"
KERNEL_GATEWAY_APP_IMAGE_NAME = "datascience-2.0"


class SagemakerStudioStack(servicecatalog.ProductStack):
    """Class to create sagemaker studio domain product"""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc: ec2.IVpc = None,
        subnets: List[ec2.Subnet] = [],
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        vpc_id = CfnParameter(
            self,
            "sg-id",
            type="AWS::SSM::Parameter::Value<String>",
            description="Account VPC id",
            min_length=1,
            default="/vpc/id",
        ).value_as_string

        subnets = CfnParameter(
            self,
            "subnet-ids",
            type="AWS::SSM::Parameter::Value<List<String>>",
            description="Account APP Subnets IDs",
            min_length=1,
            default="/vpc/subnets/private/ids",
        ).value_as_list

        sagemaker_sg_id = CfnParameter(
            self,
            "sg-id",
            type="AWS::SSM::Parameter::Value<String>",
            description="Account Default Security Group id",
            min_length=1,
            default="/vpc/sg/id",
        ).value_as_string

        lead_ds_user_id = CfnParameter(
            self,
            "LeadDSUserID",
            type="String",
            description="Lead Data Scientist user ID can be found in top right corner of AWS console,\
                 for example '<last-name>_<first-name>_<GID>' ",
            min_length=1,
        ).value_as_string

        domain_name = CfnParameter(
            self,
            "SMStudioDomainName",
            type="String",
            description="Choose a unique name for the SageMaker Studio Domain",
            min_length=1,
        ).value_as_string

        # VPC
        vpc = ec2.Vpc.from_vpc_attributes(
            self, "VPC", vpc_id=vpc_id, availability_zones=Fn.get_azs()
        )

        # Create roles to be used for sagemaker user profiles
        sm_roles = SMRoles(self, "sm-roles", domain_name)

        # Create sagemaker studio domain
        self.studio_domain = self.sagemaker_studio_domain(
            domain_name,
            sm_roles.sagemaker_studio_role,
            vpc_id=vpc.vpc_id,
            security_group_ids=[sagemaker_sg_id],
            subnet_ids=subnets,
            default_instance_type="ml.t3.medium",
            aws_region="eu-central-1",
        )

        # Enable SageMaker projects in the domain
        self.enable_sagemaker_projects(
            [
                sm_roles.sagemaker_studio_role.role_arn,
                sm_roles.data_scientist_role.role_arn,
                sm_roles.lead_data_scientist_role.role_arn,
            ]
        )

        # Create lifecycle config for autoshutdown of idle instances
        with open(
            "cdk_service_catalog/product_model/ml_ops/lifecycle_scripts/combined-pip-autoshutdown.sh",
            "r",
            encoding="utf-8",
        ) as file:
            combined_script_data = file.read()
        combined_script_base64_bytes = base64.b64encode(
            combined_script_data.encode("ascii")
        )
        combined_script_base64_string = combined_script_base64_bytes.decode("ascii")

        self.create_lifecycle_config(
            "combinedPipAutoshutdownConfig",
            combined_script_base64_string,
            "JupyterServer",
            self.studio_domain.attr_domain_id,
        )

        # Create lead DS user profile
        self.sagemaker_studio_profiles(
            self.studio_domain.attr_domain_id,
            "lead-data-scientist",
            sm_roles.lead_data_scientist_role.role_arn,
            lead_ds_user_id,
        )

        # Export SSM params
        studio_domain_id_param = ssm.StringParameter(
            self,
            "StudioDomainID",
            parameter_name="/mlops/dev/domain_id",
            string_value=self.studio_domain.attr_domain_id,
        )

    def enable_sagemaker_projects(self, roles):
        """enable_sagemaker_projects.

        Args:
            roles (_type_): _description_
        """
        with open(
            "mlops-multi-account-cdk/mlops-sm-project-template/mlops_sm_project_template/templates/functions/enable_sm_projects/index.py",
            "r",
            encoding="utf-8",
        ) as file:
            data = file.read()

        event_handler = lambda_.Function(
            self,
            "sg-project-function",
            code=lambda_.InlineCode(data),
            handler="index.handler",
            runtime=lambda_.Runtime.PYTHON_3_8,
            timeout=Duration.seconds(120),
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

        CustomResource(
            self,
            "cs-sg-project",
            service_token=event_handler.function_arn,
            removal_policy=RemovalPolicy.DESTROY,
            resource_type="Custom::EnableSageMakerProjects",
            properties={
                "iteration": 1,
                "ExecutionRoles": roles,
            },
        )

    def sagemaker_studio_domain(
        self,
        domain_name,
        sagemaker_studio_role,
        security_group_ids,
        subnet_ids,
        vpc_id,
        default_instance_type="ml.t3.medium",
        aws_region="eu-central-1",
    ):
        """sagemaker_studio_domain.

        Args:
            domain_name (_type_): _description_
            sagemaker_studio_role (_type_): _description_
            security_group_ids (_type_): _description_
            subnet_ids (_type_): _description_
            vpc_id (_type_): _description_
            default_instance_type (str, optional): _description_. Defaults to "ml.t3.medium".
            aws_region (str, optional): _description_. Defaults to "eu-central-1".

        Returns:
            _type_: _description_
        """
        domain = sagemaker.CfnDomain(
            self,
            "sagemaker-domain",
            auth_mode="IAM",
            app_network_access_type="VpcOnly",
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=sagemaker_studio_role.role_arn,
                jupyter_server_app_settings=sagemaker.CfnDomain.JupyterServerAppSettingsProperty(
                    default_resource_spec=sagemaker.CfnDomain.ResourceSpecProperty(
                        instance_type="system",
                        sage_maker_image_arn=get_sagemaker_image_arn(
                            JUPYTER_SERVER_APP_IMAGE_NAME, aws_region
                        ),
                    )
                ),
                kernel_gateway_app_settings=sagemaker.CfnDomain.KernelGatewayAppSettingsProperty(
                    default_resource_spec=sagemaker.CfnDomain.ResourceSpecProperty(
                        instance_type=default_instance_type,
                        sage_maker_image_arn=get_sagemaker_image_arn(
                            KERNEL_GATEWAY_APP_IMAGE_NAME, aws_region
                        ),
                    ),
                ),
                security_groups=security_group_ids,
                sharing_settings=sagemaker.CfnDomain.SharingSettingsProperty(),
            ),
            domain_name=domain_name,
            subnet_ids=subnet_ids,
            vpc_id=vpc_id,
        )

        return domain

    def sagemaker_studio_profiles(
        self,
        studio_domain_id,
        user_name,
        role_arn,
        lead_ds_user_id,
    ):
        """sagemaker_studio_profiles.

        Args:
            studio_domain_id (_type_): _description_
            user_name (_type_): _description_
            role_arn (_type_): _description_
            lead_ds_user_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        user_profile = sagemaker.CfnUserProfile(
            self,
            "UserProfile",
            domain_id=studio_domain_id,
            user_profile_name=user_name,
            user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                execution_role=role_arn,
            ),
            tags=[CfnTag(key="studiouserid", value=lead_ds_user_id)],
        )

        return user_profile

    def create_lifecycle_config(self, rule_name, script, rule_type, domain_id=None):
        """create_lifecycle_config.

        Args:
            rule_name (_type_): _description_
            script (_type_): _description_
            rule_type (_type_): _description_
            domain_id (_type_, optional): _description_. Defaults to None.
        """
        with open(
            "mlops-multi-account-cdk/mlops-sm-project-template/mlops_sm_project_template/templates/functions/lifecycle_rule/index.py",
            "r",
            encoding="utf-8",
        ) as file:
            data = file.read()

        event_handler = lambda_.Function(
            self,
            "lifecycle-rule-function" + rule_name,
            code=lambda_.InlineCode(data),
            handler="index.handler",
            runtime=lambda_.Runtime.PYTHON_3_8,
            timeout=Duration.seconds(120),
        )

        event_handler.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker:*Lifecycle*", "sagemaker:UpdateDomain"],
                resources=["*"],
            ),
        )

        properties = {
            "iteration": 1,
            "rule_name": rule_name,
            "script": script,
            "rule_type": rule_type,
        }
        if domain_id:
            properties["domain_id"] = domain_id

        CustomResource(
            self,
            "cr-lifecycle-rule" + rule_name,
            service_token=event_handler.function_arn,
            removal_policy=RemovalPolicy.DESTROY,
            resource_type="Custom::LifecycleRule" + rule_name,
            properties=properties,
        )
