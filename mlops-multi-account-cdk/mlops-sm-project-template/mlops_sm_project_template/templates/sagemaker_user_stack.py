"""Sagemaker studio user service catalog product"""

from aws_cdk import (
    Fn,
    CfnTag,
    CfnParameter,
    aws_iam as iam,
    aws_ssm as ssm,
    aws_sagemaker as sagemaker,
    aws_servicecatalog as servicecatalog,
)

from constructs import Construct


class SagemakerUserStack(servicecatalog.ProductStack):
    """Class to create sagemaker studio user product"""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ds_user_id = CfnParameter(
            self,
            "DSUserID",
            type="String",
            description="User ID can be found in top right corner of AWS console,\
                 for example 'Data_Scientist_Role' ",
            min_length=1,
        ).value_as_string

        user_name = CfnParameter(
            self,
            "ProfileName",
            type="String",
            description="Name to assign to the SageMaker Studio Profile",
            min_length=1,
        ).value_as_string

        domain_id = ssm.StringParameter.value_for_string_parameter(
            self, "/mlops/dev/domain_id"
        )
        execution_role = ssm.StringParameter.value_for_string_parameter(
            self, "/mlops/dev/domain/role/execution"
        )

        # create sagemaker studio user
        self.studio_user = self.sagemaker_studio_user(
            domain_id, user_name, execution_role, ds_user_id
        )

        ds_role = iam.Role.from_role_arn(self, "DsRole", execution_role)
        ds_role.grant_assume_role(
            iam.ArnPrincipal(
                Fn.sub("arn:aws:sts::${AWS::AccountId}:assumed-role/${DSUserID}")
            )
        )

    def sagemaker_studio_user(self, domain_id, user_name, execution_role, ds_user_id):
        user_profile = sagemaker.CfnUserProfile(
            self,
            "UserProfile",
            domain_id=domain_id,
            user_profile_name=user_name,
            user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                execution_role=execution_role,
                # security_groups=["securityGroups"],
            ),
            tags=[CfnTag(key="studiouserid", value=ds_user_id)],
        )

        return user_profile
