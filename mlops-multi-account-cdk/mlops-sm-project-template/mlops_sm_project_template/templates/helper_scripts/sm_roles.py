from weakref import ref
from aws_cdk import aws_iam as iam, aws_ssm as ssm, Fn

from constructs import Construct


class SMRoles(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        domain_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # create policies required for the roles

        sm_deny_policy = iam.Policy(
            self,
            "sm-deny-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.DENY,
                    actions=[
                        "sagemaker:CreateProject",
                    ],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.DENY,
                    actions=["sagemaker:UpdateModelPackage"],
                    resources=["*"],
                ),
            ],
        )

        services_policy = iam.Policy(
            self,
            "services-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "lambda:Create*",
                        "lambda:Update*",
                        "lambda:Invoke*",
                    ],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sagemaker:ListTags",
                    ],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "ecr:BatchGetImage",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:GetRepositoryPolicy",
                        "ecr:DescribeRepositories",
                        "ecr:DescribeImages",
                        "ecr:ListImages",
                        "ecr:GetAuthorizationToken",
                        "ecr:GetLifecyclePolicy",
                        "ecr:GetLifecyclePolicyPreview",
                        "ecr:ListTagsForResource",
                        "ecr:DescribeImageScanFindings",
                        "ecr:CreateRepository",
                        "ecr:CompleteLayerUpload",
                        "ecr:UploadLayerPart",
                        "ecr:InitiateLayerUpload",
                        "ecr:PutImage",
                    ],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "license-manager:ExtendLicenseConsumption",
                        "license-manager:ListReceivedLicenses",
                        "license-manager:GetLicense",
                        "license-manager:CheckoutLicense",
                        "license-manager:CheckInLicense",
                    ],
                    resources=["*"],
                ),
            ],
        )

        kms_policy = iam.Policy(
            self,
            "kms-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "kms:CreateGrant",
                        "kms:Decrypt",
                        "kms:DescribeKey",
                        "kms:Encrypt",
                        "kms:ReEncrypt",
                        "kms:GenerateDataKey",
                    ],
                    resources=["*"],
                )
            ],
        )

        s3_policy = iam.Policy(
            self,
            "s3-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "s3:AbortMultipartUpload",
                        "s3:DeleteObject",
                        "s3:Describe*",
                        "s3:GetObject",
                        "s3:PutBucket*",
                        "s3:PutObject",
                        "s3:PutObjectAcl",
                        "s3:GetBucketAcl",
                        "s3:GetBucketLocation",
                    ],
                    resources=["arn:aws:s3:::*"],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["s3:ListBucket"],
                    resources=["arn:aws:s3:::*"],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.DENY,
                    actions=["s3:DeleteBucket*"],
                    resources=["*"],
                ),
            ],
        )

        secrets_manager_policy = iam.Policy(
            self,
            "secrets-manager-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "secretsmanager:GetSecretValue",
                        "secretsmanager:DescribeSecret",
                    ],
                    resources=["*"],
                ),
            ],
        )

        ## create role for each persona

        # role for Data Scientist persona
        self.data_scientist_role = iam.Role(
            self,
            "data-scientist-role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AWSLambda_ReadOnlyAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonEC2ContainerRegistryReadOnly"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
            ],
        )

        sm_deny_policy.attach_to_role(self.data_scientist_role)
        services_policy.attach_to_role(self.data_scientist_role)
        kms_policy.attach_to_role(self.data_scientist_role)
        s3_policy.attach_to_role(self.data_scientist_role)
        secrets_manager_policy.attach_to_group(self.data_scientist_role)

        ssm.StringParameter(
            self,
            "ssm-sg-ds-role",
            parameter_name="/mlops/dev/domain/role/ds",
            string_value=self.data_scientist_role.role_arn,
            simple_name=False,
        )

        # role for Lead Data Scientist persona
        self.lead_data_scientist_role = iam.Role(
            self,
            "lead-data-scientist-role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMReadOnlyAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AWSLambda_ReadOnlyAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonEC2ContainerRegistryReadOnly"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
            ],
        )

        services_policy.attach_to_role(self.lead_data_scientist_role)
        kms_policy.attach_to_role(self.lead_data_scientist_role)
        s3_policy.attach_to_role(self.lead_data_scientist_role)
        secrets_manager_policy.attach_to_group(self.lead_data_scientist_role)

        ssm.StringParameter(
            self,
            "ssm-sg-lead-role",
            parameter_name="/mlops/dev/domain/role/lead",
            string_value=self.lead_data_scientist_role.role_arn,
            simple_name=False,
        )

        # SageMaker Studio execution role
        self.sagemaker_studio_role = iam.Role(
            self,
            "sagemaker-studio-role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AWSLambda_ReadOnlyAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonEC2ContainerRegistryReadOnly"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
            ],
        )

        services_policy.attach_to_role(self.sagemaker_studio_role)
        kms_policy.attach_to_role(self.sagemaker_studio_role)
        s3_policy.attach_to_role(self.sagemaker_studio_role)

        ssm.StringParameter(
            self,
            "ssm-sg-execution-role",
            parameter_name="/mlops/dev/domain/role/execution",
            string_value=self.sagemaker_studio_role.role_arn,
            simple_name=False,
        )
