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
    # Duration,
    Aws,
    Stack,
    aws_ssm as ssm,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_iot as iot,
    aws_lambda as _lambda,
    custom_resources as custres,
)
import constructs
import os

from config.constants import (
    PROJECT_NAME,
    PROJECT_ID,
    MODEL_PACKAGE_GROUP_NAME,
    DEV_ACCOUNT,
    ECR_REPO_ARN,
    MODEL_BUCKET_ARN,
    UBUNTU_AMI
)

class DeployEC2AndIotRole(Stack):
    """
    Deploy EC2 and Iot Role stack which provisions Iot related ressources to create an EC2 digital twin and the necessary role to run GDK deploy in the deployment accounts.
    """

    def __init__(
        self,
        scope: constructs,
        id: str,
        **kwargs,
    ):
        # The code that defines your stack goes here
        super().__init__(scope, id, **kwargs)

        # Get the instance type from the environment. If none then defaults c4.2xlarge.
        if "INSTANCE_TYPE" in os.environ:
            instance_type = os.getenv("INSTANCE_TYPE")
        else:
            instance_type = "m5.large"

        ####
        # 1. Create a VPC to control the network our instance lives on.
        ####
        vpc_id = ssm.StringParameter.value_from_lookup(self, "/vpc/id")
        vpc = ec2.Vpc.from_lookup(self, "VPC", vpc_id = vpc_id)
        
        sg_id = ssm.StringParameter.value_from_lookup(self, "/vpc/sg/id")
        # Create or refer to a security group that only allows inbound traffic.
        security_group = ec2.SecurityGroup.from_lookup_by_id(self, "SG", security_group_id = sg_id)
        # If creating new security group:
        # ec2.SecurityGroup(
        #     self,
        #     f"{PROJECT_NAME}-devices-security-group",
        #     vpc=vpc,
        #     allow_all_outbound=True,
        #     security_group_name=f"{PROJECT_NAME}-devices-security-group",
        # )
        
        ####
        # 2. Create role for edge devices
        ####
        tes_role = iam.Role(
            self,
            "simulated-tes-role",
            assumed_by=iam.ServicePrincipal("credentials.iot.amazonaws.com"),
        )

        ## Add permissions for simulated token exchange
        # https://docs.aws.amazon.com/greengrass/v2/developerguide/provision-minimal-iam-policy.html
        tes_policy = iam.Policy(
            self, 
            "simulated-tes-policy",
            policy_name=f"{tes_role.role_name}Access", # Expects the policy to have the same name as the role + Access
            statements=[iam.PolicyStatement(
                actions=[
                    "iot:DescribeCertificate",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                    "s3:GetBucketLocation"
                ],
                resources=["*"],
                effect=iam.Effect.ALLOW,
            )]
        )

        ## Add permissions for handling of kms encrypted material in dev account
        kms_policy = iam.Policy(self, 
            "power-user-decrypt-kms-policy",
            statements=[iam.PolicyStatement(
                actions= [
                    "kms:CreateAlias",
                    "kms:CreateKey",
                    "kms:Decrypt",
                    "kms:DeleteAlias",
                    "kms:Describe*",
                    "kms:GenerateRandom",
                    "kms:Get*",
                    "kms:List*",
                    "kms:TagResource",
                    "kms:UntagResource",
                    "iam:ListGroups",
                    "iam:ListRoles",
                    "iam:ListUsers"
                ],
                resources=["*"],
                effect=iam.Effect.ALLOW,
            )]
        )

        kms_policy.attach_to_role(tes_role)
        tes_policy.attach_to_role(tes_role)

        ## Add permissions for interacting with IoT Core
        tes_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "greengrass:CreateComponentVersion",
                    "greengrass:DescribeComponent"
                ],
                effect=iam.Effect.ALLOW,
                resources=["*"],
            )
        )

        ## Add permissions for interacting with S3, SageMaker
        tes_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonS3ReadOnlyAccess",
            )
        )

        tes_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AmazonSageMakerEdgeDeviceFleetPolicy",
            )
        )
        # https://github.com/aws/aws-cdk/issues/10320


        ## Create role for digital twin EC2 to act as greengrass devices
        edge_role = iam.Role(
            self,
            "gg-provisioning",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("ec2.amazonaws.com"),
                iam.ServicePrincipal("ssm.amazonaws.com")
            ),
        )

        ## Provide access to SSM for secure communication with the instance.
        edge_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSSMManagedInstanceCore",
            )
        )
        
        edge_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSSMPatchAssociation",
            )
        )

        ## Provide access to credentials and iot
        ec2_policy = iam.Policy(self, 
            "gg-provision-policy",
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "iam:AttachRolePolicy",
                        "iam:CreatePolicy",
                        "iam:CreateRole",
                        "iam:GetPolicy",
                        "iam:GetRole",
                        "iam:PassRole"
                    ],
                    effect=iam.Effect.ALLOW,
                    resources=[
                        f"arn:aws:iam::{Aws.ACCOUNT_ID}:policy/{tes_policy.policy_name}",
                        tes_role.role_arn,
                    ],
                    sid="CreateTokenExchangeRole",
                ),
                iam.PolicyStatement(
                    actions=[
                        "iot:AddThingToThingGroup",
                        "iot:AttachPolicy",
                        "iot:AttachThingPrincipal",
                        "iot:CreateKeysAndCertificate",
                        "iot:CreatePolicy",
                        "iot:CreateRoleAlias",
                        "iot:CreateThing",
                        "iot:CreateThingGroup",
                        "iot:DescribeEndpoint",
                        "iot:DescribeRoleAlias",
                        "iot:DescribeThingGroup",
                        "iot:GetPolicy"
                    ],
                    effect=iam.Effect.ALLOW,
                    resources=["*"],
                    sid="CreateIoTResources",
                ),
                iam.PolicyStatement(
                    actions=[
                        "greengrass:CreateDeployment",
                        "iot:CancelJob",
                        "iot:CreateJob",
                        "iot:DeleteThingShadow",
                        "iot:DescribeJob",
                        "iot:DescribeThing",
                        "iot:DescribeThingGroup",
                        "iot:GetThingShadow",
                        "iot:UpdateJob",
                        "iot:UpdateThingShadow"
                    ],
                    effect=iam.Effect.ALLOW,
                    resources=["*"],
                    sid="DeployDevTools",
                )
            ]
        )

        ## Provide access to artifacts bucket
        s3_policy = iam.Policy(self, 
            "gg-artifacts-bucket-access-policy",
            statements=[
                iam.PolicyStatement(
                    actions= [
                        "s3:PutObject",
                        "s3:PutObjectAcl",
                        "s3:GetObject",
                        "s3:GetObjectAcl",
                        "s3:GetObjectVersion",
                        "s3:GetBucketAcl",
                        "s3:GetBucketLocation"
                    ],
                resources=[
                    MODEL_BUCKET_ARN,
                    f"{MODEL_BUCKET_ARN}/*"
                ],
                effect=iam.Effect.ALLOW,
                )
            ]
        )

        ec2_policy.attach_to_role(edge_role)
        s3_policy.attach_to_role(edge_role)
        kms_policy.attach_to_role(edge_role)

        ####
        # 3. Provision custom resource to create IoT thing group and Token Exchange Role Alias
        ####
        iot_things_policy = custres.AwsCustomResourcePolicy.from_sdk_calls(
            resources=["*"]
        )

        thing_group_name = PROJECT_NAME
        iot_thing_group = custres.AwsCustomResource(self,
            id=f'iotThingGroup',
            policy=iot_things_policy,
            on_create=custres.AwsSdkCall(
                action='createThingGroup',
                service='Iot',
                parameters={
                    "thingGroupName": thing_group_name
                },
                # Must keep the same physical resource id, otherwise resource is deleted by CloudFormation
                physical_resource_id=custres.PhysicalResourceId.of(
                    thing_group_name
                ),
            ), 
            on_delete=custres.AwsSdkCall(
                action='deleteThingGroup',
                service='Iot',
                parameters={
                    "thingGroupName": thing_group_name
                },
                # Must keep the same physical resource id, otherwise resource is deleted by CloudFormation
                physical_resource_id=custres.PhysicalResourceId.of(
                        thing_group_name
                ),
            ),
            # resource_type='Custom::AWSIotThingGroup',
            timeout=None)


        role_alias_name = f"{PROJECT_NAME}-SimulatedEdgeTokenExchangeRoleAlias"
        iot_role_alias = custres.AwsCustomResource(self,
            id='iotRoleAlias',
            policy=iot_things_policy,
            on_create=custres.AwsSdkCall(
                action='createRoleAlias',
                service='Iot',
                parameters={
                    "roleAlias": role_alias_name,
                    "roleArn": tes_role.role_arn
                },
                # Must keep the same physical resource id, otherwise resource is deleted by CloudFormation
                physical_resource_id=custres.PhysicalResourceId.of(
                    role_alias_name),
                ), 
            on_delete=custres.AwsSdkCall(
                action='deleteRoleAlias',
                service='Iot',
                parameters={
                    "roleAlias": role_alias_name
                },
                # Must keep the same physical resource id, otherwise resource is deleted by CloudFormation
                physical_resource_id=custres.PhysicalResourceId.of(
                    role_alias_name),
                ),
            # resource_type='Custom::AWSIotThingGroup',
            timeout=None)

        iot_role_alias.grant_principal.add_to_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"], 
                resources=[tes_role.role_arn]
            )
        )


        ####
        # 4. Create role for publishing and deploying to greengrass
        ####
        deploy_role = iam.Role(
            self,
            "gg-deploy-role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("ec2.amazonaws.com"), # TODO if creating EC2 in preprod and prod: pass role from dev account as trusted entity instead of ec2
            ),
        )

        gg_deploy_policy = iam.Policy(self, 
            "gg-deploy-policy",
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "greengrass:ListComponentVersions",
                        "greengrass:CreateComponentVersion",
                        "greengrass:CreateDeployment",
                        "iot:DescribeThingGroup",
                        "iot:DescribeJob",
                        "iot:CreateJob",
                        "iot:CancelJob"
                    ],
                    effect=iam.Effect.ALLOW,
                    resources=["*"],
                    sid="PublishAndDeploy",
                ),
                iam.PolicyStatement(
                    actions=[
                        "s3:PutObject",
                        "s3:PutObjectAcl",
                        "s3:GetObject",
                        "s3:GetObjectAcl",
                        "s3:GetObjectVersion",
                        "s3:GetBucketAcl",
                        "s3:GetBucketLocation"
                    ],
                    effect=iam.Effect.ALLOW,
                    resources=[
                        MODEL_BUCKET_ARN,
                        f"{MODEL_BUCKET_ARN}/*"
                    ],
                    sid="AccessGGArtifactsBucket",
                ),
            ]
        )
        
        gg_deploy_policy.attach_to_role(deploy_role)
        ## Provide access to kms
        kms_policy.attach_to_role(deploy_role)

        # Add GG installation script to EC2
        multipart_user_data = ec2.MultipartUserData()
        commands_user_data = ec2.UserData.for_linux()
        with open("iot_infra/greengrass/install_ggv2.sh", "r") as f:
            greengrass_install = f.read()  
        greengrass_install = greengrass_install.replace("$1", self.region) \
            .replace("$2", f'{PROJECT_NAME}-digitaltwin') \
            .replace("$3", thing_group_name) \
            .replace("$4", tes_role.role_name) \
            .replace("$5", role_alias_name)
        commands_user_data.add_commands(greengrass_install)
        multipart_user_data.add_user_data_part(commands_user_data, ec2.MultipartBody.CLOUD_BOOTHOOK, True)
        
        # Increase the disk space on the device.
        root_volume = ec2.BlockDevice(
            device_name="/dev/xvda", volume=ec2.BlockDeviceVolume.ebs(25)
        )

        # Create a generic machine image for use with CPU.
        # Still hardcoded
        # TODO: PLEASE CHANGE AMI IF SWITCHING REGIONS. This is a specific ubuntu image required for the Edge application
        image = ec2.MachineImage.generic_linux(
             ami_map = {"eu-west-1": "ami-00aa9d3df94c6c354"}
        )

        # image = ec2.MachineImage.latest_amazon_linux(generation=ec2.AmazonLinuxGeneration.AMAZON_LINUX_2)

        # image = ec2.MachineImage.generic_linux(
        #      ami_map = {"eu-west-1": "ami-06d94a781b544c133"}
        # )
        
        # The market place UBUNTU AMIs do not connect to SSM so we had to hardcode the above version
        # image = ec2.MachineImage.generic_linux(
        #      ami_map = {self.region: UBUNTU_AMI}
        # )

        # Create ec2 instance to be used instead of edge device
        ec2.Instance(
            self,
            f"{PROJECT_NAME}-digitaltwin",
            role=edge_role,
            instance_type=ec2.InstanceType(instance_type),
            machine_image=image,
            vpc=vpc,
            security_group=security_group,
            user_data=multipart_user_data,
            block_devices=[root_volume],
        )
