import json
import boto3
import cfnresponse
from botocore.exceptions import ClientError

sm_client = boto3.client("sagemaker")


def handler(event, context):
    try:
        if "RequestType" in event and event["RequestType"] in {"Create"}:
            properties = event["ResourceProperties"]
            create_rule(properties)

        elif "RequestType" in event and event["RequestType"] in {"Update"}:
            properties = event["ResourceProperties"]
            delete_rule(properties)
            create_rule(properties)

        elif "RequestType" in event and event["RequestType"] in {"Delete"}:
            properties = event["ResourceProperties"]
            delete_rule(properties)

        cfnresponse.send(event, context, cfnresponse.SUCCESS, {}, "")
    except ClientError as exception:
        print(exception)
        cfnresponse.send(
            event,
            context,
            cfnresponse.FAILED,
            {},
            physicalResourceId=event.get("PhysicalResourceId"),
        )


def create_rule(properties):
    rule_name = properties["rule_name"]
    script = properties["script"]
    rule_type = properties["rule_type"]

    response = sm_client.create_studio_lifecycle_config(
        StudioLifecycleConfigName=rule_name,
        StudioLifecycleConfigContent=script,
        StudioLifecycleConfigAppType=rule_type,  # 'JupyterServer'|'KernelGateway',
    )
    rule_arn = response["StudioLifecycleConfigArn"]

    attach_domain = False
    try:
        domain_id = properties["domain_id"]
        attach_domain = True
    except:
        print("No domain specified")

    if attach_domain == True:
        if rule_type == "JupyterServer":
            response = sm_client.update_domain(
                DomainId=domain_id,
                DefaultUserSettings={
                    "JupyterServerAppSettings": {
                        "DefaultResourceSpec": {
                            "InstanceType": "system",
                            "LifecycleConfigArn": rule_arn,
                        },
                        "LifecycleConfigArns": [
                            rule_arn,
                        ],
                    }
                },
            )
        elif rule_type == "KernelGateway":
            response = sm_client.update_domain(
                DomainId=domain_id,
                DefaultUserSettings={
                    "KernelGatewayAppSettings": {
                        "DefaultResourceSpec": {
                            "InstanceType": "system",
                            "LifecycleConfigArn": rule_arn,
                        },
                        "LifecycleConfigArns": [
                            rule_arn,
                        ],
                    }
                },
            )


def delete_rule(properties):
    rule_name = properties["rule_name"]
    response = sm_client.delete_studio_lifecycle_config(
        StudioLifecycleConfigName=rule_name
    )
