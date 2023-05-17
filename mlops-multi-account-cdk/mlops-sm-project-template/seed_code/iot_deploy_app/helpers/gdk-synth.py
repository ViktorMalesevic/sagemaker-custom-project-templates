import argparse
import json
import logging
import os
import yaml
import boto3
from botocore.exceptions import ClientError
import s3fs
import sagemaker

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")
sagemaker_session = sagemaker.Session(sagemaker_client=sm_client)
fs = s3fs.S3FileSystem()
soft_dependencies = {
    "aws.greengrass.Nucleus": {
        "DependencyType": "SOFT",
        "VersionRequirement": ">=2.0.3 <3.1.0"
    },
}

class GreengrassComponent:
    def __init__(self, **kwargs):
        self.component_name = kwargs.get('component_name')
        self.folder = kwargs.get('RootFolder')
        self.bucket = kwargs.get('bucket')
        self.region = kwargs.get('region')
        self.lifecycle_instructions = kwargs.get('lifecycle_instructions')
        self.artifacts = kwargs.get("artifacts")
        self.dependencies = kwargs.get("dependencies")
        self.version = kwargs.get("version")
        self.build = kwargs.get("build")


    def create_gdk_config(self, path):
        gdk_config = {
            "component": {
                self.component_name: {
                    "author": "mlops@edge-cicd",
                    "version": self.version,
                    "build": self.build,
                    "publish": {
                        "bucket": self.bucket,
                        "region": self.region
                    }
                }
            },
            "gdk_version": "1.1.0"
        }

        os.makedirs(path, exist_ok=True) 
        with open(f"{path}/gdk-config.json", "w") as f:
            json.dump(gdk_config, f, indent=4)

        return None
    

    def create_recipe(self, path):
        recipe = {
            "RecipeFormatVersion": "2020-01-25",
            "ComponentDependencies": self.dependencies,
            "Manifests": [
                {
                "Platform": {
                    "os": "all"
                },
                "Lifecycle": self.lifecycle_instructions,
                "Artifacts": self.artifacts
                }
            ]
        }

        os.makedirs(path, exist_ok=True) 
        with open(f"{path}/recipe.json", "w") as f:
            json.dump(recipe, f, indent=4)
        return None
    

def create_deployment(project_name_id, components):
    deployment = {
        "targetArn": "$THING_GROUP_ARN$",
        "deploymentName": f"{project_name_id}-greengrass-deployment",
        "components": {
            "aws.greengrass.Cli": {
            "componentVersion": "2.9.4",
            "configurationUpdate": {}
            }
        }
    }

    for component in components:
        deployment["components"][component.component_name] = {
            "componentVersion": component.version,
            "configurationUpdate": {}
        }

    with open("./deployment.json", "w") as f:
        json.dump(deployment, f, indent=4)
    return None


def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug("Getting more packages for token: {}".format(response["NextToken"]))
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


def download_model_package_version(path, model_package_arn):
    """Downloads the latest approved model package for a model package group.

    Args:
        path: The path where to download the model.
        model_package_arn: The model package ARN.
        bucket: S3 bucket where the model is stored.

    Returns:
        None.
    """
    try:
        # get info on latest model package
        latest_model_package = sm_client.describe_model_package(ModelPackageName=model_package_arn)
        model_s3_uri = latest_model_package['InferenceSpecification']['Containers'][0]['ModelDataUrl']

        # download latest version of model
        fs.download(model_s3_uri, f'{path}/model.tar.gz')
    
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


def get_latest_version(artifacts_bucket, component_name):
    """Lists prefixes or objects under an input s3 path.
    Args:
        artifacts_bucket: The s3 bucket for archiving greengrass components.
        component_name: The name of the greengrass component whose version to retrieve.
    Returns:
        versions: list of archived versions for the input component.
    """
    logger.info(f"Getting latest version for component: {component_name}")
    logger.info(f"Reading from bucket: {artifacts_bucket}")
    objs = fs.ls(path=f"s3://{artifacts_bucket}/artifacts/greengrass/{component_name}")
    child_paths = [obj.split(f'{component_name}/')[1].split('/')[0] for obj in objs]
    if not child_paths:
        version = "1.0.0"
    else:
        child_paths.sort(reverse=True)
        version = child_paths[0]
    return version


def update_version(last_version, how="patch"):
    how=how.lower()
    parts = ["major", "minor", "patch"]
    if how not in parts:
        raise ValueError("Incorrect versioning strategy.")
    ix = parts.index(how)
    values = last_version.split('.')
    values[ix] = str(int(values[ix]) + 1)
    return ".".join(values)


# TODO: use MODEL_BUCKET_URI
def update_s3_uri(config):
    """
    Updates S3 location of greengrass components
    """
    URI = config["Artifacts"][0]["URI"]
    if ("$BUCKET_NAME$" in URI and config["bucket"]):
        URI = URI.replace("$BUCKET_NAME$", config["bucket"])
    else:
        raise ValueError("Missing bucket name")
    if ("$COMPONENT_NAME$" in URI and config["component-name"]):
        URI = URI.replace("$COMPONENT_NAME$", config["component-name"])
    else:
        raise ValueError("Missing component name")
    if ("$COMPONENT_VERSION$" in URI and config["version"]):
        URI = URI.replace("$COMPONENT_VERSION$", config["version"])
    else:
        raise ValueError("Missing component version")

    config["Artifacts"][0]["URI"]=URI

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--import-config-file", type=str, default="greengrass-config.yml")
    parser.add_argument("--project-name-id", type=str)
    parser.add_argument("--artifact-bucket-name", type=str)
    parser.add_argument("--region", type=str)
    parser.add_argument("--cache-models", type=str, default="False")
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    # Read the staging config
    with open(args.import_config_file, "r") as f:
        greengrass_config = yaml.safe_load(f)

    # 1. evaluate components version and download latest models from SageMaker if SageMaker managed model
    for item, config in greengrass_config["custom-components"].items():
        # change component name to match project name
        component_name = config["component-name"]
        if '$PROJECT_NAME_ID$' in component_name:
            component_name = component_name.replace('$PROJECT_NAME_ID$', args.project_name_id)
            config["component-name"] = component_name
            if config["build"].get("custom_build_command"):
                custom_build_command = config["build"].get("custom_build_command")
                new_custom_build_command = []
                for line in custom_build_command:
                    if '$PROJECT_NAME_ID$' in line:
                        logger.info(f"Replacing PROJECT_NAME_ID in {line}")
                        line = line.replace('$PROJECT_NAME_ID$', args.project_name_id)
                        logger.info(f"Updated custom command line: {line}")
                    new_custom_build_command.append(line)
                config["build"]["custom_build_command"] = new_custom_build_command
        else:
            raise ValueError("Project name missing")

        # evaluate if component is model and versioning strategy
        model = config.get("model")
        sagemaker_managed = ""
        if model: 
            sagemaker_managed = model.get("sagemaker-managed")

        if sagemaker_managed:
            # 1. Versioning by use of latest approved package version
            # TODO: change logic to allow for several model package groups
            model_package_arn = get_approved_package(args.project_name_id)
            config["version"] = f"{model_package_arn.split('/')[-1]}.0.0"

            caching = eval(args.cache_models)
            missing_model = model["name"] not in os.listdir(config['root'])
            if not caching or missing_model: 
                # downloading latest model version
                download_model_package_version(config['root'], model_package_arn)

        else:
            # 2. Versioning by use of versions file
            last_version = get_latest_version(args.artifact_bucket_name, component_name)
            config["version"] = update_version(last_version, how='patch')

        config["bucket"] = args.artifact_bucket_name
        config = update_s3_uri(config)


    # 2. update component dependencies to match latest version and create gdk-config.json and recipe.json files
    components = [v["component-name"] for k,v in greengrass_config["custom-components"].items()]
    models = []

    model_dependencies = {}
    for item, config in greengrass_config["custom-components"].items():
        model = config.get("model")
        if model:
            models.append(config["component-name"])
            # add soft dependencies to model components
            config["ComponentDependencies"] = soft_dependencies
            model_dependencies[config["component-name"]] = {
                "DependencyType":"HARD",
                "VersionRequirement":config["version"]
            }

    # add model dependencies to non-model components
    hard_dependencies = soft_dependencies.copy()
    hard_dependencies.update(model_dependencies)
    for item, config in greengrass_config["custom-components"].items():
        if config["component-name"] not in models:
            # config["ComponentDependencies"].update(dependencies)
            config["ComponentDependencies"] = hard_dependencies

    # dump update yaml file
    with open("greengrass-config-updated.yml", "w") as file:
        yaml.dump(greengrass_config, file)

    ## build files
    deploy_components = []
    for item, config in greengrass_config["custom-components"].items():
        component = GreengrassComponent(
            component_name=config["component-name"],
            bucket=config["bucket"],
            region=args.region,
            folder=config["root"], 
            artifacts=config["Artifacts"],
            dependencies=config["ComponentDependencies"],
            lifecycle_instructions=config["Lifecycle"],
            version=config["version"],
            build=config["build"]
        )

        component.create_gdk_config(path=config["root"])
        component.create_recipe(path=config["root"])
        deploy_components.append(component)

    # 3. create deployment.json 
    create_deployment(args.project_name_id, deploy_components)
