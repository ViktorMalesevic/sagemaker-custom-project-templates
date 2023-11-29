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


# !/usr/bin/env python3

from logging import Logger

import aws_cdk as cdk

import cdk_utilities
from cdk_pipelines.codecommit_stack import CdkPipelineCodeCommitStack
from cdk_pipelines.pipeline_stack import CdkPipelineStack
from mlops_commons.utilities.cdk_app_config import CdkAppConfig
from mlops_commons.utilities.config_helper import ConfigHelper
from mlops_commons.utilities.log_helper import LogHelper


class MLOpsInfraCdkApp:

    def __init__(self):
        self.logger: Logger = LogHelper.get_logger(self)
        self.logger.info(f'mlops_commons path : {cdk_utilities.mlops_commons_base_dir}')

    def main(self):

        self.logger.info('Starting cdk app...')

        app = cdk.App()
        config_helper: ConfigHelper = ConfigHelper()
        cac: CdkAppConfig = config_helper.app_config.cdk_app_config

        for dc in cac.deployments:

            self.logger.info(f'Start deploying config set_name : {dc.set_name}')

            if not dc.enabled:
                self.logger.info(f'Skipping deployment of config ->'
                                 f'set name : {dc.set_name}, '
                                 f' as it is disabled in configuration file. To enable it, set the attribute '
                                 f'enabled=True at deployments level in yaml configuration file ')
                continue

            # if there are more than on business unit config then backup main config file
            # then create a business unit specific config
            # then create code commit repo
            # then restore the config file for local reference but this main file will not be available in specific repo
            if len(cac.deployments) > 1:
                config_helper.backup_config_file()
                config_helper.create_set_name_specific_config(set_name=dc.set_name)

            repo_stack: CdkPipelineCodeCommitStack = CdkPipelineCodeCommitStack.get_instance(
                app,
                set_name=dc.set_name,
                pipeline_conf=cac.pipeline
            )

            if len(cac.deployments) > 1:
                config_helper.restore_config_file()

            CdkPipelineStack(
                app,
                f"ml-infra-deploy-pipeline-{dc.set_name}",
                app_prefix=cac.app_prefix,
                set_name=dc.set_name,
                deploy_stages_conf=dc.stages,
                pipeline_conf=cac.pipeline,
                description="CI/CD CDK Pipelines for MLOps Infra",
                env=cdk.Environment(account=str(cac.pipeline.account), region=cac.pipeline.region)
            ).add_dependency(repo_stack)

        app.synth()


if __name__ == "__main__":
    MLOpsInfraCdkApp().main()
