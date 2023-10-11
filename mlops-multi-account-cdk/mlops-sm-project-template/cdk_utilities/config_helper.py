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


import logging
from logging import Logger
import os
from typing import Optional, Dict, List, Set

from cdk_utilities.cdk_app_config import (
    AppConfig,
    AppConfigOld,
    CdkAppConfig
)
from pathlib import Path


class ConfigHelper:
    logging.basicConfig(level=logging.INFO)

    INSTANCE: 'ConfigHelper' = None

    def __init__(self):
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.base_dir: str = os.path.abspath(f'{os.path.dirname(__file__)}{os.path.sep}..')
        self.default_region: str = os.environ['CDK_DEFAULT_REGION']
        self.default_account: str = os.environ['CDK_DEFAULT_ACCOUNT']
        self.app_config: Optional[AppConfig] = AppConfig()

        self.logger.info(f'cdk app base directory : {self.base_dir}')
        self.logger.info(f'CDK_DEFAULT_REGION : {self.default_region}')
        self.logger.info(f'CDK_DEFAULT_ACCOUNT : {self.default_account}')

        yaml_config_path: str = os.path.join(
            self.base_dir,
            # 'cdk_service_catalog',
            'config',
            'cdk-app.yml'
        )

        self.logger.info(f'Trying to loading cdk app configuration file : {yaml_config_path}')
        load_status: bool = False
        if os.path.exists(yaml_config_path):
            self.app_config.load(Path(yaml_config_path))
            load_status = True

        if not load_status:
            self.logger.info(f'cdk app yaml config not found at path : {yaml_config_path}')
            json_config_path: str = os.path.join(
                self.base_dir,
                'cdk_service_catalog',
                'config',
                'accounts.json'
            )
            self.logger.info(f'Now Trying to loading cdk app configuration file : {json_config_path}')
            if os.path.exists(json_config_path):
                app_config_old: AppConfigOld = AppConfigOld()
                app_config_old.load(file_path=json_config_path)
                self.app_config = app_config_old.get_new_app_config()
        self.logger.info(f'cdk app config : {str(self.app_config)}')
        ConfigHelper.INSTANCE = self

    @classmethod
    def get_config(cls) -> CdkAppConfig:

        if cls.INSTANCE is None:
            cls.INSTANCE = ConfigHelper()
        return cls.INSTANCE.app_config.cdk_app_config

    @classmethod
    def get_regions_by_account(cls, set_name: str, account: str, stage_name) -> List[str]:
        account_regions: Set[str] = set()
        for ds in filter(lambda x: str(x.set_name).strip().lower() == str(set_name).strip().lower(), cls.INSTANCE.app_config.cdk_app_config.deployments):
            for stage in filter(lambda y:
                                str(y.stage_name).strip().lower().startswith(str(stage_name).strip().lower()) and str(account).strip() == str(y.account).strip(),
                                ds.get_deployment_stages()):
                account_regions.add(stage.region)
        return [*account_regions]

    @classmethod
    def get_regions_by(cls, set_name: str, stage_name) -> List[str]:
        regions: Set[str] = set()
        for ds in filter(lambda x: x.set_name == set_name, cls.INSTANCE.app_config.cdk_app_config.deployments):
            for stage in filter(lambda y: str(y.stage_name).strip().lower().startswith(str(stage_name).strip().lower()),
                                ds.get_deployment_stages()):
                regions.add(stage.region)
        return [*regions]

    @classmethod
    def get_accounts_by(cls, set_name: str, stage_name: str) -> List[str]:
        accounts: Set[str] = set()
        for ds in filter(lambda x: x.set_name == set_name, cls.INSTANCE.app_config.cdk_app_config.deployments):
            for stage in filter(lambda y: str(y.stage_name).strip().lower().startswith(str(stage_name).strip().lower()),
                                ds.get_deployment_stages()):
                accounts.add(str(stage.account))
        return [*accounts]

# os.environ['CDK_DEFAULT_REGION']='us'
# os.environ['CDK_DEFAULT_ACCOUNT']='us-west-2'
# c = ConfigHelper()
#
# print(c.get_accounts_by('second-example', 'prod'))