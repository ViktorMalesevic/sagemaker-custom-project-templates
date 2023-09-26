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
import logging
from logging import Logger
from typing import List, Any, Dict
from pathlib import Path

from cdk_service_catalog.products.constructs.discovery.configuration_based_byoc_product import \
    ConfigurationBasedBYOCProduct
from cdk_service_catalog.products.constructs.discovery.configuration_based_product import ConfigurationBasedProduct
from cdk_service_catalog.products.constructs.discovery.product_config import ProductConfig
from cdk_service_catalog.products.constructs.discovery.product_discovery import ProductDiscovery
from cdk_utilities.mlops_project_config import ProjectConfig, MlopsProjectConfig


class ConfigurationBasedProductDiscovery(ProductDiscovery):
    DEFAULT_APP_TYPE_SIMPLE: str = 'simple'
    APP_TYPE_SIMPLE: str = 'simple'
    APP_TYPE_CONTAINER: str = 'container'

    def __init__(self):
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.default_config_filename: str = 'mlops_project.yml'
        self.default_project_base_path: str = os.path.abspath(
            f'{os.path.dirname(__file__)}{os.path.sep}..{os.path.sep}..'
        )
        self.default_base_package: str = 'cdk_service_catalog.products'
        self.default_exclude_packages: List[str] = ['constructs', 'seed_code']

    def find_all(self, **kwargs) -> List[ProductConfig]:
        filter_folders: List[str] = kwargs.get('exclude_packages') \
            if kwargs.get('exclude_packages') else self.default_exclude_packages
        config_filename: str = kwargs.get('product_config_filename') \
            if kwargs.get('product_config_filename') else self.default_config_filename
        project_base_path: str = kwargs.get('product_base_path') \
            if kwargs.get('product_base_path') else self.default_project_base_path

        self.logger.info(f'filter_folders : {str(filter_folders)}, '
                         f'config_filename : {config_filename}, project_base_path : {project_base_path}')

        configs: List[ProductConfig] = list()
        for path in filter(lambda x: (len([ep for ep in filter_folders if f'/{ep}/' in str(x)]) == 0),
                           Path(project_base_path).glob(f'**{os.path.sep}{config_filename}')
                           ):
            pc: ProjectConfig = ProjectConfig()
            pc.load(path)

            conf: MlopsProjectConfig = pc.mlops_project_config
            if not Path(conf.seed_code.absolute_base_path).exists() or str(
                    conf.seed_code.absolute_base_path).strip() in ['', '.', './']:
                if not Path(conf.seed_code.absolute_base_path).exists():
                    self.logger.warning(f'Given absolute base path for seed code is not valid, '
                                        f'using default base path which the location of the {config_filename}')
                conf.seed_code.absolute_base_path = str(path.parent)
            class_ref: Any = self.get_class_by(conf.seed_code.app_type)
            configs.append(ProductConfig(class_ref, str(path.parent), conf))

        return configs

    @classmethod
    def get_class_by(cls, app_type: str) -> Any:
        classes: Dict[str, Any] = {
            cls.APP_TYPE_SIMPLE: ConfigurationBasedProduct,
            cls.APP_TYPE_CONTAINER: ConfigurationBasedBYOCProduct
        }
        return classes.get(str(app_type).strip().lower(), ConfigurationBasedProduct)
