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
from typing import List, Any
import logging
from logging import Logger
from cdk_service_catalog.products.constructs.base_product_stack import MLOpsBaseProductStack
from cdk_service_catalog.products.constructs.discovery.product_discovery import ProductDiscovery
from cdk_utilities.class_utilities import ClassUtilities
from cdk_service_catalog.products.constructs.discovery.product_config import ProductConfig
import inspect
from pathlib import Path


class ClassBasedProductDiscovery(ProductDiscovery):
    logging.basicConfig(level=logging.INFO)

    def __init__(self):
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.default_base_package: str = 'cdk_service_catalog.products'
        self.default_exclude_packages: List[str] = ['constructs', 'seed_code']

    def find_all(self, **kwargs) -> List[Any]:
        base_package: str = kwargs.get('base_package') if kwargs.get('base_package') else self.default_base_package
        exclude_packages: List[str] = kwargs.get('exclude_packages') \
            if kwargs.get('exclude_packages') else self.default_exclude_packages

        self.logger.info(f'base_package : {base_package} , exclude_packages : {str(exclude_packages)}')

        product_classes: List[Any] = ClassUtilities.find_subclasses(
            base_class=MLOpsBaseProductStack,
            base_package=base_package,
            exclude_packages=exclude_packages
        )
        return [ProductConfig(pc, self.get_class_base_path(pc)) for pc in product_classes]

    @staticmethod
    def get_class_base_path(class_ref) -> str:
        return str(Path(inspect.getfile(class_ref)).parent)

