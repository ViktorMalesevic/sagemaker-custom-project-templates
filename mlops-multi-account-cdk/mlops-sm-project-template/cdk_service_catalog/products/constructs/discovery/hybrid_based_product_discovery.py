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
from typing import List

from cdk_service_catalog.products.constructs.discovery.class_based_product_discovery import ClassBasedProductDiscovery
from cdk_service_catalog.products.constructs.discovery.configuration_based_product_discovery import \
    ConfigurationBasedProductDiscovery
from cdk_service_catalog.products.constructs.discovery.product_config import ProductConfig
from cdk_service_catalog.products.constructs.discovery.product_discovery import ProductDiscovery


class HybridBasedProductDiscovery(ProductDiscovery):
    logging.basicConfig(level=logging.INFO)

    DISCOVERY_PRECEDENCE_CLASS: str = 'class'
    DISCOVERY_PRECEDENCE_CONFIG: str = 'config'

    def __init__(self, discovery_precedence: str):
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.discovery_precedence: str = discovery_precedence
        self.class_discovery: ProductDiscovery = ClassBasedProductDiscovery()
        self.config_discovery: ProductDiscovery = ConfigurationBasedProductDiscovery()

    def find_all(self, **kwargs) -> List[ProductConfig]:
        self.logger.debug(f'kwargs : {kwargs}')
        class_products: List[ProductConfig] = self.class_discovery.find_all(**kwargs)
        conf_products: List[ProductConfig] = self.config_discovery.find_all(**kwargs)

        if self.discovery_precedence == self.DISCOVERY_PRECEDENCE_CLASS:
            products = [*class_products,
                        *list(filter(lambda y: y.path not in map(lambda x: x.path, class_products), conf_products))]
        elif self.discovery_precedence == self.DISCOVERY_PRECEDENCE_CONFIG:
            products = [*conf_products,
                        *list(filter(lambda y: y.path not in map(lambda x: x.path, conf_products), class_products))]
        else:
            products = [*class_products, *conf_products]

        return products
