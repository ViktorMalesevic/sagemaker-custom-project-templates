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
from typing import List, Optional, Dict

from cdk_service_catalog.products.constructs.discovery.class_based_product_discovery import ClassBasedProductDiscovery
from cdk_service_catalog.products.constructs.discovery.configuration_based_product_discovery import \
    ConfigurationBasedProductDiscovery
from cdk_service_catalog.products.constructs.discovery.hybrid_based_product_discovery import HybridBasedProductDiscovery
from cdk_service_catalog.products.constructs.discovery.product_config import ProductConfig
from cdk_service_catalog.products.constructs.discovery.product_discovery import ProductDiscovery


class ProductDiscoveryHelper:
    logging.basicConfig(level=logging.INFO)

    DISCOVERY_TYPE_CONFIGURATION: str = 'config'
    DISCOVERY_TYPE_CLASS: str = 'class'
    DISCOVERY_TYPE_HYBRID: str = 'hybrid'

    DEFAULT_DISCOVERY_PRECEDENCE: str = 'class'

    def __init__(self, discovery_type: str = None, discovery_precedence: str = None):
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.discovery_type: str = discovery_type \
            if discovery_type and discovery_type.strip() != '' else self.DISCOVERY_TYPE_HYBRID
        self.discovery_precedence: str = discovery_precedence \
            if discovery_precedence and discovery_precedence.strip() != '' else self.DEFAULT_DISCOVERY_PRECEDENCE

    def get_product_classes(self, **kwargs) -> List[ProductConfig]:
        self.logger.info(f'Getting product classes...')

        discovery: Optional[ProductDiscovery] = self.get_product_discovery()

        return discovery.find_all(**kwargs)

    def get_product_discovery(self) -> ProductDiscovery:
        discoveries: Dict[str, ProductDiscovery] = {
            self.DISCOVERY_TYPE_CONFIGURATION: ConfigurationBasedProductDiscovery(),
            self.DISCOVERY_TYPE_CLASS: ClassBasedProductDiscovery(),
            self.DISCOVERY_TYPE_HYBRID: HybridBasedProductDiscovery(self.discovery_precedence)
        }

        return discoveries.get(str(self.discovery_type).strip().lower(),
                               HybridBasedProductDiscovery(self.discovery_precedence))
