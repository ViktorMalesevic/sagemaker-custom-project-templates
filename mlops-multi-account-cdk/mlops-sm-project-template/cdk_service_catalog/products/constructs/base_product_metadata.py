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
from typing import List
from aws_cdk import (
    aws_s3 as s3,
    aws_servicecatalog as sc,
)
from constructs import Construct
from abc import abstractmethod


class BaseProductMetadata(sc.ProductStack):

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str):
        self._description = value

    @property
    def support_email(self) -> str:
        return self._support_email

    @support_email.setter
    def support_email(self, value: str):
        self._support_email = value

    @property
    def product_name(self) -> str:
        return self._template_name

    @product_name.setter
    def product_name(self, value: str):
        self._template_name = value

    @property
    def support_url(self) -> str:
        return self._support_url

    @support_url.setter
    def support_url(self, value: str):
        self._support_url = value

    @property
    def support_description(self) -> str:
        return self._support_description

    @support_description.setter
    def support_description(self, value: str):
        self._support_description = value

    def get_subclass_realpath(self, base_dir: str) -> str:
        # cdk_service_catalog.products.train_deploy_basic_product.basic_project
        # last part is module i.e, python file name
        sub_path: str = ""
        sub_class_package: str = self.__class__.__module__

        # removing last part as that is the python file name, which we don't want
        parts: List[str] = sub_class_package.split('.')[:-1]

        # common part from base_dir will be last part with sub_class_package parts
        common_part: str = base_dir.split(os.path.sep)[-1]
        if common_part in parts:
            sub_path = os.path.sep.join(parts).split(f'{os.path.sep}{common_part}{os.path.sep}')[-1]
        return f'{base_dir}{os.path.sep}{sub_path}'

    @abstractmethod
    def get_seed_code_base_path(self) -> str:
        pass

    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            asset_bucket: s3.Bucket = None,
            **kwargs
    ) -> None:
        super().__init__(scope, construct_id, asset_bucket=asset_bucket, **kwargs)
        # ################ Product Metadata #############################################################

        self._description: str = ''
        self._template_name: str = ''

        self._support_email: str = ''

        self._support_url: str = ''

        self._support_description: str = ''
        self.set_product_metadata()
        # ###############################################################################################

    @abstractmethod
    def set_product_metadata(self):
        pass
