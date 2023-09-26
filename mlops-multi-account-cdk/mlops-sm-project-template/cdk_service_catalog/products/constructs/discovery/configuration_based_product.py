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
    aws_s3 as s3,
)
from constructs import Construct

from cdk_service_catalog.products.constructs.base_product_stack import MLOpsBaseProductStack
from cdk_utilities.mlops_project_config import MlopsProjectConfig, MetadataConfig


class ConfigurationBasedProduct(MLOpsBaseProductStack):

    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            asset_bucket: s3.Bucket = None,
            mlops_project_config: MlopsProjectConfig = None,
            **kwargs
    ) -> None:
        self.mlops_project_config: MlopsProjectConfig = mlops_project_config
        super().__init__(scope, construct_id, asset_bucket=asset_bucket, **kwargs)

    def set_product_metadata(self):
        metadata: MetadataConfig = self.mlops_project_config.metadata
        self.description = metadata.description
        self.product_name = metadata.product_name
        self.support_email = metadata.support_email
        self.support_url = metadata.support_url
        self.support_description = metadata.support_description

    def get_build_app_seed_code_relative_path(self) -> str:
        build_path: str = str(self.mlops_project_config.seed_code.build_app_relative_path).strip()
        return build_path if build_path else self._default_build_app_seed_code_relative_path

    def get_deploy_app_seed_code_relative_path(self) -> str:
        deploy_path: str = str(self.mlops_project_config.seed_code.deploy_app_relative_path).strip()
        return deploy_path if deploy_path else self._default_deploy_app_seed_code_relative_path

    def get_seed_code_base_path(self) -> str:
        return self.mlops_project_config.seed_code.absolute_base_path
