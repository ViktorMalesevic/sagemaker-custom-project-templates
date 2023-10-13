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


class ClassBasedProduct(MLOpsBaseProductStack):

    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            asset_bucket: s3.Bucket = None,
            **kwargs
    ) -> None:
        super().__init__(scope, construct_id, asset_bucket=asset_bucket, **kwargs)

    def set_product_metadata(self):
        self.description = ("This template includes a model building pipeline that includes a workflow to pre-process, "
                            "train, evaluate and register a model. The deploy pipeline creates a dev,preprod and "
                            "production endpoint. The target DEV/PREPROD/PROD accounts are parameterized in this "
                            "template."
                            )
        self.product_name = ("Build & Deploy MLOps parameterized "
                             "template for real-time deployment"
                             )

        self.support_email = 'basic_project@example.com'

        self.support_url = 'https://example.com/support/basic_project'

        self.support_description = 'Example of support details for basic project'
