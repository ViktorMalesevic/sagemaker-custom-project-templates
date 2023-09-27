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


class BatchInferenceProjectProduct(MLOpsBaseProductStack):

    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            asset_bucket: s3.Bucket = None,
            **kwargs
    ) -> None:
        super().__init__(scope, construct_id, asset_bucket=asset_bucket, **kwargs)

    def set_product_metadata(self):
        self.description = ("This template includes a build and a deploy code repository (CodeCommit) associated "
                            "to their respective CICD pipeline (CodePipeline). The build repository and CICD pipeline "
                            "are used to run SageMaker pipeline(s) in dev and promote the pipeline definition to an "
                            "artefact bucket. The deploy repository and CICD pipeline loads the artefact SageMaker "
                            "pipeline definition to create a Sagemaker pipeline in preprod and production as "
                            "infrastructure as code (eg for batch inference). The target PREPROD/PROD accounts are "
                            "provided as cloudformation parameters and must be provided during project creation. "
                            "The PREPROD/PROD accounts need to be cdk bootstraped in advance to have the right "
                            "CloudFormation execution cross account roles.")
        self.product_name = "MLOps Batch Inference template to build and deploy SageMaker pipeline"

        self.support_email = 'batch_inference_project@example.com'

        self.support_url = 'https://example.com/support/batch_inference_project'

        self.support_description = 'Example of support details for batch inference project'

    def get_create_model_event_rule(self) -> bool:
        return self.BOOL_FALSE
