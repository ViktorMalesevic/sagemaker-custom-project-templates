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
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_codecommit as codecommit,
)

from constructs import Construct
from cdk_utilities.zip_utils import ZipUtility
from cdk_utilities.cdk_app_config import (
    PipelineConfig,
    CodeCommitConfig
)


class CdkPipelineCodeCommitStack(Stack):
    INSTANCE = None

    def __init__(self, scope: Construct, construct_id: str, conf: CodeCommitConfig, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        base_dir: str = os.path.abspath(f'{os.path.dirname(__file__)}{os.path.sep}..')

        self.repo = codecommit.Repository(
            self,
            "BuildAppCodeRepo",
            repository_name=conf.repo_name,
            description="CDK Code with Sagemaker Projects Service Catalog products",
            code=codecommit.Code.from_zip_file(file_path=ZipUtility.create_zip(base_dir), branch=conf.branch_name)
        )

    @classmethod
    def get_repo(cls, scope, pipeline_conf: PipelineConfig) -> codecommit.Repository:
        if not cls.INSTANCE:
            cls.INSTANCE = CdkPipelineCodeCommitStack(
                scope,
                'ml-sc-cc-repo',
                conf=pipeline_conf.code_commit,
                env=cdk.Environment(account=str(pipeline_conf.account), region=pipeline_conf.region)
            )
        return cls.INSTANCE.repo
