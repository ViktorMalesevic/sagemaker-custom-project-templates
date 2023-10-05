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
    Stack,
)
from constructs import Construct
from mlops_infra.networking_stack import NetworkingStack
from mlops_infra.sagemaker_studio_stack import SagemakerStudioStack


class SagemakerInfraStack(Stack):

    def __init__(
            self,
            scope: Construct,
            construct_id: str,
            app_prefix: str,
            deploy_sm_domain: bool = False,
            **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        networking_stack = NetworkingStack(self, "networking", deploy_sm_domain, **kwargs)

        # TODO: If SM Studio is not created in the dev account, the mlops-sm-project-template Service Catalog
        #  Portfolio still expects an execution role in as SSM /mlops/role/lead which will have to be created
        #  manually and will need to have ssm:PutParameter policy
        if deploy_sm_domain:
            sagemaker_studio_stack = SagemakerStudioStack(
                self,
                "sagemaker-studio",
                app_prefix=app_prefix,
                vpc=networking_stack.primary_vpc,
                subnets=networking_stack.primary_vpc.private_subnets,
                **kwargs,
            )
