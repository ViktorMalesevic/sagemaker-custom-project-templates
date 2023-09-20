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

import json
import boto3
import time
import os
import model_card


def lambda_handler(event, context):
    sm_client = boto3.client('sagemaker')
    print(event)
    with open(os.getcwd() + '/model_card_template.txt') as f:
        file = f.read()
    file = json.loads(file.replace("'", '"'))
    role_arn = os.environ['role']
    model_arn = sm_client.create_model(PrimaryContainer={
        'ModelPackageName': event['detail']['ModelPackageArn']},
        ExecutionRoleArn=role_arn,
        ModelName='-'.join(event['detail']['ModelPackageName'].split('/')))
    model_arn = model_arn['ModelArn']
    model_card._create_model_card(file, event, model_arn)
    return {
        'statusCode': 200,
        'body': json.dumps('Created Model Object and Card!')
    }
