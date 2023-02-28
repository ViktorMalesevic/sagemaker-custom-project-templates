#!/usr/bin/env python3
import aws_cdk as cdk
from dynamic_accounts_project_stack import MLOpsStack

app = cdk.App()
MLOpsStack(
    app,
    'DynamicAccountsLocalCDKStack',
    env=cdk.Environment(region='eu-west-1'),
    synthesizer=cdk.DefaultStackSynthesizer()
)
app.synth()

'''
cdk deploy \
    --app "python3 local_test_app.py"\
    --parameters SageMakerProjectName=mlops-test \
    --parameters SageMakerProjectId=project-id \
    --parameters PreProdAccount=<pre_prod> \
    --parameters ProdAccount=<prod> \
    --parameters DeploymentRegion=eu-west-1 \
    --parameters LocalDeployment=True
'''