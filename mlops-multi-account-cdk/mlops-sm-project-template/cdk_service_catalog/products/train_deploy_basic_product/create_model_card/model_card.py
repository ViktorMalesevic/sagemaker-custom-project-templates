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

sm_client = boto3.client('sagemaker')


def _create_model_card(file, event, model_arn):
    file['model_overview']['model_name'] = '-'.join(event['detail']['ModelPackageName'].split('/'))
    file['model_overview']['model_id'] = model_arn
    file['model_overview']['model_artifact'] = [x['ModelDataUrl'] for x in
                                                event['detail']['InferenceSpecification']['Containers']]
    file['model_overview']['model_version'] = event['detail']['ModelPackageVersion']
    file['model_overview']['problem_type'] = ""
    file['model_overview']['algorithm_type'] = \
    event['detail']['InferenceSpecification']['Containers'][-1]['Image'].split('/')[-1]
    file['model_overview']['model_description'] = ""
    file['model_overview']['model_creator'] = ""
    file['model_overview']['model_owner'] = ""
    file['model_overview']['inference_environment']['container_image'] = [x['Image'] for x in
                                                                          event['detail']['InferenceSpecification'][
                                                                              'Containers']]
    file['business_details']['business_problem'] = ""
    file['business_details']['business_stakeholders'] = ""
    file['business_details']['line_of_business'] = ""
    file['intended_uses']['intended_uses'] = ""
    file['intended_uses']['explanations_for_risk_rating'] = ""
    file['intended_uses']['factors_affecting_model_efficiency'] = "Data Quality"
    file['intended_uses']['risk_rating'] = "Low"
    file['training_details']['training_job_details']['training_arn'] = ""
    file['training_details']['training_job_details']['training_datasets'] = []
    file['training_details']['training_job_details']['training_environment']['container_image'] = []
    file['training_details']['training_job_details']['hyper_parameters'] = []
    file['training_details']['training_job_details']['user_provided_training_metrics'] = []
    file['training_details']['training_job_details']['user_provided_hyper_parameters'] = []
    file = json.loads(str(file).replace("'", '"'))

    sm_client.create_model_card(
        ModelCardName='-'.join(event['detail']['ModelPackageName'].split('/')),
        Content=json.dumps(file),
        ModelCardStatus='PendingReview',
    )
