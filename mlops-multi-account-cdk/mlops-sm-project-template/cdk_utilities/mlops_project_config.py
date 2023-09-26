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
from dataclasses import dataclass, field
from pathlib import Path

from dataclasses_json import DataClassJsonMixin
from yamldataclassconfig.config import YamlDataClassConfig


@dataclass
class SeedCodeConfig(DataClassJsonMixin):
    app_type: str = field(default='')
    build_app_relative_path: str = field(default='')
    deploy_app_relative_path: str = field(default='')
    absolute_base_path: str = field(default='')


@dataclass
class MetadataConfig(DataClassJsonMixin):
    description: str = field(default='')
    product_name: str = field(default='')
    support_email: str = field(default='')
    support_url: str = field(default='')
    support_description: str = field(default='')


@dataclass
class MlopsProjectConfig(DataClassJsonMixin):
    metadata: MetadataConfig = field(default=None)
    seed_code: SeedCodeConfig = field(default=None)


@dataclass
class ProjectConfig(YamlDataClassConfig):
    mlops_project_config: MlopsProjectConfig = field(default=None)


# cnf: ProjectConfig = ProjectConfig()
# base_dir: str = os.path.abspath(f'{os.path.dirname(__file__)}{os.path.sep}..')
# cnf.load(Path(os.path.join(
#     base_dir,
#     'cdk_service_catalog',
#     'products',
#     'train_deploy_basic_product',
#     'mlops_project.yml'
# )))
#
# print(str(cnf.mlops_project_config))
