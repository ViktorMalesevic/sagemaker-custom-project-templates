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


import logging
from logging import Logger
import importlib
import os
import pathlib
import inspect
from typing import List, Any


class ClassUtilities:
    logging.basicConfig(level=logging.INFO)
    logger: Logger = logging.getLogger('ClassUtilities')

    @classmethod
    def find_subclasses(cls, base_class: Any, base_package: str, exclude_packages: List[str]) -> List[Any]:

        sub_classes: List[Any] = list()
        base_package_module = importlib.import_module(name=base_package)
        base_package_module_path: str = base_package_module.__path__[0]

        for sub_module_dir in filter(
                lambda d: d not in exclude_packages and not (d.startswith('__') and d.endswith('__')) and os.path.isdir(
                    f'{base_package_module_path}{os.path.sep}{d}')
                ,
                os.listdir(base_package_module_path)
        ):

            sub_module_dir_path: pathlib.Path = pathlib.Path(base_package_module_path, sub_module_dir)
            for filepath in filter(
                    lambda x:
                    x.name != '__init__.py' and
                    x.name != 'setup.py' and
                    (len([ep for ep in exclude_packages if f'/{ep}/' in str(x)]) == 0),
                    sub_module_dir_path.glob(f'**{os.path.sep}*.py')
            ):

                module_name: str = filepath.stem
                sub_module_package = f'.{sub_module_dir}{".".join(str(filepath.parent).split(sub_module_dir)[1:]).replace(os.path.sep, ".")}.{module_name}'
                module = importlib.import_module(name=sub_module_package, package=base_package_module.__package__)
                cls.logger.debug(
                    f'sub_module_package : {sub_module_package}, base_package_module : {base_package_module.__package__}')
                for attr in filter(
                        lambda x: not (x.startswith('__') and x.endswith('__')), dir(module)):

                    clss = getattr(module, attr)

                    if inspect.isclass(clss) and \
                            issubclass(clss, base_class) \
                            and clss != base_class and \
                            (len([ep for ep in exclude_packages if f'.{ep}.' in str(clss)]) == 0):
                        sub_classes.append(clss)
                        cls.logger.debug(f'Class found : {str(clss)}')
        return sub_classes
