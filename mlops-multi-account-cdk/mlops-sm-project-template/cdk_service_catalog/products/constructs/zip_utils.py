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


from pathlib import Path
from zipfile import ZipFile
import os


def create_zip(local_path: [Path, str], outout_path: Path=Path(".zip_archives")):
    """
    Create a zip archive with the content of `local_path`

    :param local_path: The path to the directory to zip
    :param outout_path: The path to the output zip file The file name is created from
    the local path one
    """

    if isinstance(local_path, str):
        local_path = Path(local_path)

    outout_path.mkdir(exist_ok=True)
    local_sub_path = "_".join(str(local_path.absolute()).split(os.path.sep)[-4:])
    outout_path = outout_path / f"{local_sub_path}.zip"

    with ZipFile(outout_path, mode="w") as archive:
        [
            archive.write(k, arcname=f"{k.relative_to(local_path)}")
            for k in local_path.glob("**/*")
            if not f"{k.relative_to(local_path)}".startswith(("cdk.out"))
            if "__pycache__" not in f"{k.relative_to(local_path)}"
            if not f"{k.relative_to(local_path)}".endswith(".zip")
        ]
        # if (gitignore_path := local_path / ".gitignore").exists():
        #     archive.write(gitignore_path, arcname=".gitignore")

    # zip_size = Path(outout_path).stat().st_size / 10**6
    return f"{outout_path}"
