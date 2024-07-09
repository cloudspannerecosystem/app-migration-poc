# Copyright 2024 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

def list_files(directory: str, suffixes: List[str]) -> List[str]:
    """
    List all files in a directory recursively with specific suffixes.

    Args:
    directory (str): Path to the directory.
    suffixes (List[str]): List of suffixes to filter files.

    Returns:
    List[str]: A list of absolute paths of files with specified suffixes.
    """
    files_list = []
    for root, directories, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if any(file.endswith(suffix) for suffix in suffixes):
                files_list.append(file_path)
    return files_list