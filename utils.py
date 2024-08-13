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
import jinja2
from pathlib import Path

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

def replace_and_save_html(template_file, output_file, replacements):
        """
        Loads an HTML template, replaces placeholders, and saves the result.

        Args:
            template_file (str): Path to the HTML template file.
            output_file (str): Path to save the modified HTML file.
            replacements (dict): Key-value pairs of placeholders and their replacements.
        """

        try:
            template_path = Path(template_file)
            if not template_path.is_file():
                raise FileNotFoundError(f"Template file not found: {template_file}")

            template_loader = jinja2.FileSystemLoader(template_path.parent)
            template_env = jinja2.Environment(loader=template_loader)
            template = template_env.get_template(template_path.name)

            # Perform placeholder replacements
            output_html = template.render(replacements)

            # Save the modified HTML
            output_path = Path(output_file)
            with open(output_path, "w") as f:
                f.write(output_html)

            print(f"Modified HTML saved to: {output_path}")
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")