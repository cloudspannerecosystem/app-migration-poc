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

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import javalang
import jinja2

from logger_config import setup_logger

logger = setup_logger(__name__)


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


async def parse_json_with_retries(
    llm, original_prompt: str, response: str, retries: int = 3, identifier: str = ""
) -> Dict:
    """
    Attempts to correct and parse JSON responses from the language model multiple times if errors occur.

    Args:
        original_prompt (str): The original prompt sent to the language model.
        response (str): The initial response from the language model.
        retries (int): The number of retries for parsing the JSON response.
        identifier (str): An optional identifier for the item being processed (for logging).

    Returns:
        Dict: A parsed JSON object, or an empty dictionary if no changes were detected or parsing failed.
    """

    prompt_template = """
    The following generated JSON value failed to parse, it contained the following
    error. Please return corrected string as a valid JSON in the dictionary format. All strings should be
    single-line strings.

    The original prompt was:
    {}

    And the generated JSON is:

    ```json
    {}
    ```

    Error: `{}`
    """

    for i in range(retries):
        response_text = response.strip()
        error = ""

        if response_text == "":
            logger.info("No changes detected for item: %s", identifier)
            return {}

        if response_text.startswith("```json\n") and response_text.endswith("\n```"):
            response_text = response_text[8:-4]
            if response_text == "":
                logger.info(
                    "No changes detected within JSON block for item: %s", identifier
                )
                return {}

        try:
            result = json.loads(response_text)
            if isinstance(result, dict):
                logger.debug("Successfully parsed JSON for item: %s", identifier)
                return result
            else:
                error = "The output is not in the desired format, top-level object is not a dictionary"
                raise Exception(error)
        except Exception as e:
            logger.warning(
                "JSON parsing error for item %s (attempt %d/%d): %s",
                identifier,
                i + 1,
                retries,
                e,
            )
            print(
                f"Attempting to correct JSON parsing error for item {identifier} (attempt {i + 1}/{retries}): {e}",
                file=sys.stderr,
            )

        response = await llm.ainvoke(
            prompt_template.format(original_prompt, response_text, error)
        )

    if response == "":
        logger.info("No changes detected after retries for item: %s", identifier)
        return {}

    logger.error(
        "Failed to parse JSON after retries for item: %s. Response: %s",
        identifier,
        response,
    )
    print(
        f"Failed to parse JSON after retries for item {identifier}. Check the logs for details."
    )

    return {}


def preprocess_code(code: str) -> str:
    # Function to preprocess the code

    # Step 1: Unescape the escaped characters
    unescaped_code = code.replace(r"\"", '"').replace(r"\n", "\n")

    # Step 2: Add newlines after semicolons, open and close braces
    formatted_code = re.sub(
        r";", ";\n", unescaped_code
    )  # Add newlines after semicolons
    formatted_code = re.sub(
        r"{", "{\n", formatted_code
    )  # Add newlines after open brace '{'
    formatted_code = re.sub(
        r"}", "\n}\n", formatted_code
    )  # Add newlines after close brace '}'

    # Step 3: Clean up extra spaces and newlines
    formatted_code = re.sub(
        r"\n\s*\n", "\n", formatted_code
    )  # Remove excessive newlines
    formatted_code = formatted_code.strip()  # Trim leading and trailing whitespace

    # Step 4: Optional - Indentation (simple indentation logic)
    indent_level = 0
    lines = formatted_code.split("\n")
    indented_code = []
    for line in lines:
        stripped_line = line.strip()
        # Decrease indentation after closing brace
        if stripped_line.startswith("}"):
            indent_level -= 1
        indented_code.append("    " * indent_level + stripped_line)
        # Increase indentation after opening brace
        if stripped_line.endswith("{"):
            indent_level += 1

    return "\n".join(indented_code)


def is_dao_class(file_path: str, file_content: str) -> bool:
    """
    Analyzes the given Java file content and its path to determine if it represents a DAO class.

    Args:
        file_path (str): The path to the Java file.
        file_content (str): The content of the Java file.

    Returns:
        bool: True if the file likely represents a DAO class, False otherwise.
    """

    dao_keywords = ["Dao", "Repository", "Persistence", "Mapper"]
    dao_annotations = ["@Repository", "@Service", "@Component", "@Mapper"]
    dao_libraries = ["javax.persistence", "org.springframework.data", "org.hibernate"]
    jdbc_keywords = [
        "Connection",
        "Statement",
        "PreparedStatement",
        "ResultSet",
        "DriverManager",
    ]

    if not file_path.endswith(".java"):
        return False

    try:
        # Check if the file path contains "dao" directory (case-insensitive)
        if "dao" in file_path.lower():
            logger.info(f"Found 'dao' in file path: {file_path}")
            return True

        tree = javalang.parse.parse(file_content)
        for path, class_declaration in tree.filter(javalang.tree.ClassDeclaration):

            # Check class name, annotations, and superclasses
            if (
                any(
                    keyword.lower() in class_declaration.name.lower()
                    for keyword in dao_keywords
                )
                or any(
                    annotation.name.lower() in dao_annotations
                    for annotation in class_declaration.annotations
                )
                or (
                    class_declaration.extends is not None
                    and any(
                        keyword.lower() in str(base).lower()
                        for base in class_declaration.extends
                        for keyword in dao_keywords
                    )
                )
            ):
                return True

            # Check for database-related method calls
            for method in class_declaration.methods:
                for path, node in method.filter(javalang.tree.MethodInvocation):
                    if node.qualifier is not None and any(
                        library in node.qualifier for library in dao_libraries
                    ):
                        return True

                for path, node in method.filter(javalang.tree.MemberReference):
                    if any(
                        jdbc_keyword in node.member for jdbc_keyword in jdbc_keywords
                    ):
                        return True

                if method.throws is not None:
                    for exc in method.throws:
                        if "SQLException" in exc:
                            return True

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return False
