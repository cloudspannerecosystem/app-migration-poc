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

from utils import list_files # type: ignore
import os
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import dataclasses
import json
from dependency_analyzer.java_analyze import JavaAnalyzer
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclasses.dataclass(frozen=True)
class FileAnalysis:
    filename: str
    code_sample: str
    start_line: int
    end_line: int
    suggested_change: str
    description: str
    warnings: List[str]

@dataclasses.dataclass(frozen=True)
class MethodSignatureChange:
    filename: str
    original_signature: str
    new_signature: str
    explanation: str

class MigrationSummarizer:
    def __init__(self, google_generative_ai_api_key: Optional[str] = None):
        """
        Initializes the MigrationSummarizer with an optional Google Generative AI API key.
        Sets up the language model for generating migration suggestions.

        Args:
            google_generative_ai_api_key (Optional[str]): The API key for Google Generative AI.
        """
        if google_generative_ai_api_key is not None:
            os.environ["GOOGLE_API_KEY"] = google_generative_ai_api_key
        self._llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro")

    def analyze_file(self, filepath: str, file_content: Optional[str] = None, method_changes: str = None) -> Tuple[List[FileAnalysis], List[MethodSignatureChange]]:
        """
        Analyzes a given file to determine necessary modifications for migrating from MySQL JDBC to Cloud Spanner JDBC.

        Args:
            filepath (str): The path to the file to be analyzed.
            file_content (Optional[str]): The content of the file, if already loaded.
            method_changes (str): Changes in public method signatures from dependent files.

        Returns:
            Tuple[List[FileAnalysis], List[MethodSignatureChange]]: A list of file modifications and method signature changes.
        """
        if file_content is None:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except UnicodeDecodeError as e:
                return [], []

        prompt = f"""
            You are a Cloud Spanner expert. You are working on migrating an application
            from MySQL JDBC to Cloud Spanner JDBC. Review the following file and identify
            how it will need to be modified to work with Cloud Spanner. The code provided contains blank lines, comments, documentation, and other non-executable elements.
            You can refer function docs and comments to understand more about it and then suggest the chagnes. 

            Return your results in JSON format.  The JSON output should have three top-level keys:

            *   `file_modifications`: Contains a list of dictionaries, each detailing a code modification required for Cloud Spanner compatibility. Follow the format outlined below for each modification.
            *   `method_signature_changes`: Contains a list of dictionaries, each detailing a public method signature changes required for caller. This includes the original and modified signature, along with any relevant explanations. No need to include changes if it's just a change in parameters name.
            *   `general_warnings`: Contains a list of general warnings or considerations related to the migration, even if they don't require direct code changes.

            Make sure changes in `file_modifications` and `method_signature_changes` are consistent with each other.

            Ensure that line numbers in the file_modifications are accurate and correspond to the line numbers in the original code file, including comments, 
            blank lines, and other non-executable elements.

            **Format for `file_modifications`:**

            ```json
            [
            {{
                "code_sample": "<piece of code to update>",
                "start_line": <starting line number of the affected code w.r.t complete code contains non executable section>,
                "end_line": <ending line number of the affected code w.r.t complete code contains non executable section>,
                "suggested_change": "<example modification to the file>",
                "description": "<human-readable description of the required change>",
                "warnings": [
                "<thing to be aware of>",
                "<another thing to be aware of>",
                ...
                ]
            }},
            ...
            ]
            ```

            **Format for `method_signature_changes`:**
            ```json
            [
            {{
                "original_signature": "<original method signature>",
                "new_signature": "<modified method signature>",
                "explanation": "<description of why the change is needed and how to update the code>",
            }},
            ...
            ]
            ```

            All generated results values should be single-line strings. Please only suggest relevant changes and don't hallucinate.

            If the code contains any known Spanner anti-patterns, describe the problem and
            any known workarounds under "warnings".

            Please analyze the following file:
            `{filepath}`
            ```
            {file_content}
            ```

            The above source file is dependent on some files which also has following changes in the public method signature: 
            ```
            {method_changes}
            ```
            Please also consdier how the method changes in dependent files will impact the changes in this file.
            """
        
        response = self._llm.invoke(prompt).content
        response_parsed = self.json_multishot(prompt, response)

        file_analysis = []
        method_signatures = []

        for res in response_parsed:
            file_modifications = res.get('file_modifications', [])
            methods = res.get('method_signature_changes', [])

            for mod in file_modifications:
                file_analysis.append(FileAnalysis(
                    filename=filepath,
                    code_sample=mod.get('code_sample'),
                    start_line=int(mod.get('start_line', '-1')),
                    end_line=int(mod.get('end_line', '-1')),
                    suggested_change=mod.get('suggested_change'),
                    description=mod.get('description'),
                    warnings=mod.get('warnings', []),
                ))

            for method in methods:
                method_signatures.append(MethodSignatureChange(
                    filename=filepath,
                    original_signature=method.get('original_signature'),
                    new_signature=method.get('new_signature'),
                    explanation=method.get('explanation'),
                ))

        return file_analysis, method_signatures
    
    def json_multishot(self, original_prompt: str, response: str, retries: int = 10) -> List[Dict[str, Union[str, List]]]:
        """
        Attempts to correct and parse JSON responses from the language model multiple times if errors occur.

        Args:
            original_prompt (str): The original prompt sent to the language model.
            response (str): The initial response from the language model.
            retries (int): The number of retries for parsing the JSON response.

        Returns:
            List[Dict[str, Union[str, List]]]: A list of parsed JSON objects.
        """
        prompt_template = """
            The following generated JSON value failed to parse, it contained the following
            error. Please return corrected string as a valid JSON list. All strings should be
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

            if response_text.startswith('```json\n') and response_text.endswith('\n```'):
                response_text = response_text[8:-4]

            try:
                result = json.loads(response_text)
                if isinstance(result, dict):
                    return [result]
                return result
            except json.decoder.JSONDecodeError as e:
                error_message = str(e)
                response = self._llm.invoke(prompt_template.format(original_prompt, response_text, error_message)).content

        return []

    def analyze_project(self, directory: str) -> List[List[FileAnalysis]]:
        """
        Analyzes all files in a given project directory to determine necessary modifications for migrating from MySQL JDBC to Cloud Spanner JDBC.

        This method uses dynamic programming (DP) to optimize the analysis process by storing intermediate results of file analyses. 
        The project is represented as a dependency graph where nodes are files and edges indicate dependencies between files.
        The analysis is performed in a topological order of the dependency graph to ensure that dependent files are analyzed 
        after their dependencies have been analyzed.

        Args:
            directory (str): The root directory of the Java project.

        Returns:
            List[List[FileAnalysis]]: A list of file analyses, each containing necessary modifications for the migration.
        """
        # List all Java and XML files in the given directory
        files = list_files(directory, ['java', 'xml'])
        
        # Initialize a JavaAnalyzer to analyze dependencies between files
        dependency_analyzer = JavaAnalyzer()
        
        # Get the execution order of files as a graph G and a list of lists representing dependency groups
        G, list_of_lists = dependency_analyzer.get_execution_order(directory)
        
        # Initialize a list to store the summaries of file analyses
        summaries: List[List[FileAnalysis]] = []
        
        # Initialize a dictionary to store the results of analyzed files (dynamic programming cache)
        dp = {}
        
        # Maximum number of threads for concurrent processing
        max_threads = 10

        def process_dependency(node):
            """
            Analyzes a file node, taking into account its dependencies.

            This function processes a given node (file) by first gathering results of all its child nodes (dependencies).
            It then analyzes the current node using the gathered results and stores the analysis results in the dp dictionary.

            Args:
                node: The file node to be analyzed.

            Returns:
                file_analysis: The result of the file analysis.
            """
            results = []

            # Collect results from all child nodes (dependencies)
            for child in G.successors(node):
                results.extend(dp[child])

            # Convert the results to a JSON string for method changes
            result_dicts = [result.__dict__ for result in results]
            json_string = json.dumps(result_dicts, indent=2)
            
            # Analyze the current file using the gathered results of its dependencies
            file_analysis, method_changes = self.analyze_file(filepath=node, method_changes=json_string)
            
            # Store the analysis result in the dp dictionary
            dp[node] = method_changes

            return file_analysis

        # Process each list of dependencies concurrently using ThreadPoolExecutor
        for dependencies in list_of_lists:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                # Submit the process_dependency function for each dependency in the current list
                futures = {executor.submit(process_dependency, dep): dep for dep in dependencies}
                for future in as_completed(futures):
                    product = futures[future]
                    try:
                        # Get the result of the future and add it to the summaries
                        result = future.result()
                        summaries.append(result)
                    except Exception as e:
                        print(f"Error processing {product}: {e}")

        return summaries

    def summarize_report(self, summaries: List[List[FileAnalysis]]) -> str:
        """
        Summarizes the analysis report for the migration project into an HTML formatted report.

        Args:
            summaries (List[List[FileAnalysis]]): The list of file analyses to be summarized.

        Returns:
            str: The HTML formatted summary report.
        """
        flattened_summaries = [item for sublist in summaries for item in sublist]
        change_dicts = [change.__dict__ for change in flattened_summaries]
        json_analysis = json.dumps(change_dicts, indent=2)

        prompt = f"""
        You are a Cloud Spanner expert. You are working on migrating an application from MySQL JDBC to Cloud Spanner JDBC.
        You see the following analysis of the changes needed to be updated and where. Write a report which is easy to read and describes the major categories
        of issues identified, and how complex each category of issue is in terms of whether it requires just basic coding knowledge,
        deep architectural experience, or something in between.

        Please write the report in HTML format, with formatted section headers. The report should detail what to change, where to change, why to change,
        and how to change, with exact line numbers. Include tables to present the data clearly. 

        Use this structure:
        1. Title: Cloud Spanner Migration Report
        2. Summary: Overview of the report's purpose.
        3. Classification of differnt categories of chagnes & complexity associated with it.
        4. Detailed Analysis for Each File:
            - File Name
            - Table with columns: Start Line, End Line, Code Sample, Suggested Change, Description, Warnings

        
        Plese use code highlighting, different colors & styling to make it easy to read.

        Analysis:
        ```json
        {json_analysis}
        """

        response = self._llm.invoke(prompt).content

        return response