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

from utils import list_files, replace_and_save_html # type: ignore
import os
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
) # type: ignore
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import dataclasses
import json
from dependency_analyzer.java_analyze import JavaAnalyzer
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiofiles
import logging
import time

# Define a custom log format
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"

# Create a logger instance
logger = logging.getLogger(__name__)

# Set the log level to capture all log messages
logger.setLevel(level=logging.DEBUG)

# Create a handler for console output
# Create a file handler (optional, logs to a file)
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(log_format, log_datefmt))

# Create a console handler (optional, logs to the console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format, log_datefmt))
console_handler.setLevel(logging.ERROR)  # On

# Add the console handler and file handler to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


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
        self._llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )

    async def analyze_file(self, filepath: str, file_content: Optional[str] = None, method_changes: str = None) -> Tuple[List[FileAnalysis], List[MethodSignatureChange]]:
        """
        Analyzes a given file to determine necessary modifications for migrating from MySQL JDBC to Cloud Spanner JDBC.

        Args:
            filepath (str): The path to the file to be analyzed.
            file_content (Optional[str]): The content of the file, if already loaded.
            method_changes (str): Changes in public method signatures from dependent files.

        Returns:
            Tuple[List[FileAnalysis], List[MethodSignatureChange]]: A list of file modifications and method signature changes.
        """
        try:
            async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
                file_content = await f.read()
        except UnicodeDecodeError as e:
            print ("Reading Fife Error: ", str(e))
            return [], []    # Return empty lists on decode error

        prompt = f"""
            You are a Cloud Spanner expert. You are working on migrating an application
            from PostgreSQL JDBC, Hibernate & Spring Data JPA to Spanner with PostgreSQL dialect with JDBC, Hibernate & Spring Data JPA. Review the following file and identify
            how it will need to be modified to work with Cloud Spanner. The code provided contains blank lines, comments, documentation, and other non-executable elements.
            You can refer function docs and comments to understand more about it and then suggest the chagnes. 

            Return your results in JSON dictionary format.  The JSON output should have three top-level keys:

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
        
        response = await self._llm.ainvoke(prompt)
        
        response_parsed = await self.json_multishot(prompt, response.content, filepath)

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
    
    async def json_multishot(self, original_prompt: str, response: str,  filepath: str, retries: int = 3) -> List[Dict[str, Union[str, List]]]:
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
            if response_text == '':
                logger.info ('No Changes for file: %s', filepath)
                return []
            if response_text.startswith('```json\n') and response_text.endswith('\n```'):
                response_text = response_text[8:-4]
                if response_text == '':
                    logger.info ('No Changes for file but json: %s', filepath)
                    return []
            try:
                result = json.loads(response_text)
                if isinstance(result, dict):
                    return [result]

                error = 'The output is not in the desired format, top level object is a list instead of dictionary'
            
            except json.decoder.JSONDecodeError as e:
                error = str(e)

            logger.warning ('Error: %s %s %s', error, filepath, response_text)
            response = await self._llm.ainvoke(prompt_template.format(original_prompt, response_text, error))
            response = response.content
        
        
        if response == '':
            logger.info ('No Changes for file last: %s', filepath)
            return []

        logger.error('Not Done: %s %s', filepath, response)
        return []

    async def analyze_project(self, directory: str, max_threads = 2, batch_size=100) -> List[List[FileAnalysis]]:
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
        
        # Initialize a JavaAnalyzer to analyze dependencies between files
        dependency_analyzer = JavaAnalyzer()
        
        # Get the execution order of files as a graph G and a list of lists representing dependency groups
        G, list_of_lists = dependency_analyzer.get_execution_order(directory)
        
        # Initialize a list to store the summaries of file analyses
        summaries: List[List[FileAnalysis]] = []
        
        # Initialize a dictionary to store the results of analyzed files (dynamic programming cache)
        dp = {}

        logger.info("Total number of groups for Project Java Files: %s", len(list_of_lists))
        logger.info("Groups Sizes: %s", [len(x) for x in list_of_lists])

        async def process_dependency(node):
            """
            Analyzes a file node, taking into account its dependencies.

            This function processes a given node (file) by first gathering results of all its child nodes (dependencies).
            It then analyzes the current node using the gathered results and stores the analysis results in the dp dictionary.

            Args:
                node: The file node to be analyzed.

            Returns:
                file_analysis: The result of the file analysis.
            """
            result = []

            for child in G.successors(node):
                result.append(dp[child]);
        
            result = [item for sublist in result for item in sublist]
            change_dicts = [change.__dict__ for change in result]
            json_string = json.dumps(change_dicts, indent=2)

            file_analysis, methods_changes = await self.analyze_file(filepath=node, method_changes=json_string)

            dp[node] = methods_changes

            return file_analysis

        async def process_batch(batch):  # Process a batch of dependencies
            tasks = [process_dependency(bb) for bb in batch] 
    
            # Run tasks concurrently and get results
            results = await asyncio.gather(*tasks)  

            return results

        for dependencies in list_of_lists:
            logger.info('Processing Group Length: %s', len(dependencies))
            start_time = time.time()
            # Batching logic (improved)
            for i in range(0, len(dependencies), batch_size):
                logger.info('Batch Start Index- %s', str(i))
                batch = dependencies[i:i + batch_size]

                # Schedule the batch for processing
                batch_results = await process_batch(batch)

                summaries.extend(batch_results)  # Collect results from all batches

            end_time = time.time()
            execution_time = end_time - start_time

            logger.info("Execution time: %s", execution_time)

        return summaries

    async def summarize_report(self, summaries: List[List[FileAnalysis]], output_file: str = 'result/spanner_migration_report.html'):
        """
        Summarizes the analysis report for the migration project into an HTML formatted report.

        Args:
            summaries (List[List[FileAnalysis]]): The list of file analyses to be summarized.

        Returns:
            str: The HTML formatted summary report.
        """
        template_path: str = 'result/migration_template.html'
        flattened_summaries = [item for sublist in summaries for item in sublist]
        change_dicts = [change.__dict__ for change in flattened_summaries]
        json_analysis = json.dumps(change_dicts, indent=2)

        prompt = f"""
        You are a Cloud Spanner expert with deep experience migrating applications from PostgreSQL. You are reviewing an analysis of changes required to 
        transition an application from PostgreSQL JDBC, Hibernate, and Spring Data JPA to Spanner with its PostgreSQL dialect.

        Please be focused towards migrating of complex data types and transactional handling.

        **Task:**
        1. Craft a concise introduction summarizing the overall migration effort.
        2. Analyze the following JSON data containing change details and identify the categories of issues.
        ```json
        {json_analysis}

        Please structure your response in the following JSON format, output should be sorted in the order of complexity from low to high.
        {{
            "introduction": "<Summary of the Changes>",
            "categories": [
                {{
                    "Category": "<Category of Changes>",
                    "Description": ["<human-readable description Point 1>", "<human-readable description Point 2>" ...],
                    "Complexity": "<Complexity of the Issue>" 
                }}
                // ... more categories
            ]
        }}
        """

        response = await self._llm.ainvoke(prompt)
        response = response.content

        data = json.loads(response[8:-4])

        changes = [dataclasses.asdict(x) for x in flattened_summaries]
        data['summaries'] = changes

        replace_and_save_html(template_path, output_file, data)