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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import dataclasses
import json
from dependency_analyzer.java_analyze import JavaAnalyzer
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_vertexai import VertexAI
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
    def __init__(self, google_generative_ai_api_key: Optional[str] = None, gemini_version = "gemini-1.5-flash-001"):
        """
        Initializes the MigrationSummarizer with an optional Google Generative AI API key.
        Sets up the language model for generating migration suggestions.

        Args:
            google_generative_ai_api_key (Optional[str]): The API key for Google Generative AI.
        """
        if google_generative_ai_api_key is not None:
            os.environ["GOOGLE_API_KEY"] = google_generative_ai_api_key
        self._llm = VertexAI(model_name=gemini_version)

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
            from MySQL JDBC to Spanner JDBC and not Java Client library. Review the following file and identify
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

            **Instructions:**
            1. All generated results values should be single-line strings. Please only suggest relevant changes and don't hallucinate.
            2. If the code contains any known Spanner anti-patterns, describe the problem and any known workarounds under "warnings".
            3. Feel free to write code in applicaiton layer if some functioanlity can't be supported in database layer.
            4. Make sure return JSON is correct, verify it so we don't get error parsing it.
            5. Strictly capture the larger code snippet to modify and provide their changes in one object and provide a cummilative desciption of change instead of providing it line by line.
            6. You need to migrte to Spanner JDBC and not Spanner Client Library.


            *****Older Schema****
            `````````````
            CREATE TABLE Account (
            AccountId INT NOT NULL AUTO_INCREMENT,
            CreationTimestamp DATETIME(6) NOT NULL DEFAULT NOW(6),
            AccountStatus INT NOT NULL,
            Balance NUMERIC(18,2) NOT NULL,
            PRIMARY KEY (AccountId)
            );

            CREATE TABLE TransactionHistory (
            EventId BIGINT NOT NULL AUTO_INCREMENT,
            AccountId INT NOT NULL,
            EventTimestamp DATETIME(6) NOT NULL DEFAULT NOW(6),
            IsCredit BOOL NOT NULL,
            Amount NUMERIC(18,2) NOT NULL,
            Description TEXT,
            PRIMARY KEY (EventId)
            );

            CREATE TABLE Customer (
            CustomerId INT NOT NULL AUTO_INCREMENT,
            Name TEXT NOT NULL,
            Address TEXT NOT NULL,
            PRIMARY KEY (CustomerId)
            );

            CREATE TABLE CustomerRole (
            CustomerId INT NOT NULL,
            RoleId INT NOT NULL AUTO_INCREMENT,
            Role TEXT NOT NULL,
            AccountId INT NOT NULL,
            CONSTRAINT FK_AccountCustomerRole FOREIGN KEY (AccountId)
                REFERENCES Account(AccountId),
            PRIMARY KEY (CustomerId, RoleId),
            KEY (RoleId)
            );

            CREATE INDEX CustomerRoleByAccount ON CustomerRole(AccountId, CustomerId);

            CREATE TABLE SampleApp (
            Id BIGINT NOT NULL AUTO_INCREMENT,
            PRIMARY KEY (Id)
            );

            `````````````

            *****New Schema with Spanner****
            `````````````
            CREATE TABLE Account (
            AccountId BYTES(16) NOT NULL,
            CreationTimestamp TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
            AccountStatus INT64 NOT NULL,
            Balance NUMERIC NOT NULL
            ) PRIMARY KEY (AccountId);

            CREATE TABLE TransactionHistory (
            AccountId BYTES(16) NOT NULL,
            EventTimestamp TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
            IsCredit BOOL NOT NULL,
            Amount NUMERIC NOT NULL,
            Description STRING(MAX)
            ) PRIMARY KEY (AccountId, EventTimestamp DESC),
            INTERLEAVE IN PARENT Account ON DELETE CASCADE;

            CREATE TABLE Customer (
            CustomerId BYTES(16) NOT NULL,
            Name STRING(MAX) NOT NULL,
            Address STRING(MAX) NOT NULL,
            ) PRIMARY KEY (CustomerId);

            CREATE TABLE CustomerRole (
            CustomerId BYTES(16) NOT NULL,
            RoleId BYTES(16) NOT NULL,
            Role STRING(MAX) NOT NULL,
            AccountId BYTES(16) NOT NULL,
            CONSTRAINT FK_AccountCustomerRole FOREIGN KEY (AccountId)
                REFERENCES Account(AccountId),
            ) PRIMARY KEY (CustomerId, RoleId),
            INTERLEAVE IN PARENT Customer ON DELETE CASCADE;

            CREATE INDEX CustomerRoleByAccount ON CustomerRole(AccountId, CustomerId);

            CREATE TABLE CloudSpannerSampleApp (
            Id INT64 NOT NULL
            ) PRIMARY KEY (Id)

            `````````````

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
        
        response_parsed = await self.json_multishot(prompt, response)
        res = response_parsed

        file_analysis = []
        method_signatures = []

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
    
    async def json_multishot(self, original_prompt: str, response: str, retries: int = 3, identifier = ''):
        """
        Attempts to correct and parse JSON responses from the language model multiple times if errors occur.

        Args:
            original_prompt (str): The original prompt sent to the language model.
            response (str): The initial response from the language model.
            retries (int): The number of retries for parsing the JSON response.

        Returns:
            Dict: A parsed JSON object
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
                logger.info ('No Changes for file: %s', identifier)
                return []
            if response_text.startswith('```json\n') and response_text.endswith('\n```'):
                response_text = response_text[8:-4]
                if response_text == '':
                    logger.info ('No Changes for file but json: %s', identifier)
                    return []
            try:
                result = json.loads(response_text)
                if isinstance(result, dict):
                    return result

                error = 'The output is not in the desired format, top level object is not dictionary'
            
            except json.decoder.JSONDecodeError as e:
                error = str(e)

            logger.warning ('Error: %s %s %s', error, identifier, response_text)
            response = await self._llm.ainvoke(prompt_template.format(original_prompt, response_text, error))
            response = response
        
        
        if response == '':
            logger.info ('No Changes for file last: %s', identifier)
            return []

        logger.error('Not Done: %s %s', identifier, response)
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

    async def summarize_report(self, summaries: List[List[FileAnalysis]], output_file: str = 'spanner_migration_report.html'):
        summaries = [item for sublist in summaries for item in sublist]
        change_dicts = [change.__dict__ for change in summaries]
        json_analysis = json.dumps(change_dicts, indent=2)
        
        prompt = f"""
        You are a Cloud Spanner expert with deep experience migrating applications from MySQl JDBC. You are reviewing an analysis of changes required to 
        transition an application from MySQL JDBC to Spanner JDBC.

        Please be focused towards migrating of complex data types and transactional handling.

        Analyze the following code changes and generate a migration report, you don't have to return the input data back in output:
        ```json
        {json_analysis}
        ```

        Instructions:
        1.Please make sure output is less than 5000 words.
        2. All generated results values should be single-line strings.
        3. Please use backslash for special charactes and not include unecessary whitespace character.
        4. Please use backslash for backticks. Extra careful with code snippets.
        5. Please generate the ouput in JSON format having following strucutre. 
        ```json
        {{
        "appName": "[App Name]",
        "sourceTargetAssessment": "[Source & Target Database Assessment details]",
        "riskAssessmentMitigation": "[Risk Assessment & Mitigation details]",
        "effortEstimation": "[Effort Estimation details]",
        "developerSummary": "[Concise paragraph about app purpose and functionality]",
        "appSize": "[Size in MB or GB]",
        "programmingLanguages": "[List of languages and percentages]",
        "currentDBMS": "[Current database system(s)]",
        "clientDrivers": "[List of client libraries/drivers]",
        "ormsToolkits": "[ORMs or toolkits found]",
        "additionalNotes": "[Any other observations]",
        "migrationComplexity": "[Summary of migration complexity]",
        "codeImpact": [
            "[List of files/directories needing modification]"
        ],
        "majorEfforts": [
            {{
            "category": "[Category]",
            "taskShortname": "[Task Shortname]",
            "description": "[Executive-friendly description]"
            }},
            // ... more major efforts
        ],
        "minorEfforts": [
            // ... similar structure as majorEfforts
        ],
        "notes": [
            "[List of smaller tasks]"
        ],
        "tasks": [
            {{
            "taskShortname": "[Task Shortname]",
            "description": "[Detailed task description]",
            "affectedFiles": [
                "[File 1]",
                "[File 2]",
                // ... more affected files
            ],
            "exampleCodeChanges": [
                {{
                "description": "[Brief change description]",
                "codeDiff": [JSON Escaped String with proper indentation]
                "similarChangeLocations": [
                    "[File 1: Line Number]",
                    "[File 2: Line Number]",
                    // ... more locations
                ]
                }}
                // ... more examples if required
            ]
            }},
            // ... more tasks
        ]
        }}
        ```
        """

        response = await self._llm.ainvoke(prompt)

        response = await self.json_multishot(prompt, response, 3)

        replace_and_save_html('result/migration_template.html', output_file, {'report_data': response})