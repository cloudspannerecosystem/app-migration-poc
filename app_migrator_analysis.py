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
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import asyncio
import aiofiles
import logging
import time
from textwrap import dedent
from example_database import ExampleDb
from collections import defaultdict

from logger_config import setup_logger

# Setup logger for this module
logger = setup_logger(__name__)

@dataclasses.dataclass(frozen=True)
class FileAnalysis:
    filename: str
    code_sample: str
    start_line: int
    end_line: int
    suggested_change: str
    description: str
    complexity: str
    notes: List[str]
    warnings: List[str]

@dataclasses.dataclass(frozen=True)
class FileMetadata:
    filename: str
    line_count: int

@dataclasses.dataclass(frozen=True)
class MethodSignatureChange:
    filename: str
    original_signature: str
    new_signature: str
    explanation: str

class MigrationSummarizer:
    def __init__(self, google_generative_ai_api_key: Optional[str] = None, gemini_version = "gemini-1.5-pro-001"):
        """
        Initializes the MigrationSummarizer with an optional Google Generative AI API key.
        Sets up the language model for generating migration suggestions.

        Args:
            google_generative_ai_api_key (Optional[str]): The API key for Google Generative AI.
        """
        if google_generative_ai_api_key is not None:
            os.environ["GOOGLE_API_KEY"] = google_generative_ai_api_key

        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        self._llm = VertexAI(model_name=gemini_version, safety_settings=safety_settings)

        self._code_example_db = ExampleDb.CodeExampleDb()
        self._context_example_db = ExampleDb.ConceptExampleDb()

    async def analyze_file(self, filepath: str, file_content: Optional[str] = None, method_changes: str = None) -> Tuple[List[FileAnalysis], List[MethodSignatureChange]]:
        """
        Analyzes a given file to determine necessary modifications for migrating from MySQL JDBC to Cloud Spanner JDBC.

        Args:
            filepath (str): The path to the file to be analyzed.
            file_content (Optional[str]): The content of the file, if already loaded.
            method_changes (str): Changes in public method signatures from dependent files.

        Returns:
            List[FileAnalysis]: A list of file modifications
            List[MethodSignatureChange]: A list of method signature changes
            FileMetadata: Metadata about the file being analyzed
        """
        try:
            async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
                file_content = await f.read()
        except UnicodeDecodeError as e:
            print ("Reading Fife Error: ", str(e))
            # Return empty lists on decode error
            return [], [], FileMetadata(filepath, None)

        file_metadata = FileMetadata(
            filename=filepath,
            line_count=file_content.count('\n'),
        )

        examples_prompt = ""
        relevant_records = self._code_example_db.search(file_content)
        if relevant_records:
            example_prompt = dedent(
                    """
                    Code like the following
                    ```
                    {example}
                    ```

                    can be rewritten as follows:

                    ```
                    {rewrite}
                    ```
                    """)
            examples = [
                example_prompt.format(**record)
                for record in relevant_records
            ]
            examples_prompt_template = dedent("""
            The following are examples of how to rewrite code for Spanner.

            {examples}
            """)
            examples_prompt = examples_prompt_template.format(examples=examples)

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
                "complexity": "<Complexity(eg: SIMPLE|MODERATE|COMPLEX)>",
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

            {examples_prompt}

            **Instructions:**
            1. All generated results values should be single-line strings. Please only suggest relevant changes and don't hallucinate.
            2. If the code contains any known Spanner anti-patterns, describe the problem and any known workarounds under "warnings".
            3. Feel free to write code in applicaiton layer if some functioanlity can't be supported in database layer.
            4. Make sure return JSON is correct, verify it so we don't get error parsing it.
            5. Strictly capture the larger code snippet to modify and provide their changes in one object and provide a cummilative desciption of change instead of providing it line by line.
            6. You need to migrte to Spanner JDBC and not Spanner Client Library.
            7. Consider the following factors when evaluating complexity: difficulty of implementation, level of technical expertise required, and the clarity of the requirements. 
                Classify the complexity as SIMPLE, MODERATE, COMPLEX


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

        response_parsed = await self.json_multishot(prompt, response, 3, filepath)

        if isinstance(response_parsed, dict):
            response_parsed = [response_parsed]

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
                    complexity=mod.get('complexity'),
                    notes=[x["rewrite"] for x in
                           self._context_example_db.search(mod.get('description')).values()],
                    warnings=mod.get('warnings', []),
                ))

            for method in methods:
                method_signatures.append(MethodSignatureChange(
                    filename=filepath,
                    original_signature=method.get('original_signature'),
                    new_signature=method.get('new_signature'),
                    explanation=method.get('explanation'),
                ))

        return file_analysis, method_signatures, file_metadata

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
                logger.info("No changes detected for item: %s", identifier)
                return {}  # Return an empty dictionary to indicate no changes

            if response_text.startswith('```json\n') and response_text.endswith('\n```'):
                response_text = response_text[8:-4]
                if response_text == '':
                    logger.info("No changes detected within JSON block for item: %s", identifier)
                    return {}

            try:
                result = json.loads(response_text)
                if isinstance(result, dict):
                    logger.debug("Successfully parsed JSON for item: %s", identifier)
                    return result
                else:
                    error = "The output is not in the desired format, top-level object is not a dictionary"
            except json.decoder.JSONDecodeError as e:
                error = str(e)

            logger.warning("JSON parsing error for item %s (attempt %d/%d): %s", identifier, i + 1, retries, error)
            print(f"Attempting to correct JSON parsing error for item {identifier} (attempt {i + 1}/{retries})")  # Print statement for visibility

            response = await self._llm.ainvoke(prompt_template.format(original_prompt, response_text, error))

        if response == '':
            logger.info("No changes detected after retries for item: %s", identifier)
            return {}

        logger.warning("Failed to parse JSON after retries for item: %s %s", identifier, response)
        logger.error("Failed to parse JSON after retries for item: %s ", identifier)
        print(f"Failed to parse JSON after retries for item {identifier}. Check the logs for details.")  # Print statement for critical failure

        return {}

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
            List[FileMetadata]: A list of procedurally-computed metadata about each file
        """

        start_time = time.time()

        # Initialize a JavaAnalyzer to analyze dependencies between files
        dependency_analyzer = JavaAnalyzer()

        # Get the execution order of files as a graph G and a list of lists representing dependency groups
        G, list_of_lists = dependency_analyzer.get_execution_order(directory)

        # Initialize a list to store the summaries of file analyses
        summaries: List[List[FileAnalysis]] = []

        # Metadata computed deterministically about each file
        files_metadata: List[FileMetadata] = []

        # Initialize a dictionary to store the results of analyzed files (dynamic programming cache)
        dp = {}

        logger.info("Project analysis started. Analyzing files in directory: %s", directory)
        logger.info("Total number of dependency groups: %s", len(list_of_lists))
        logger.info("Dependency group sizes: %s", [len(x) for x in list_of_lists])

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
            logger.debug("Analyzing file: %s", node)

            result = []

            for child in G.successors(node):
                result.append(dp[child]);

            result = [item for sublist in result for item in sublist]
            change_dicts = [change.__dict__ for change in result]
            json_string = json.dumps(change_dicts, indent=2)

            file_analysis, methods_changes, file_metadata = await self.analyze_file(filepath=node, method_changes=json_string)

            dp[node] = methods_changes

            logger.debug("File analysis completed for: %s", node)

            return file_analysis, file_metadata

        async def process_batch(batch):  # Process a batch of dependencies
            logger.info("Processing batch of %s files", len(batch))
            tasks = [process_dependency(bb) for bb in batch]

            # Run tasks concurrently and get results
            results = await asyncio.gather(*tasks)

            logger.info("Batch processing completed.")

            return results

        for dependencies in list_of_lists:
            logger.info('Processing dependency group with %s files', len(dependencies))
            group_start_time = time.time()
            # Batching logic (improved)
            for i in range(0, len(dependencies), batch_size):
                logger.info('Batch Start Index- %s', str(i))
                batch = dependencies[i:i + batch_size]

                # Schedule the batch for processing
                batch_results = await process_batch(batch)

                summaries.extend([x[0] for x in batch_results])  # Collect results from all batches
                files_metadata.extend([x[1] for x in batch_results])

            group_end_time = time.time()
            group_execution_time = group_end_time - group_start_time
            logger.info("Dependency group processing completed. Execution time: %s seconds", group_execution_time)

        end_time = time.time()
        total_execution_time = end_time - start_time

        logger.info("Project analysis completed. Total execution time: %s seconds", total_execution_time)

        return summaries, files_metadata

    async def summarize_report(self, summaries: List[List[FileAnalysis]], files_metadata: List[FileMetadata], output_file: str = 'spanner_migration_report.html'):
        """
        Generates a comprehensive HTML report summarizing the analysis of code changes for a Cloud Spanner migration.

        This method orchestrates the report generation process. It takes a list of file analyses, file metadata,
        and an optional output file name as input. It then performs the following steps:

        1. Organizes the file analyses by filename and complexity.
        2. Extracts relevant information from the file analyses.
        3. Asynchronously generates the report overview, tasks summary, and task descriptions.
        4. Compiles the report data, including overview, tasks, efforts, notes, and code complexity.
        5. Combines the report data with application data (file metadata and lines of code).
        6. Renders the final HTML report by populating a template with the collected data.

        Args:
            summaries: A list of lists, where each inner list contains `FileAnalysis` objects representing code 
                    changes for a specific file.
            files_metadata: A list of `FileMetadata` objects containing information about the analyzed files.
            output_file: The name of the output HTML file. Defaults to 'spanner_migration_report.html'.
        """
        file_analyses = defaultdict(lambda: defaultdict(list))
        for file_data in summaries:
            for analysis in file_data:
                file_analyses[analysis.filename][analysis.complexity].append(analysis)

        summaries = [item for sublist in summaries for item in sublist]
        summaries_dictionary_list = [change.__dict__ for change in summaries]
        summaries_dictionary_list = [{**item, 'id': i} for i, item in enumerate(summaries_dictionary_list)]

        report_data = {}
        
        async_io_results = await asyncio.gather(self.__summarize_report_overview(summaries_dictionary_list), 
                                       self.__summarize_report_tasks(summaries_dictionary_list))
        

        report_overview = async_io_results[0]

        logger.debug("Report Overview: %s", report_overview)
        report_tasks, tasks_by_efforts = async_io_results[1]

        code_lines_by_complexity = dict((k, sum(int(item['numberOfLines']) for item in report_tasks['tasks'] 
                                           if item['complexity'] == k)) for k in set(item['complexity'] for item in report_tasks['tasks'] ))

        logger.debug("Tasks by efforts: %s", tasks_by_efforts)


        report_data.update(report_overview)
        logger.info("%s", report_data)
        report_data['efforts'] = tasks_by_efforts
        report_data['notes'] = report_tasks['misc_efforts']

        logger.info("%s", report_data)
        report_tasks_response = await self.__summarize_report_tasks_description(summaries_dictionary_list, report_tasks['tasks'])


        report_data['tasks'] = report_tasks_response
        report_data['code_lines_by_complexity'] = code_lines_by_complexity

        app_data = {
            'files': files_metadata,
            'linesOfCode': sum(x.line_count for x in files_metadata),
        }

        logger.info(report_data)

        replace_and_save_html('result/migration_template.html', output_file, {
            'report_data': report_data,
            'app_data': app_data,
            'file_analyses': file_analyses,
        })

    async def __summarize_report_tasks_description(self, summaries_dictionary_list: List, report_tasks: List, batch_size: int = 10) -> Dict:
        """
        Generates detailed descriptions for each task in the migration report.

        This method iterates through the tasks identified in the `report_tasks` list and uses an LLM prompt
        to generate a detailed description for each task, including example code changes and affected files.
        It processes tasks in batches to improve efficiency.

        Args:
            summaries_dictionary_list: A list of dictionaries, where each dictionary represents a code change
                                    and contains information like code sample, suggested change, complexity,
                                    description, and line numbers.
            report_tasks: A list of tasks identified in the previous summarization step. Each task includes a category
                        and a list of indexes referring to relevant code changes in `summaries_dictionary_list`.
            batch_size: The number of tasks to process concurrently. Defaults to 10.

        Returns:
            A list of dictionaries, where each dictionary contains a detailed description of a task.
        """
        prompt_template = """
        You are a Cloud Spanner expert specializing in migrating applications from MySQL JDBC. I need your help generating a report section for a specific task in a database migration project.

        **Task Details:**
        * taskShortname: {category}

        **Code Changes:**
        JSON array of code changes relevant to this {category} is as follows, with details like `code_sample`, `suggested_change`, `complexity`, `description`, and `filename`:
        {code_changes}

        **Steps:**
        1. Analyze the code_changes w.r.t to the task. 
        2. Generate the output, make sure to use DELIMITER_CODE_START & DELIMITER_CODE_END with codeDiff.
        3. Check if count of DELIMITER_CODE_START and DELIMITER_CODE_END is equal. If not then go back to step2.

        **Important Considerations:**
        1. Use `DELIMITER_CODE_START` & `DELIMITER_CODE_END` for codeDiff.

        Please generate a report section for this task in the following JSON format:

        ```json
        {{
                "taskShortname": "{category}",
                "description": "[Developer friendly Description]",
                "affectedFiles": [ /* List of affected files extracted from the code changes. File Names should be relative with the project. */ ],
                "exampleCodeChanges": [
                {{
                "description": "[Brief change description]",
                "codeDiff": "DELIMITER_CODE_START@GIT_PATCH_FORMAT_IN_JSON_ESCAPED_STRING_WITH_PROPER_INDENTATION@DELIMITER_CODE_END", 
                "similarChangeLocations": [ /* List of files with similar changes. File Names should be relative with the project. Use it wisely
                and only if out of all the affected files, the changes is needed in few. */ ]
                }}
                // ... more examples if required
                ]
        }}
        ```
        """

        for item in report_tasks:
            item['code_changes'] = [summaries_dictionary_list[index] for index in item['indexes']]  # Replace with actual objects
            del item['indexes'] 

        final_results = []

        async def process_task(task):
            """Processes a single task and generates its description."""
            logger.info("Processing Task: %s", task['category'])
            prompt = prompt_template.format(category=task['category'], code_changes=task['code_changes'])
            response = await self._llm.ainvoke(prompt)
            response = await self.json_multishot(prompt, response, 4, 'summarize_report_task_' + task['category'])
            logger.info("Task processing completed")
            return response 

        for i in range(0, len(report_tasks), batch_size):
            batch = report_tasks[i:i + batch_size]
            tasks_future = [process_task(task) for task in batch]
            # Run tasks concurrently and get results
            results = await asyncio.gather(*tasks_future)
            final_results.extend(results)

        return final_results

    async def __summarize_report_overview(self, summaries_dictionary: List) -> Dict:
        """
        Generates an overview report summarizing the migration analysis from MySQL JDBC to Cloud Spanner JDBC.

        This method leverages an LLM prompt to analyze a list of code changes and generate a comprehensive 
        overview report. The report includes information like application details, source and target database assessment, 
        risk assessment, effort estimation, and code impact.

        Args:
            summaries_dictionary: A list of dictionaries, where each dictionary represents a code change 
                                and contains information like code sample, suggested change, complexity, 
                                description, and line numbers.

        Returns:
            A dictionary containing the overview report in the specified JSON format.
        """
        prompt = f"""
            You are a Cloud Spanner expert with deep experience migrating applications from MySQL JDBC. You are reviewing an analysis of changes
            required to transition an application from MySQL JDBC to Spanner JDBC, with a specific focus on the migration of complex data types
            and transaction handling.

            I have analyzed the necessary changes and represented them as a JSON array below. Each object in the array has the following properties:
            *  `code_sample`: The original code snippet.
            *  `suggested_change`: The proposed code modification for Spanner compatibility.
            *  `complexity`:  The estimated complexity of the change (SIMPLE, MODERATE, COMPLEX).
            *  `description`: A description of the change.
            *  `start_line`: Starting line number of the affected code w.r.t original code contains non executable section.
            *  `end_line`: Ending line number of the affected code w.r.t original code contains non executable section.
            
            **Here is the JSON array containing the code changes:**
            ```json
            {summaries_dictionary}
            ```

            Instructions:
            1. `codeDiff` should be a JSON-escaped string, ensuring proper handling of special characters.
            2. All generated values within the report should be single-line strings.
            3. Use backslashes to escape backticks, especially within code snippets.
            4. Generate the output in the following JSON structure:
            ```json
            {{
            "appName": "[App Name]",
            "sourceTargetAssessment": "[Source & Target Database Assessment details, one paragraph]",
            "riskAssessmentMitigation": "[Risk Assessment & Mitigation details]",
            "effortEstimation": "[Effort Estimation details]",
            "developerSummary": "[Concise paragraph about app purpose and functionality]",
            "appSize": "[Size in MB or GB]",
            "programmingLanguages": "[List of languages and percentages]",
            "currentDBMS": "[Current database system(s)]",
            "clientDrivers": "[List of client libraries/drivers]",
            "ormsToolkits": "[ORMs or toolkits found]",
            "additionalNotes": "[Any other observations]",
            "migrationComplexity": "[Brief Summary of migration complexity with impact]",
            "codeImpact": [
            /* List of affected files extracted from the code changes. File Names should be relative with the project. */ 
               'file1',
               'file2',
                // ... more files
            ],
            "numberOfLines": "[total number of lines affected in source code, you can use `start_line` & `end_line` to find it out]",
            }}
            ```
            """

        response = await self._llm.ainvoke(prompt)
        response = await self.json_multishot(prompt, response, 2, 'summarize_report_overview')

        return response
    
    async def __summarize_report_tasks(self, summaries_dictionary: List) -> Dict:
        """
        Categorizes and summarizes code changes required for migrating an application from MySQL JDBC to Cloud Spanner JDBC.

        This method uses an LLM prompt to analyze a list of code changes, group them into categories, 
        and provide a summary of the migration tasks.

        Args:
            summaries_dictionary: A list of dictionaries, where each dictionary represents a code change 
                                and contains information like code sample, suggested change, complexity, 
                                description, and line numbers.

        Returns:
            A tuple containing:
                - A dictionary containing the categorized tasks in the specified JSON format.
                - A dictionary containing the tasks grouped by effort.
        """
            
        prompt_template = """
            You are a Cloud Spanner expert specializing in migrating applications from MySQL JDBC. I need your help categorizing code changes required for a migration to Spanner JDBC.

            I have analyzed the necessary changes and represented them as a JSON array below. Each object in the array has the following properties:
            *  `id`: Id of the object.
            *  `code_sample`: The original code snippet.
            *  `suggested_change`: The proposed code modification for Spanner compatibility.
            *  `complexity`:  The estimated complexity of the change (SIMPLE, MODERATE, COMPLEX).
            *  `description`: A description of the change.
            *  `start_line`: Starting line number of the affected code w.r.t original code contains non executable section.
            *  `end_line`: Ending line number of the affected code w.r.t original code contains non executable section.

            To categorize these changes, please follow these steps:
            **Steps:**
            1. **Identify Categories:** Carefully examine the code changes in the JSON array, paying close attention to the `code_sample`, `suggested_change`, `complexity`, and `description` properties. Determine the relevant categories for this database migration. Feel free to create as many categories as you see fit, and split common categories into more specific subcategories if necessary. Categories should be unique. Some examples of potential categories include:
                * Data Type Conversion
                * Handling Auto Increment
                * Method Signature Changes because of change in DAO
                * Transaction Management
                * Dependency Management
                * Query Adaptation


            2. **Categorize Changes:**  Once you have identified the categories, iterate through the code changes again.
                For each change, ask yourself, "Is this change falling into this category?" and assign it to the most appropriate one by using field `id`.
                Ensure that each code change is assigned to only one category.

            3. **Verify Results: ** Iterate on categories and verify whether the mappings are coorect.. If not repeat the step 2. 
            
            **Here is the JSON array containing the code changes:**
            ```json
            {summaries_dictionary}
            ```

            **Important Considerations:**
            1. The array contains {records} items.
            2. Ensure that each code change is assigned to only one category.
            3. Make the categories are generic and not tied to business logic in the code.
            4. If some JSON objects in the input suggests no changes, then omit it from the output.
            5. Consider the following factors when evaluating effort of a task: number of places or components that require modification & time take to understand and make the change. Categorize the effort as 'Minor,' 'Moderate,' or 'Major' based on the extent of the required changes."
            6. Categories of task should be unique. 
            7. Consider the following factors when evaluating complexity of task: difficulty of implementation, level of technical expertise required, and the clarity of the requirements. Classify the complexity as SIMPLE, MODERATE, COMPLEX.

            Total number of records are {records}.

            **Output Format:**

            Please provide your categorization in the following JSON format:
            ```json
            {{
            "tasks": [
                {{
                "category": "[Task Category]",
                "description": "[Executive-friendly description]",
                "complexity": "[Complexity(eg: SIMPLE, MODERATE, COMPLEX)]",
                "effort": "[Task Effort Required: Minor | Moderate | Major]",
                "indexes": "[JSON objects ids from the original input JSON Array which are relevant for this category]",
                "numberOfLines": "[Number of Total Lines Affected in Source Code]",
                }},
                // ... more tasks
            ],
            "misc_efforts": [
                    "[List of miscellaneous efforts]"
                ]
            }}
            ```
        """

        prompt = prompt_template.format(summaries_dictionary=summaries_dictionary, records=len(summaries_dictionary))
        response = await self._llm.ainvoke(prompt)
        response = await self.json_multishot(prompt, response, 4, 'summarize_report_tasks')

        # Define a custom sort order for effort
        effort_order = {'Major': 1, 'Moderate': 2, 'Minor': 3}
        
        # Sort the data based on the effort field
        sorted_tasks = sorted(response['tasks'], key=lambda x: effort_order.get(x['effort'], 4))
        
        # Group the tasks by effort level
        grouped_data = {}
        for item in sorted_tasks:
            effort = item['effort']
            if effort not in grouped_data:
                grouped_data[effort] = []
            grouped_data[effort].append({
                'category': item['category'],
                'description': item['description'],
                'complexity': item['complexity'],
                'numberOfLines': item['numberOfLines']
            })

        response['tasks'] = sorted_tasks

        return response, grouped_data
        

