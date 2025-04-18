# Copyright 2024 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

import os
import pathlib

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

# ToDo:Create a module
BASE_DIR = pathlib.Path(__file__).parent
# Set PYTHONPATH to include the *parent* of our containing dir
sys.path.append(os.path.abspath(BASE_DIR.parent.resolve()))
import argparse
import asyncio
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import pandas as pd
import torch
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, VertexAI
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaModel, RobertaTokenizer

from app_migrator_analysis import MigrationSummarizer
from logger_config import setup_logger
from utils import parse_json_with_retries, preprocess_code  # type: ignore

logger = setup_logger(__name__)


class CodeDescriptionEvaluator:
    def __init__(self):
        # Load UniXcoder for code and BERT for text
        self.code_tokenizer = RobertaTokenizer.from_pretrained(
            "microsoft/unixcoder-base"
        )
        self.code_model = RobertaModel.from_pretrained("microsoft/unixcoder-base")

        self.desc_tokenizer = RobertaTokenizer.from_pretrained(
            "microsoft/unixcoder-base"
        )
        self.desc_model = RobertaModel.from_pretrained("microsoft/unixcoder-base")

    def get_code_embedding(self, code_snippet):
        """Generate embeddings for code snippets."""
        inputs = self.code_tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=True,
        )
        with torch.no_grad():
            outputs = self.code_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    def get_desc_embedding(self, description):
        """Generate embeddings for descriptions."""
        inputs = self.desc_tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=True,
        )
        with torch.no_grad():
            outputs = self.desc_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    def compare_embeddings(self, embedding1, embedding2):
        """Compare embeddings using cosine similarity."""
        similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
        return similarity_score


@dataclass(frozen=True)
class TestCase:
    functionality: str
    source_code: str
    mysql_schema: str
    spanner_schema: str
    ground_truth_solution: str
    description: str

    def to_json(self, exclude_keys=None):
        """
        Converts the TestCase instance to a JSON string, optionally excluding specified keys.

        Args:
            exclude_keys (list): A list of keys to exclude from the JSON representation.

        Returns:
            str: A JSON string representation of the TestCase.
        """
        if exclude_keys is None:
            exclude_keys = []

        data = asdict(self)
        for key in exclude_keys:
            if key in data:
                del data[key]
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)


class AccuracyEvaluator:
    def __init__(
        self,
        google_generative_ai_api_key: Optional[str] = None,
        gemini_version="gemini-1.5-pro-001",
    ):
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

        # ToDo: Use RAG Powered evaluator from app migration analysis
        self.migration_summarizer = MigrationSummarizer(google_generative_ai_api_key)
        self._code_and_description_evaluator = CodeDescriptionEvaluator()

    async def convert_code(self, eval_data_point: TestCase) -> Dict:
        logger.info("Evaluating Test: %s", eval_data_point.functionality)

        input_json = eval_data_point.to_json(
            ["functionality", "ground_truth_solution", "description"]
        )

        prompt = f"""
            You are a Cloud Spanner expert tasked with migrating a code snippet from MySQL JDBC to Spanner JDBC.

            Given the `source_code`, old `mysql_schema`, and new `spanner_schema` in the following JSON:
            ```json
            {input_json}
            ```

            **Steps:**
            1. **Understand Context:** Analyze the `mysql_schema` and `source_code` to see how the code interacts with the MySQL schema.
            2. **Understand New Schema:**  Examine the `spanner_schema` and understand how the `mysql_schema` has been transformed.
            3. **Code Changes:**  Use your Spanner knowledge, the schema changes, and the application context to convert the `source_code` to work with Spanner JDBC.

            **Important Considerations:**
            Modify the code to work with Cloud Spanner using JDBC. If a direct equivalent is not available, implement the logic in the application layer.

            **Output Format:**
            ```json
            {{
                "converted_code": "The migrated code snippet (single-line string)",
                "description": "Explanation for the changes (single-line string)"
            }}
            ```
            """

        response = await self.migration_summarizer.migration_code_conversion_invoke(
            prompt,
            eval_data_point.source_code,
            eval_data_point.mysql_schema,
            eval_data_point.spanner_schema,
            "evaluation_code_accuracy",
        )

        logger.info("Evaluated Test: %s", eval_data_point.functionality)
        return response

    async def evaluate_accuracy(
        self, test_file: str = (BASE_DIR / "test_dataset.json").resolve(), batch_size=3
    ):
        logger.info("Starting generating recommendations...")
        start_time = time.time()

        test_cases: List[TestCase] = AccuracyEvaluator.load_test_cases_from_file(
            test_file
        )

        final_results = []
        for i in range(0, len(test_cases), batch_size):
            group_start_time = time.time()

            logger.info("Test Case batch starting with index %s", str(i))
            batch = test_cases[i : i + batch_size]

            tasks_future = [self.convert_code(test_case) for test_case in batch]
            # Run tasks concurrently and get results
            results = await asyncio.gather(*tasks_future)

            final_results.extend(results)

            group_end_time = time.time()
            group_execution_time = group_end_time - group_start_time
            logger.info(
                "Test Case batch started with index %s converted. Execution time: %s seconds",
                str(i),
                group_execution_time,
            )

        end_time = time.time()
        total_execution_time = end_time - start_time

        logger.info(
            "Gemini generated recommendations for all test cases. Total execution time: %s seconds",
            total_execution_time,
        )

        return self.calculate_accuracy(test_cases, final_results)

    def calculate_accuracy(
        self,
        test_cases_list,
        generated_list,
        code_threshold=0.8,
        description_threshold=0.6,
    ):
        """Evaluate the accuracy of generated descriptions and codes against ground truth."""
        correct_count = 0
        total_count = len(test_cases_list)

        data = []
        for test_case, generated in zip(test_cases_list, generated_list):
            (
                functionality,
                source_code,
                mysql_schema,
                spanner_schema,
                gt_code,
                gt_desc,
            ) = (
                test_case.functionality,
                test_case.source_code,
                test_case.mysql_schema,
                test_case.spanner_schema,
                test_case.ground_truth_solution,
                test_case.description,
            )

            gen_code, gen_desc = generated["converted_code"], generated["description"]

            # Generate embeddings and compare for code
            code_embedding_gt = self._code_and_description_evaluator.get_code_embedding(
                preprocess_code(gt_code)
            )
            code_embedding_gen = (
                self._code_and_description_evaluator.get_code_embedding(
                    preprocess_code(gen_code)
                )
            )

            logger.info(preprocess_code(gt_code))
            logger.info(preprocess_code(gen_code))
            code_similarity = self._code_and_description_evaluator.compare_embeddings(
                code_embedding_gen, code_embedding_gt
            )

            # Generate embeddings and compare for description
            desc_embedding_gt = self._code_and_description_evaluator.get_desc_embedding(
                gt_desc
            )
            desc_embedding_gen = (
                self._code_and_description_evaluator.get_desc_embedding(gen_desc)
            )
            logger.info(preprocess_code(gt_desc))
            logger.info(preprocess_code(gen_desc))
            desc_similarity = self._code_and_description_evaluator.compare_embeddings(
                desc_embedding_gen, desc_embedding_gt
            )

            logger.info(
                f"Test: {test_case.functionality} Code Similarity: {code_similarity:.4f} | Description Similarity: {desc_similarity:.4f}"
            )

            # Check if both code and description similarities exceed the threshold
            if (
                code_similarity >= code_threshold
                and desc_similarity >= description_threshold
            ):
                correct_count += 1

            data.append(
                (
                    functionality,
                    source_code,
                    mysql_schema,
                    spanner_schema,
                    gt_code,
                    gen_code,
                    code_similarity,
                    gt_desc,
                    gen_desc,
                    desc_similarity,
                )
            )

        # Calculate accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0

        # Convert list of tuples to DataFrame
        data_frame = pd.DataFrame(
            data,
            columns=[
                "Functionality",
                "Source Code",
                "My SQL Schema",
                "Spanner Schema",
                "Spanner Code",
                "Predicted Spanner Code",
                "Similarity Score",
                "Description",
                "Predicted Description",
                "Similarity Score",
            ],
        )

        data_frame.to_csv("output.csv", index=False)

        return accuracy

    def load_test_cases_from_file(filepath: str) -> List[TestCase]:
        """
        Loads test cases from a JSON file.

        Args:
            filepath (str): The path to the JSON file containing the test cases.

        Returns:
            List[TestCase]: A list of TestCase objects.
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                return [TestCase(**test_case) for test_case in data]
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {filepath}")
            return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--github-benchmark-data-file",
        help="Where to write GitHub-formatted results data",
    )
    args = parser.parse_args()

    eval = AccuracyEvaluator("api_key")
    result = asyncio.run(eval.evaluate_accuracy())

    if args.github_benchmark_data_file:
        with open(args.github_benchmark_data_file, "w") as output_file:
            json.dump(
                [
                    {
                        "name": "Overall Eval Result",
                        "unit": "Accuracy (percent)",
                        "value": result * 100,
                        # "range": "3",  # Variance (not currently computed)
                        # "extra": "[optional tooltip]"
                    },
                ],
                output_file,
            )
    print("Accuracy: {}%".format(result * 100))
