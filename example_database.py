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
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import vertexai
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

vertexai.init()

import pathlib
BASE_DIR = pathlib.Path(__file__).parent


class ExampleDb:

    @classmethod
    def CodeExampleDb(cls):
        return cls(examples_file=(BASE_DIR/"code_examples_embedded.json").resolve())

    @classmethod
    def ConceptExampleDb(cls):
        return cls(examples_file=(BASE_DIR/"concept_examples_embedded.json").resolve())

    def __init__(self, examples_file: str):
        self._examples_file = examples_file

        with open(examples_file) as f:
            data = json.load(f)

        for record in data:
            record["example_embedding"] = np.array(
                record["example_embedding"], dtype=np.float32
            )
            record["rewrite_embedding"] = np.array(
                record["rewrite_embedding"], dtype=np.float32
            )

        self._data = {record["id"]: record for record in data}

    def _embed_search_terms(self, search_terms: List[str]) -> np.ndarray:
        if not search_terms:
            # `model.get_embeddings()` doesn't handle empty input gracefully.
            # Correct behavior is to return empty output.
            return []

        model = TextEmbeddingModel.from_pretrained("text-embedding-preview-0815")
        inputs = [
            TextEmbeddingInput(search_term, "SEMANTIC_SIMILARITY")
            for search_term in search_terms
        ]
        embeddings = model.get_embeddings(inputs)
        return [
            np.array(embedding.values, dtype=np.float32) for embedding in embeddings
        ]

    def search(self,
               search_terms: str|List[str],
               distance: float = 0.25,
               top_k: int = 10
              ) -> Dict[int, float|str]:
      if not search_terms:
        # If no input, shirt-circuit and return no output.
        # (The code below assumes at least one input term to search for.)
        return {}

      target_similarity = 1 - distance

      if isinstance(search_terms, str):
          # Heuristic for breaking up big string blocks
          search_terms = [x for x in search_terms.split("\n\n") if x.strip()]

      search_embeddings = self._embed_search_terms(search_terms)

      results_filtered_list = []
      for record in self._data.values():
          example_embedding = record["example_embedding"]
          similarity = max(
              [
                  float(
                      cosine_similarity([search_embedding], [example_embedding])[0, 0]
                  )
                  for search_embedding in search_embeddings
              ]
          )
          if similarity >= target_similarity:
              results_filtered_list.append((similarity, record["id"]))

      results_topk = sorted(results_filtered_list, reverse=True)[:top_k]

      results = OrderedDict(
          [
              (
                  id_,
                  {
                      "distance": 1 - similarity,
                      "example": self._data[id_]["example"],
                      "rewrite": self._data[id_]["rewrite"],
                  },
              )
              for similarity, id_ in results_topk
          ]
      )

      return results


if __name__ == "__main__":
    # Do an example search
    from pprint import pprint

    pprint(
        ExampleDb().search(
            """\
      CREATE TABLE users (
        user_id INT AUTO_INCREMENT,
        name STRING,
        age INT,
        profile TEXT,
        test_value BOOL
    );
    """
        )
    )
