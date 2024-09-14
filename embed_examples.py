#!/usr/bin/env python
# # Copyright 2024 Google Inc.

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
from typing import Dict, List, Set
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

vertexai.init()
Embeddings = List[Dict[str, int|str]]


def get_examples(examples_file: str) -> Embeddings:
  with open(examples_file, "r") as f:
    data = json.load(f)

  ids = set()
  for record in data:
    assert 'id' in record, "Record has no ID: %r" % record
    assert record["id"] not in ids, "Record has a duplicate ID: %r" % record
    ids.add(record["id"])

  return data

def get_already_embedded(output_file: str) -> Embeddings:
  try:
    with open(output_file, "r") as f:
      return json.load(f)
  except (FileNotFoundError, json.decoder.JSONDecodeError):
    return []  # Not generated yet, generate from scratch here


def get_un_embedded_strings(examples_file: Embeddings, output_file: Embeddings) -> Set[str]:
  strings = set()

  for record in examples_file:
    strings.add(record["example"])
    strings.add(record["rewrite"])

  for record in output_file:
    if "example_embedding" in record and record["example"] in strings:
      strings.remove(record["example"])
    if "rewrite_embedding" in record and record["rewrite"] in strings:
      strings.remove(record["rewrite"])

  return strings


def embed_strings(strings: Set[str]) -> Dict[str, List[float]]:
  if len(strings) == 0:
    # Vertex AI doesn't support empty input.
    # In this case, we have nothing to input; return an empty result.
    return {}

  model = TextEmbeddingModel.from_pretrained("text-embedding-preview-0815")
  inputs = [TextEmbeddingInput(text, "SEMANTIC_SIMILARITY") for text in strings]
  embeddings = model.get_embeddings(inputs)
  return {
    input.text: embedding.values
    for input, embedding in zip(inputs, embeddings)
  }


def generate_missing_embeddings(
    example_file: str = "code_examples.json",
    output_file: str = "code_examples_embedded.json"):
  examples = get_examples(example_file)
  already_embedded = get_already_embedded(output_file)

  un_embedded_strings = get_un_embedded_strings(examples, already_embedded)

  embeddings = embed_strings(un_embedded_strings)

  examples_by_id = {x["id"]: x for x in examples}
  already_embedded_by_id = {x["id"]: x for x in already_embedded}

  for id in examples_by_id.keys():
    if id not in already_embedded_by_id:
      already_embedded_by_id[id] = {}
    already_embedded_by_id[id].update(examples_by_id[id])

  newly_embedded = [
    already_embedded_by_id[id]
    for id in sorted(already_embedded_by_id)
  ]

  for record in newly_embedded:
    if record["example"] in embeddings:
      record["example_embedding"] = embeddings[record["example"]]
    if record["rewrite"] in embeddings:
      record["rewrite_embedding"] = embeddings[record["rewrite"]]

  with open(output_file, "w") as f:
    json.dump(newly_embedded, f, indent=2)


if __name__ == "__main__":
  generate_missing_embeddings()
