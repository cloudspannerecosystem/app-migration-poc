#!/usr/bin/env python

import pandas as pd
import sys
from typing import List
import vertexai
from vertexai.generative_models import GenerativeModel


vertexai.init()


COMPARE_PROMPT = """\
You are a code reviewer; you have been presented with the following two pieces of source code.  Please review the Generated code and rate it based on how similar it is to the Ground-Truth code.  Focus on functional similarities -- it's ok if the code uses different terminology to implement the same behavior.  Return just the score as a single word, nothing else.

Things to consider:
* If the code is handling IDs, it must handle IDs in the same way, otherwise it is different.

Compare the code on the following scale:
* EQUIVALENT
* MANY-SIMILARITIES
* SOME-SIMILARITIES
* DIFFERENT

Ground-Truth code:
```
{ground_truth_code}
```

Generated code:
```
{generated_code}
```
"""

VALID_COMPARE_OUTPUTS = {
  "EQUIVALENT",
  "MANY-SIMILARITIES",
  "SOME-SIMILARITIES",
  "DIFFERENT"
}


def compare_one(ground_truth: str, generated: str) -> str:
  model = GenerativeModel("gemini-1.5-flash-002")
  prompt = COMPARE_PROMPT.format(
    ground_truth_code=ground_truth, generated_code=generated)

  response = model.generate_content(prompt)
  text = response.text.strip()
  if text not in VALID_COMPARE_OUTPUTS:
    text = "ERROR"

  return text


def compare(sheet: pd.DataFrame) -> List[str]:
  return (
    compare_one(ground_truth, generated)
    for ground_truth, generated in zip(
      sheet["Spanner Code"], sheet["Predicted Spanner Code"])
  )


if __name__ == "__main__":
  filename = sys.argv[1]
  data = pd.read_csv(filename)
  for result in compare(data):
    print(result)
