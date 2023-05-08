# struct-gpt

[![codecov](https://codecov.io/gh/knowsuchagency/struct-gpt/branch/main/graph/badge.svg?token=TMUQNTCTDI)](https://codecov.io/gh/knowsuchagency/struct-gpt)
![PyPI](https://img.shields.io/pypi/v/struct-gpt)


Get structured output from LLM's.

```python
from struct_gpt import OpenAiBase
from pydantic import Field


class SentimentSchema(OpenAiBase):
    """
    Determine the sentiment of the given text:

    {content}
    """
    # 👆this becomes the prompt

    sentiment: str = Field(description="Either -1, 0, or 1.")


result = SentimentSchema.from_openai(content="I love pizza!").json()
                                     # 👆this ends up in the prompt
print(result)
```
outputs:
```json
{
  "sentiment": "1"
}
```

## Features

* Easy creation of custom models using the OpenAI API
* Integration with Pydantic for model validation and serialization
* Flexible configuration with retries and temperature settings

## Installation

`pip install struct-gpt`


## Usage

* Template variables in the class' docstring are replaced with the keyword arguments passed to `from_openai`
* Classes can reference one another
* You can use the `OpenAiMixin` to add functionality to existing Pydantic classes

```python
from struct_gpt import OpenAiBase, OpenAiMixin
from pydantic import Field, BaseModel
from typing import Mapping


class SentimentSchema(OpenAiBase):
    """
    Determine the sentiment of the given text:

    {content}
    """

    sentiment: str = Field(description="Either -1, 0, or 1.")


class SentimentAnalysis(BaseModel, OpenAiMixin):
    """
    Determine the sentiment of each word in the following: {text}

    Also determine the overall sentiment of the text and who the narrator is.
    """

    words_to_sentiment: Mapping[str, SentimentSchema]
    overall_sentiment: SentimentSchema
    narrator: str


print(
    SentimentAnalysis.from_openai(
        text="As president, I loved the beautiful scenery, but the long hike was exhausting."
    ).json(indent=2)
)
```
<details>
<summary><b>See Output</b></summary>

```json
{
  "words_to_sentiment": {
    "As": {
      "sentiment": "0"
    },
    "president,": {
      "sentiment": "1"
    },
    "I": {
      "sentiment": "0"
    },
    "loved": {
      "sentiment": "1"
    },
    "the": {
      "sentiment": "0"
    },
    "beautiful": {
      "sentiment": "1"
    },
    "scenery,": {
      "sentiment": "1"
    },
    "but": {
      "sentiment": "-1"
    },
    "long": {
      "sentiment": "-1"
    },
    "hike": {
      "sentiment": "-1"
    },
    "was": {
      "sentiment": "0"
    },
    "exhausting.": {
      "sentiment": "-1"
    }
  },
  "overall_sentiment": {
    "sentiment": "0"
  },
  "narrator": "president"
}
```

</details>

### Improving reliability with examples

`create` can accept a list of examples to guide the model and improve its accuracy. 

Each example is a dictionary containing an `input` and `output` key. 

`output` should be an instance of the model, a dictionary, or its json string representation,.

```python
from struct_gpt import OpenAiBase
from pydantic import Field


class SentimentSchema(OpenAiBase):
    """
    Determine the sentiment of the given text:

    {content}
    """

    sentiment: str = Field(description="Either -1, 0, or 1.")


examples = [
    {
        "input": "I love the beach!",
        "output": {"sentiment": "1"},
    },
    {
        "input": "don't touch that",
        "output": '{"sentiment": "-1"}',
    },
    {
        "input": "this library is neat!",
        "output": SentimentSchema(sentiment="1"),
    },
]

print(SentimentSchema.from_openai(content="I love pizza!", examples=examples).json())
```
outputs:
```json
{
  "sentiment": "1"
}
```
