# struct-gpt

structured llm outputs

## Usage

Template variables in the class' docstring are replaced with the keyword arguments passed to `create`.

```python
from struct_gpt import OpenAiBase
from pydantic import Field

class SentimentSchema(OpenAiBase):
    """
    Determine the sentiment of the given text:

    {content}
    """

    sentiment: str = Field(description="Either -1, 0, or 1.")

print(SentimentSchema.create(content="I love pizza!").json())
```
outputs:
```json
{
  "sentiment": "1"
}
```

Your classes can reference one another. You can also use the `OpenAiMixin` to add functionality to your own classes.

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

# you can use the OpenAiMixin to add functionality to your own classes
class SentimentAnalysis(BaseModel, OpenAiMixin):
    """
    Determine the sentiment of each word in the following: {text}

    Also determine the overall sentiment of the text and who the narrator is.
    """

    words_to_sentiment: Mapping[str, SentimentSchema]
    overall_sentiment: SentimentSchema
    narrator: str = Field(description="The narrator of the text.")


print(
    SentimentAnalysis.create(
        text="As president, I loved the beautiful scenery, but the long hike was exhausting."
    ).json(indent=2)
)
```
outputs:
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
