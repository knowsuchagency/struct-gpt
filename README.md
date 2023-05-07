# struct-gpt

structured llm outputs

## Usage

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

```python
class SentimentAnalysis(OpenAiBase):
    """
    Determine the sentiment of each word in the following: {text}
    """

    sentiment: Mapping[str, SentimentSchema]


analysis = SentimentAnalysis.create(
    text="I love the beautiful scenery, but the long hike was exhausting."
)
print(analysis.json(indent=2))
```
outputs:
```json
{
  "sentiment": {
    "I": {
      "sentiment": "1"
    },
    "love": {
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
      "sentiment": "0"
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
  }
}
```
