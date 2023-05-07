from typing import Mapping

import pytest
from pydantic import Field, BaseModel

from struct_gpt.models import OpenAiBase, OpenAiMixin

# Test data
EXAMPLES = [
    {
        "input": "this library is neat!",
        "output": '{"sentiment": "1"}',
    },
    {
        "input": "don't touch that",
        "output": '{"sentiment": "-1"}',
    },
]

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
    narrator: str = Field(description="The narrator of the text.")

def test_SentimentSchema_create():
    sentiment = SentimentSchema.create(content="I love pizza!", examples=EXAMPLES)
    assert sentiment.sentiment == "1"

def test_SentimentAnalysis_create():
    sentiment_analysis = SentimentAnalysis.create(
        text="As president, I loved the beautiful scenery, but the long hike was exhausting.",
        examples=EXAMPLES
    )
    assert sentiment_analysis.narrator.lower() == "president"
    assert sentiment_analysis.overall_sentiment.sentiment in ["-1", "0", "1"]
    assert all(word in sentiment_analysis.words_to_sentiment for word in ["As", "president,", "I", "loved", "the", "beautiful", "scenery,", "but", "long", "hike", "was", "exhausting."])


def test_create_with_no_examples_or_kwargs():
    with pytest.raises(AssertionError):
        # call create without providing examples or kwargs
        SentimentSchema.create()

def test_no_docstring_raises_exception():
    with pytest.raises(AssertionError) as e:
        OpenAiBase.create(content="I love pizza!", examples=EXAMPLES)
        assert "please add a docstring" in str(e)
