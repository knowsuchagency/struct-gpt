import json
from typing import Mapping
from unittest import mock

import pytest
from pydantic import Field, BaseModel, ValidationError

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
    narrator: str

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

def test_incorrect_example_format_raises_exception():
    incorrect_examples = [
        {
            "input": "this library is neat!",
            # missing output key
        },
        {
            # missing input key
            "output": '{"sentiment": "1"}',
        },
    ]

    with pytest.raises(KeyError):
        SentimentSchema.create(content="I love pizza!", examples=incorrect_examples)

@mock.patch("openai.ChatCompletion.create")
def test_openai_api_failure_raises_exception(mock_chat_completion):
    # Arrange: Mock the OpenAI API to simulate a failure
    class OpenAiError(Exception):
        pass

    mock_chat_completion.side_effect = OpenAiError("API request failed")

    # Act and Assert: Ensure that the OpenAI error is propagated
    with pytest.raises(OpenAiError):
        SentimentSchema.create(content="I love pizza!", examples=EXAMPLES)

def test_create_with_invalid_temperature_raises_exception():
    with pytest.raises(AssertionError):
        # temperature should be between 0 and 1
        SentimentSchema.create(content="I love pizza!", examples=EXAMPLES, temperature=-1)


def test_create_with_invalid_retries_raises_exception():
    with pytest.raises(AssertionError):
        # retries should be a non-negative integer
        SentimentSchema.create(content="I love pizza!", examples=EXAMPLES, retries=-1)

def test_create_with_invalid_json():
    class DummyModel(BaseModel, OpenAiMixin):
        """
        Example model for testing

        {content}
        """
        field1: str
        field2: int

    # Mock the OpenAI API to return a JSON that doesn't match the model
    with mock.patch('openai.ChatCompletion.create') as mock_create:
        mock_create.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "field1": "valid string",  # this is valid
                            "field2": "invalid string",  # this should be an int
                        })
                    }
                }
            ]
        }

        with pytest.raises(ValidationError):
            DummyModel.create(content="")
