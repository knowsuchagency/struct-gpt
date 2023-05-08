import json
from typing import Mapping
from unittest import mock

import pytest
from pydantic import Field, BaseModel, ValidationError

from struct_gpt.models import OpenAiBase, OpenAiMixin


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


def test_SentimentSchema_from_openai():
    examples = [
        {
            "input": "this library is neat!",
            "output": '{"sentiment": "1"}',
        },
        {
            "input": "don't touch that",
            "output": '{"sentiment": "-1"}',
        },
    ]
    sentiment = SentimentSchema.from_openai(content="I love pizza!", examples=examples)
    assert sentiment.sentiment == "1"


def test_SentimentAnalysis_from_openai():
    text = (
        "As president, I loved the beautiful scenery, but the long hike was exhausting."
    )
    sentiment_analysis = SentimentAnalysis.from_openai(
        text=text,
    )
    assert sentiment_analysis.narrator.lower() == "president"
    assert sentiment_analysis.overall_sentiment.sentiment in ["-1", "0", "1"]
    assert all(word in sentiment_analysis.words_to_sentiment for word in text.split())
    assert all(
        sentiment.sentiment in ["-1", "0", "1"]
        for sentiment in sentiment_analysis.words_to_sentiment.values()
    )


def test_from_openai_with_no_examples_or_kwargs():
    with pytest.raises(AssertionError):
        # call from_openai without providing examples or kwargs
        SentimentSchema.from_openai()


def test_no_docstring_raises_exception():
    with pytest.raises(AssertionError) as e:
        OpenAiBase.from_openai(content="I love pizza!")
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
        SentimentSchema.from_openai(
            content="I love pizza!", examples=incorrect_examples
        )


@mock.patch("openai.ChatCompletion.create")
def test_openai_api_failure_raises_exception(mock_chat_completion):
    # Arrange: Mock the OpenAI API to simulate a failure
    class OpenAiError(Exception):
        pass

    mock_chat_completion.side_effect = OpenAiError("API request failed")

    # Act and Assert: Ensure that the OpenAI error is propagated
    with pytest.raises(OpenAiError):
        SentimentSchema.from_openai(content="I love pizza!")


def test_from_openai_with_invalid_temperature_raises_exception():
    with pytest.raises(AssertionError):
        # temperature should be between 0 and 1
        SentimentSchema.from_openai(content="I love pizza!", temperature=-1)


def test_from_openai_with_invalid_retries_raises_exception():
    with pytest.raises(AssertionError):
        # retries should be a non-negative integer
        SentimentSchema.from_openai(content="I love pizza!", retries=-1)


def test_from_openai_with_invalid_json():
    class DummyModel(BaseModel, OpenAiMixin):
        """
        Example model for testing

        {content}
        """

        field1: str
        field2: int

    # Mock the OpenAI API to return a JSON that doesn't match the model
    with mock.patch("openai.ChatCompletion.create") as mock_from_openai:
        mock_from_openai.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "field1": "valid string",  # this is valid
                                "field2": "invalid string",  # this should be an int
                            }
                        )
                    }
                }
            ]
        }

        with pytest.raises(ValidationError):
            DummyModel.from_openai(content="")


class TestExamplesHandling:
    class TestSchema(OpenAiBase):
        """what is this person's first name? {full_name}"""

        first_name: str

    example_input = "Sam Stevens"

    def test_valid_stringified_json(self):
        examples = [{"input": self.example_input, "output": '{"first_name": "Sam"}'}]
        instance = self.TestSchema.from_openai(full_name="Alex Boss", examples=examples)
        assert instance.first_name == "Alex"

    def test_valid_dict(self):
        examples = [{"input": self.example_input, "output": {"first_name": "Sam"}}]
        instance = self.TestSchema.from_openai(full_name="Alex Boss", examples=examples)
        assert instance.first_name == "Alex"

    def test_non_apology(self):
        """
        Test that the model doesn't prefix failed attempts to return json with some version of 'I apologize...'
        """
        examples = [{"input": self.example_input, "output": '{"first_name": "Sam"}'}]
        self.TestSchema.from_openai(full_name=self.example_input, examples=examples)

    def test_valid_instance(self):
        examples = [
            {"input": self.example_input, "output": self.TestSchema(first_name="Sam")}
        ]
        instance = self.TestSchema.from_openai(full_name="Alex Boss", examples=examples)
        assert instance.first_name == "Alex"

    def test_invalid_stringified_json(self):
        examples = [{"input": self.example_input, "output": "not valid json"}]
        with pytest.raises(ValueError) as e:
            self.TestSchema.from_openai(examples=examples)
            assert (
                "output (not valid json) should be a json representation of TestSchema or an instance of it"
                in str(e.value)
            )
