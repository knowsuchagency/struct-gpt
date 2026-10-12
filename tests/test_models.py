import json
import os
from types import SimpleNamespace
from typing import Mapping
from unittest import mock

import pytest
from pydantic import Field, BaseModel, ValidationError

from struct_gpt.models import OpenAiBase, OpenAiMixin

live = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY to call the OpenAI API",
)


def fake_client(*contents):
    """An object shaped like openai.OpenAI whose chat completions return `contents` in order."""
    client = mock.MagicMock()
    client.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        for content in contents
    ]
    return client


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


@live
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


@live
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


def test_openai_api_failure_raises_exception():
    # Arrange: Mock the OpenAI API to simulate a failure
    class OpenAiError(Exception):
        pass

    client = mock.MagicMock()
    client.chat.completions.create.side_effect = OpenAiError("API request failed")

    # Act and Assert: Ensure that the OpenAI error is propagated
    with pytest.raises(OpenAiError):
        SentimentSchema.from_openai(content="I love pizza!", client=client)
    assert client.chat.completions.create.call_count == 3


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
    content = json.dumps(
        {
            "field1": "valid string",  # this is valid
            "field2": "invalid string",  # this should be an int
        }
    )
    client = fake_client(content, content, content)

    with pytest.raises(ValidationError):
        DummyModel.from_openai(content="", client=client)


def test_json_mode_is_requested_by_default():
    client = fake_client('{"sentiment": "1"}')
    result = SentimentSchema.from_openai(content="I love pizza!", client=client)
    assert result.sentiment == "1"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0
    # JSON mode requires the word "json" somewhere in the messages
    assert "json" in kwargs["messages"][0]["content"]
    assert kwargs["messages"][-1] == {
        "role": "user",
        "content": "Determine the sentiment of the given text:\n\n    I love pizza!",
    }


def test_json_mode_can_be_disabled():
    client = fake_client('{"sentiment": "1"}')
    SentimentSchema.from_openai(
        content="I love pizza!", client=client, json_mode=False, model="gpt-4"
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs
    assert kwargs["model"] == "gpt-4"


def test_retries_after_bad_json_then_succeeds():
    client = fake_client("not json", '{"sentiment": "-1"}')
    result = SentimentSchema.from_openai(content="ugh", client=client)
    assert result.sentiment == "-1"
    assert client.chat.completions.create.call_count == 2
    retry_messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "not json"}
    assert retry_messages[-1]["role"] == "user"
    assert retry_messages[-1]["content"].startswith("JSONDecodeError")


def test_examples_are_sent_as_conversation():
    client = fake_client('{"sentiment": "1"}')
    examples = [
        {"input": "neat!", "output": '{"sentiment": "1"}'},
        {"input": "meh", "output": {"sentiment": "0"}},
        {"input": "awful", "output": SentimentSchema(sentiment="-1")},
    ]
    SentimentSchema.from_openai(content="hi", examples=examples, client=client)
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert [json.loads(m["content"]) for m in messages[2:7:2]] == [
        {"sentiment": "1"},
        {"sentiment": "0"},
        {"sentiment": "-1"},
    ]


def test_default_client_is_openai_client():
    with mock.patch("openai.OpenAI") as OpenAI:
        OpenAI.return_value = fake_client('{"sentiment": "0"}')
        assert SentimentSchema.from_openai(content="ok").sentiment == "0"
        OpenAI.assert_called_once_with()


class TestExamplesHandling:
    class TestSchema(OpenAiBase):
        """what is this person's first name? {full_name}"""

        first_name: str

    example_input = "Sam Stevens"

    @live
    def test_valid_stringified_json(self):
        examples = [{"input": self.example_input, "output": '{"first_name": "Sam"}'}]
        instance = self.TestSchema.from_openai(full_name="Alex Boss", examples=examples)
        assert instance.first_name == "Alex"

    @live
    def test_valid_dict(self):
        examples = [{"input": self.example_input, "output": {"first_name": "Sam"}}]
        instance = self.TestSchema.from_openai(full_name="Alex Boss", examples=examples)
        assert instance.first_name == "Alex"

    @live
    def test_non_apology(self):
        """
        Test that the model doesn't prefix failed attempts to return json with some version of 'I apologize...'
        """
        examples = [{"input": self.example_input, "output": '{"first_name": "Sam"}'}]
        self.TestSchema.from_openai(full_name=self.example_input, examples=examples)

    @live
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
