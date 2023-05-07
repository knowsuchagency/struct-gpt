import io
import json
import logging
from typing import TypeVar, Literal, Mapping

import openai
from pydantic import BaseModel, Field, ValidationError
from ruamel.yaml import YAML

Model = TypeVar("Model", bound=BaseModel)
Example = Mapping[Literal["input", "output"], str]

yaml = YAML(typ="safe")

logger = logging.getLogger(__name__)


class OpenAiMixin:
    @classmethod
    def create(
        cls: Model,
        temperature=0,
        model="gpt-3.5-turbo",
        retries=2,
        examples: list[Example] = None,
        **kwargs,
    ) -> Model:
        """
        Create a new model instance using OpenAI's API.

        Args:
            temperature: Controls the randomness of the response.
            model: The specific OpenAI model to use.
            retries: Number of times to retry in case of failure.
            examples: List of input-output example pairs. Useful to improve the model's accuracy.
            **kwargs: Used to format the model's docstring.

        Returns:
            Model instance.
        """
        assert (
            doc := cls.__doc__.strip()
        ), "please add a docstring explaining how to destructure the prompt"

        assert (
            examples or kwargs
        ), "please provide either examples or keyword args for the docstring"

        # Convert the JSON schema to YAML since it takes up fewer tokens
        with io.StringIO() as fp:
            json_schema = json.loads(cls.schema_json())
            yaml.dump(json_schema, fp)
            yaml_json_schema = fp.getvalue()

        # Set up the system message to guide the assistant
        system_message = {
            "role": "system",
            "content": f"Please respond ONLY with valid json that conforms to this json_schema: {yaml_json_schema}. Don't include additional text other than the object as it will be deserialized with pydantic_model.parse_raw",
        }

        messages = [system_message]

        # Add examples to the messages if provided
        if examples:
            for example in examples:
                messages.append({"role": "user", "content": example["input"]})
                messages.append({"role": "assistant", "content": example["output"]})

        prompt = doc.format(**kwargs)

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        attempt = 0
        last_exception = None

        # Retry the specified number of times in case of failure
        while attempt < retries:
            try:
                content = openai.ChatCompletion.create(
                    messages=messages,
                    temperature=temperature,
                    model=model,
                )["choices"][0]["message"]["content"]
                obj = json.loads(content)
            except Exception as e:
                last_exception = e
                error_msg = f"json.loads error: {e}"
                logger.error(error_msg)
                messages.append(
                    {"role": "user", "content": f"{e.__class__.__name__}: {e}"}
                )
                attempt += 1
                continue

            try:
                return cls.parse_obj(obj)
            except ValidationError as e:
                last_exception = e
                error_msg = f"pydantic.ValidationError: {e}"
                logger.error(error_msg)
                messages.append(
                    {"role": "user", "content": f"{e.__class__.__name__}: {e}"}
                )
                attempt += 1

        if last_exception:
            raise last_exception


class OpenAiBase(BaseModel, OpenAiMixin):
    ...


if __name__ == "__main__":

    class SentimentSchema(OpenAiBase):
        """
        Determine the sentiment of the given text:

        {content}
        """

        sentiment: str = Field(description="Either -1, 0, or 1.")

    examples = [
        {
            "input": "this library is neat!",
            "output": SentimentSchema(sentiment="1").json(),
        },
        {
            "input": "don't touch that",
            "output": SentimentSchema(sentiment="-1").json(),
        },
    ]

    positive = SentimentSchema.create(
        content="I love pizza!",
    )
    negative = SentimentSchema.create(
        content="I hate pizza!",
        examples=examples,
    )

    print(f"{positive = }")
    print(f"{negative = }")
    # outputs:
    # positive = SentimentSchema(sentiment='1')
    # negative = SentimentSchema(sentiment='-1')
