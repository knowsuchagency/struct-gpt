import io
import json
import logging
from pprint import pprint
from typing import TypeVar, TypedDict

import openai
from pydantic import BaseModel, ValidationError
from ruamel.yaml import YAML

Model = TypeVar("Model", bound=BaseModel)

yaml = YAML(typ="safe")

logger = logging.getLogger(__name__)


class Example(TypedDict):
    input: str
    output: str | dict | Model


class OpenAiMixin:
    @classmethod
    def from_openai(
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
        assert cls.__doc__ and (
            doc := cls.__doc__.strip()
        ), "please add a docstring explaining how to destructure the prompt"

        assert (
            examples or kwargs
        ), "please provide either examples or keyword args for the docstring"

        assert 0 <= temperature <= 1, "temperature should be between 0 and 1"

        assert retries >= 0, "retries should be a positive integer"

        # Convert the JSON schema to YAML since it takes up fewer tokens
        with io.StringIO() as fp:
            json_schema = json.loads(cls.schema_json())
            yaml.dump(json_schema, fp)
            yaml_json_schema = fp.getvalue()

        # Set up the system message to guide the assistant
        directions = [
            f"Please respond ONLY with valid json that conforms to this json_schema:",
            yaml_json_schema,
            "Don't include additional text other than the object. It gets deserialized with pydantic_model.parse_raw",
        ]

        system_message = {
            "role": "system",
            "content": "\n".join(directions),
        }

        messages = [system_message]

        # Add examples to the messages if provided
        if examples:
            for example in examples:
                input_ = example["input"]
                output = example["output"]

                if isinstance(output, str):
                    try:
                        output = cls.parse_raw(output).json()
                    except ValidationError as e:
                        class_name = cls.__name__
                        raise e from ValueError(
                            f"output ({output}) should be a json representation of {class_name} or an instance of it"
                        )
                elif isinstance(output, dict):
                    output = cls.parse_obj(output).json()
                else:
                    output = output.json()

                messages.append({"role": "user", "content": input_})
                messages.append({"role": "assistant", "content": output})

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
        while attempt <= retries:
            try:
                content = openai.ChatCompletion.create(
                    messages=messages,
                    temperature=temperature,
                    model=model,
                )["choices"][0]["message"]["content"]
                obj = json.loads(content)
            except Exception as e:
                last_exception = e
                error_msg = f"{e.__class__.__name__}: {e}"
                logger.error(error_msg)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            error_msg
                            + "\n"
                            + "note: please don't apologize. simply return the correct json object"
                        ),
                    }
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
            pprint(messages)
            raise last_exception


class OpenAiBase(BaseModel, OpenAiMixin):
    ...
