import io
import json
import logging
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


# Small shims so the mixin works with both pydantic v1 and v2.


def _json_schema(cls) -> dict:
    if hasattr(cls, "model_json_schema"):
        return cls.model_json_schema()
    return json.loads(cls.schema_json())


def _validate_json(cls, data: str):
    if hasattr(cls, "model_validate_json"):
        return cls.model_validate_json(data)
    return cls.parse_raw(data)


def _validate(cls, obj):
    if hasattr(cls, "model_validate"):
        return cls.model_validate(obj)
    return cls.parse_obj(obj)


def _dump_json(instance) -> str:
    if hasattr(instance, "model_dump_json"):
        return instance.model_dump_json()
    return instance.json()


class OpenAiMixin:
    @classmethod
    def from_openai(
        cls: Model,
        temperature=0,
        model="gpt-4o-mini",
        retries=2,
        examples: list[Example] = None,
        client: "openai.OpenAI | None" = None,
        json_mode: bool = True,
        **kwargs,
    ) -> Model:
        """
        Create a new model instance using OpenAI's API.

        Args:
            temperature: Controls the randomness of the response.
            model: The specific OpenAI model to use.
            retries: Number of times to retry in case of failure.
            examples: List of input-output example pairs. Useful to improve the model's accuracy.
            client: An `openai.OpenAI` client. Defaults to `openai.OpenAI()`, which reads
                `OPENAI_API_KEY` (and friends) from the environment.
            json_mode: Ask the API for a JSON object (`response_format={"type": "json_object"}`).
                Set to False for models that don't support JSON mode.
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
            yaml.dump(_json_schema(cls), fp)
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
                        output = _dump_json(_validate_json(cls, output))
                    except ValidationError as e:
                        class_name = cls.__name__
                        raise e from ValueError(
                            f"output ({output}) should be a json representation of {class_name} or an instance of it"
                        )
                elif isinstance(output, dict):
                    output = _dump_json(_validate(cls, output))
                else:
                    output = _dump_json(output)

                messages.append({"role": "user", "content": input_})
                messages.append({"role": "assistant", "content": output})

        prompt = doc.format(**kwargs)

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        if client is None:
            client = openai.OpenAI()

        request = {"model": model, "temperature": temperature}
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        attempt = 0
        last_exception = None

        # Retry the specified number of times in case of failure
        while attempt <= retries:
            content = None
            try:
                response = client.chat.completions.create(messages=messages, **request)
                content = response.choices[0].message.content
                obj = json.loads(content)
            except Exception as e:
                last_exception = e
                error_msg = f"{e.__class__.__name__}: {e}"
                logger.error(error_msg)
                if content is not None:
                    messages.append({"role": "assistant", "content": content})
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
                return _validate(cls, obj)
            except ValidationError as e:
                last_exception = e
                error_msg = f"pydantic.ValidationError: {e}"
                logger.error(error_msg)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {"role": "user", "content": f"{e.__class__.__name__}: {e}"}
                )
                attempt += 1

        if last_exception:
            raise last_exception


class OpenAiBase(BaseModel, OpenAiMixin):
    ...
