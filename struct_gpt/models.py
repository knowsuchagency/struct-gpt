import io
import json
from pprint import pprint
from typing import TypeVar, Literal, Mapping

import openai
from pydantic import BaseModel, Field
from ruamel.yaml import YAML

Model = TypeVar("Model", bound=BaseModel)
Example = Mapping[Literal["input", "output"], str]

yaml = YAML(typ="safe")


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
        assert (
            doc := cls.__doc__
        ), "please add a docstring explaining how to destructure the prompt"

        with io.StringIO() as fp:
            json_schema = json.loads(cls.schema_json())
            yaml.dump(json_schema, fp)
            yaml_json_schema = fp.getvalue()

        system_message = {
            "role": "system",
            "content": f"Please respond ONLY with valid YAML that conforms to this pydantic json_schema: {yaml_json_schema}. Output will be deserialized with {cls.__name__}.parse_obj(yaml.load(content)).",
        }

        messages = [system_message]

        if examples:
            for example in examples:
                messages.append({"role": "user", "content": example["input"]})
                messages.append({"role": "assistant", "content": example["output"]})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": doc.format(**kwargs),
                }
            )

        response = openai.ChatCompletion.create(
            messages=messages,
            temperature=temperature,
            model=model,
        )
        response_content = response["choices"][0]["message"]["content"]
        return cls.parse_obj(yaml.load(response_content))

        attempt = 0

        while (
            response := openai.ChatCompletion.create(
                messages=messages,
                temperature=temperature,
                model=model,
            )
            and attempt < retries
        ):
            pprint(response)

            response_content = response["choices"][0]["message"]["content"]

            try:
                return cls.parse_obj(yaml.load(response_content))
            except Exception as e:
                messages.append(
                    {"role": "user", "content": f"{e.__class__.__name__}: {e}"}
                )
                attempt += 1


class YamlMixin:
    def yaml(self: Model):
        with io.StringIO() as fp:
            yaml.dump(self.dict(), fp)
            return fp.getvalue()


class OpenAiBase(BaseModel, OpenAiMixin, YamlMixin):
    ...


if __name__ == "__main__":

    class SentimentSchema(OpenAiBase):
        """
        Determine the sentiment of the given text:

        {content}
        """

        sentiment: str = Field(description="Either -1, 0, or 1.")

    examples = [
        {"input": "I love pizza!", "output": SentimentSchema(sentiment="1").yaml()},
        {"input": "I hate pizza!", "output": SentimentSchema(sentiment="-1").yaml()},
    ]

    positive = SentimentSchema.create(
        content="I love pizza!",
        # examples=examples,
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
