import time
import os
from pydantic import BaseModel, Field
from typing import Any, Dict, Union
from openai import OpenAI
import timeout_decorator
from dotenv import load_dotenv
import re
import io
import contextlib
import traceback

from prompts.inference import prompt_final_gpt5, prompt_total_gpt5

load_dotenv()

def remove_markdown_syntax(text: str) -> str:
    # removing triple backtick code blocks (```python ... ```)
    text = re.sub(r"```[\s\S]*?```", lambda m: re.sub(r"^```.*\n|```$", '', m.group()), text)

    # removing inline code (`code`)
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # removing bold (**text** or __text__)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # removing italic (*text* or _text_)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # removing blockquotes
    text = re.sub(r"^>\s?", '', text, flags=re.MULTILINE)

    text = text.replace("python", "")
    return text.strip()

def format_prompt(prompt: str, attr: dict, **kwargs) -> str:
    return prompt.format(**attr)

def add_metadata(total_metadata, metadata):
    total_metadata["input_tokens"] += metadata['input_tokens']
    total_metadata["output_tokens"] += metadata['output_tokens']
    total_metadata["text"] += "\n" + metadata["text"]

    if "content_used" in metadata.keys():
        total_metadata["content_used"] += metadata['content_used']
        total_metadata["total_content"] += metadata['total_content']
        total_metadata["num_tables"] += 1

    return total_metadata

def extract_result(text: str, pattern: str) -> str:
    position = text.lower().rfind(pattern.lower())
    if position == -1:
        print(f"Cannot find pattern '{pattern}' in '{text}'")
        return ""
    else:
        position += len(pattern)
    return text[position:].strip()

class OpenAIModel(BaseModel):
    model_name: str = Field("gpt-5-mini", strict=True, description="Name of the openai model as per their official website")
    question_model_name: str = Field("gpt-5-mini", strict=True, description="Name of the openai model to create natural language questions")
    temperature: float = Field(.5, strict=True, description="The temperature of the model in between 0 and 1")
    temperature_question: float = Field(.0000000000000000000001, strict=True, description="The temperature of the model in between 0 and 1")
    top_p: float = Field(.1, strict=True, description="The top_p of the model in between 0 and 1")
    top_p_question: float = Field(.0000000000000000000001, strict=True, description="The top_p of the model in between 0 and 1")
    client: Any = None
    max_retries: int = Field(50, strict=True, description="Number of retries in case of failed OpenAI API call")

    def init_client(self):
        self.client = OpenAI()

    @timeout_decorator.timeout(60, timeout_exception=StopIteration)
    def call_gpt(self, prompt: Union[str, Dict], create_question=False) -> (str, dict):
        if not create_question:
            temp = self.temperature
        else:
            temp = self.temperature_question

        if "5" in self.model_name:
            temp = 1

        if isinstance(prompt, str):
            messages = [
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            ]
        else:
            messages = prompt

        completion = self.client.chat.completions.create(
                model=self.model_name if not create_question else self.question_model_name,
                messages=messages,
                temperature=temp,
                seed=42,
        )

        metadata = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }

        return completion.choices[0].message.content, metadata

    def query(self, prompt: str, attr: dict, create_question=False, **kwargs) -> tuple[str, dict]:
        if self.client is None:
            self.init_client()
        if len(attr) > 0:
            prompt = format_prompt(prompt, attr)
        text = prompt

        for i in range(self.max_retries):
            try:
                response = self.call_gpt(prompt, create_question=create_question)
                text+="\n"+response[0]
                response[1]["text"] = text
                return response
            except:
                time.sleep(20)
                print("Failed to get a response. Retrying...")

        raise RuntimeError(f"Failed to query OpenAI after {self.max_retries} retries.")

    def execute(self, python_text: str, attr: dict, fallback_prompt: str):
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        error = False
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                exec(python_text.strip())
            except Exception as e:
                error = True
                print("Generated python function is not executable. Falling back to cot...")
                traceback.print_exc(file=stderr_buffer)

        if error:
            print(stderr_buffer.getvalue(), flush=True)
            result = self.query(fallback_prompt, attr)
        else:
            result = stdout_buffer.getvalue()

        # in case error is True, there's no need to apply the last step "final answer:" after python execution, because it is already done during the error handling
        # so inside the caller function, do not launch the final LLM call if error is True
        return result, error

    def query_pot(self, fallback_prompt: str, attr: dict) -> str:
        """
        given a query and a list of tables, this function processes each table in this way:
        - PoT: the LLM generates the Python code to answer the question
        - Python execution: execute the Python code
        """
        if self.client is None:
            self.init_client()

        prompt = prompt_total_gpt5.format(**attr)
        messages =[
            {
                "role": "system",
                "content": "You are an expert data analyst and Python programmer specialized in data extraction from HTML tables.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

        python_text_raw, _ = self.call_gpt(messages)

        python_code = remove_markdown_syntax(extract_result(python_text_raw, "Final answer:"))
        results, error = self.execute(python_code, attr, fallback_prompt)

        if error:
            print("Error in pot parsing, falling back to cot...")
            return results

        continuation_message = [
            {
                "role": "assistant",
                "content": python_code + "\n" + results
            },
            {
                "role": "user",
                "content": prompt_final_gpt5
            }
        ]

        messages.extend(continuation_message)
        results_final = self.call_gpt(messages)
        text = prompt + "\n" + python_code + "\n" + results + "\n" + prompt_final_gpt5 + "\n" + results_final[0]
        results_final[1]["text"] = text

        # final answer
        return results_final


class QwenModel(BaseModel):
    model_name: str = Field("Qwen/Qwen3.5-9B", strict=True, description="Name of the Qwen model")
    temperature: float = Field(.0, strict=True, description="The temperature of the model in between 0 and 1")
    top_p: float = Field(.8, strict=True, description="The top_p of the model in between 0 and 1")
    base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"), strict=True, description="vLLM server URL")
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "EMPTY"), strict=True, description="API key for the server")
    client: Any = None
    max_retries: int = Field(50, strict=True, description="Number of retries in case of failed API call")

    def init_client(self):
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        print(f"Using base_url={self.base_url} model_name={self.model_name}")

    @timeout_decorator.timeout(60, timeout_exception=StopIteration)
    def call_gpt(self, prompt: str) -> (str, dict):
        completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}"
                    }
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                seed=42,
                max_tokens=16384,
        )

        metadata = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }

        return completion.choices[0].message.content, metadata

    def query(self, prompt: str, attr: dict, **kwargs) -> tuple[str, dict]:
        if self.client is None:
            self.init_client()
        if len(attr) > 0:
            prompt = format_prompt(prompt, attr)
        text = prompt

        for i in range(self.max_retries):
            try:
                response = self.call_gpt(prompt)
                text += "\n" + response[0]
                response[1]["text"] = text
                return response
            except Exception as e:
                print(f"Failed to get a response: {e}. Retrying...")
                time.sleep(20)

        raise RuntimeError(f"Failed to query Qwen after {self.max_retries} retries.")
