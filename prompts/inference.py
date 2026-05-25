prompt = """Answer the following question given the provided HTML table.
First reason step-by-step, then write "Final answer:" followed exclusively by the correct answer. Do not write anything else after "Final answer:"
Every calculation must be done with a precision of exactly 6 decimal places.
Only the numerical value must be written in the final answer.

Question: {question}
Table:
{table}

Let's think step-by-step. """

prompt_total_gpt5 = """You must create the python code capable of answering the following question given the provided tables. First reason step-by-step, then write "Final answer:" followed exclusively by the python code.
The Python code must be runnable "as it is", so make sure to include the relevant imports. At the same time, all the imported libraries must belong to standard python3, as the env will be run in an env where it is not possible to install new libraries.
At the end of the python function, print() the result.
Every calculation must be done with a precision of exactly 6 decimal places.
Do not rewrite and load back the whole tables/dataframes inside the Python script, just extract and use the relevant values.
The python code must not contain "if __name__ == "__main__":".
Ensure that the final answer is in the expected form. Do not write anything else after "Final answer:". Do not use Markdown syntax.

Question: {question}
Tables: {table}

Let's think step-by-step. """

prompt_final_gpt5 = """Now take the Python output, analyze its execution, and in the end, write "Final answer:" followed exclusively by the final answer to the question.
Every calculation must be done with a precision of exactly 6 decimal places.
Do not write anything else after "Final answer:": only the numerical value must be written in the final answer, and no unit of measurement. Do not use Markdown syntax."""