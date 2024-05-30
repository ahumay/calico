from datetime import datetime
from langchain.adapters.openai import convert_openai_messages
from langchain_openai import ChatOpenAI
import json5 as json

sample_json = """
{
    "debate": Panel debate as a single string, with each expert giving their reasoning,
    "answer_rationale": A single string containing the final answer chosen by the panel with a step-by-step explanation,
    "answer": A single letter corresponding to the final answer
}
"""

class WriterAgent:
    def __init__(self):
        pass

    def writer(self, question: str):

        prompt = [{
            "role": "system",
            "content": f"\"\"\"{question}\"\"\"\n "
                       f"3 experts on a panel are discussing the question with a discussion, trying to solve it step by step, and make sure the result is correct."
                       f"Please only a JSON in the following format:\n"
                       f"{sample_json}\n "
        }]
        lc_messages = convert_openai_messages(prompt)
        # WHEN CHANGING MODEL from gpt-3.5-turbo-0301, can add this back in:
        # optional_params = {
        #     "response_format": {"type": "json_object"}
        # }
        # response = ChatOpenAI(model='gpt-3.5-turbo-0301', max_retries=1, model_kwargs=optional_params).invoke(lc_messages).content
        try:
            response = ChatOpenAI(model='gpt-3.5-turbo-0301', max_retries=1).invoke(lc_messages).content
        except Exception as e:
            print(f"An error occurred: {e}")
        return json.loads(response)

    def revise(self, article: dict):
        prompt = [{
            "role": "system",
            "content": f"\"\"\"{article['question']}\"\"\"\n "
                       f"3 experts on a panel are discussing the question. They previously chose {article['answer']}, but the following critique enclosed in the \"\"\" delimiters from another panel came in:"
                       f"\"\"\"{article['critique']}\"\"\"\""
                       f"You can now update the answer if you think the critique is valid. Return only a JSON in the following format:\n"
                       f"{sample_json}\n "
        }]

        lc_messages = convert_openai_messages(prompt)
        # WHEN CHANGING MODEL from gpt-3.5-turbo-0301, can add this back in:
        # optional_params = {
        #     "response_format": {"type": "json_object"}
        # }
        # response = ChatOpenAI(model='gpt-3.5-turbo-0301', max_retries=1, model_kwargs=optional_params).invoke(lc_messages).content
        try:
            response = ChatOpenAI(model='gpt-3.5-turbo-0301', max_retries=1).invoke(lc_messages).content
        except Exception as e:
            print(f"An error occurred: {e}")

        response = json.loads(response)
        # print(f"DEBUG: Revising")
        return response

    def run(self, article: dict):
        critique = article.get("critique")
        if critique is not None:
            article.update(self.revise(article))
        else:
            article.update(self.writer(article["question"]))
        return article
