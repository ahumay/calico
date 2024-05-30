from datetime import datetime
from langchain.adapters.openai import convert_openai_messages
from langchain_openai import ChatOpenAI
import json5 as json

sample_json = """
{
    "thought_process": What you know about the problem, and how you arrived at your answer
    "answer": A single letter corresponding to the final answer
    "message": "Optional message to your counterpart Professor"
}
"""

class WriterAgent:
    def __init__(self):
        pass

    def writer(self, question: str):

        prompt = [{
            "role": "system",
            "content": f"You're a savant at the Institute for Advanced Study. You and a colleague have been tasked to answer this question correctly, or else the world will blow up: \"\"\"{question}\"\"\"\n "
                       f"Please return nothing but a JSON in the following format:\n"
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
            "content": f"You're a savant at the Institute for Advanced Study. You and a colleague have been tasked to answer this question correctly, or else the world will blow up. You gave an answer, and your colleague has given feedback below, separated by \"\"\" delimiters."
                       f"\"\"\"{article['critique']}\"\"\"\""
                       f"Please either update the answer or reasoning and return nothing but a JSON in the following format:\n"
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
        print(f"DEBUG: Revising")
        return response

    def run(self, article: dict):
        critique = article.get("critique")
        if critique is not None:
            article.update(self.revise(article))
        else:
            article.update(self.writer(article["question"]))
        return article
