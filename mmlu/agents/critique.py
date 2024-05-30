from datetime import datetime
from langchain.adapters.openai import convert_openai_messages
from langchain_openai import ChatOpenAI

class CritiqueAgent:
    def __init__(self):
        pass

    def critique(self, article: dict):
        prompt = [{
            "role": "system",
            "content": f"{article['question']}"
                       f"3 experts on a panel are discussing the question with a discussion, trying to solve it step by step, and make sure the result is correct."
                       f"A different panel's answer was {article['answer']}"
                       f"The following \"\"\" delimiters contain their debate: \"\"\"{article['debate']}\"\"\" "
                       f"return ONLY the word 'None' if you agree with the answer.\n"
        }]

        lc_messages = convert_openai_messages(prompt)
        response = ChatOpenAI(model='gpt-3.5-turbo-0301', max_retries=1).invoke(lc_messages).content
        if response == 'None':
            # print(f"DEBUG: Critique: None\n")
            return {'critique': None}
        else:
            # print(f"DEBUG: Feedback: {response}\n")
            return {'critique': response, 'message': None}

    def run(self, article: dict):
        article.update(self.critique(article))
        return article
