from datetime import datetime
from langchain.adapters.openai import convert_openai_messages
from langchain_openai import ChatOpenAI

class CritiqueAgent:
    def __init__(self):
        pass

    def critique(self, article: dict):
        prompt = [{
            "role": "system",
            "content": f"You're a savant at the Institute for Advanced Study. You and a colleague have been tasked to answer this question in ``` delimiters correctly: ```{article['question']}```"
                       f"Your colleague's answer was {article['answer']}"
                       f"The following \"\"\" delimiters contain the thought process: \"\"\"{article['thought_process']}\"\"\" "
                       f"The following --- delimiters contain their optional message to you: ---{article['message']}---"
                       f"return None if you disagree with the answer.\n"
                       f"Please return a string of your critique or None.\n"
        }]

        lc_messages = convert_openai_messages(prompt)
        response = ChatOpenAI(model='gpt-3.5-turbo-0301', max_retries=1).invoke(lc_messages).content
        if response == 'None':
            print(f"DEBUG: Critique: None\n")
            return {'critique': None}
        else:
            # print(f"For article: {article['title']}")
            print(f"DEBUG: Feedback: {response}\n")
            return {'critique': response, 'message': None}

    def run(self, article: dict):
        article.update(self.critique(article))
        return article
