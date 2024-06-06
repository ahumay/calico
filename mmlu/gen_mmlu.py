from glob import glob
import pandas as pd
import json
import time
import random
import openai
from agents.langgraph_agent import MasterAgent
from dotenv import load_dotenv
import os
import dspy
from dspy.predict import aggregation
load_dotenv()

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return "Can you double check that your answer is correct. Put your final answer in the form of its corresponding capitalized letter choice such as (i.e. '(A)') as the last text in your response."

    prefix_string = "\nOther agents are working on this problem as well. Here are their solutions in --- delimiters: "

    for i, agent in enumerate(agents):
        agent_response = agent[idx]
        response = "\nAgent {} solution: ---{}---\n".format(i + 1, agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\nUsing the reasoning from other agents as advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your final answer in the form of its corresponding capitalized letter choice such as (i.e. '(A)') as the last text in your response.""".format(question)
    return prefix_string


def construct_assistant_message(completion):
    # Langraph:
    # content = completion["thought_process"] + " " + completion["answer"]
    # Raw GPT:
    # content = completion["choices"][0]["message"]["content"]
    # DSPy: needs to update content to completion.answer (or completion.rationale + completion.answer):
    content = completion.rationale + completion.answer
    return content

def generate_answer(answer_context):
    master_agent = MasterAgent()
    try:
        # Langraph:
        # completion = master_agent.run(answer_context)
        # Raw GPT: 
        # completion = openai.ChatCompletion.create(
        #           model="gpt-3.5-turbo-0301",
        #           messages=answer_context,
        #           n=1)
        # DSPy:
        completions = []
        print("Answer context:", answer_context)
        qa = dspy.ChainOfThought('question -> answer')
        for idx in range(5):
            completions.append(qa(question=answer_context, config=dict(temperature=0.0+random.uniform(0.0, 1.1))))
        completion = aggregation.majority(completions)
        print("Completion:", completion)
    except Exception as e:
        print("Error in generate_answer:", e)
        time.sleep(1)
        return generate_answer(answer_context)

    return completion

def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = "Can you answer the following question as accurately as possible? ```{}: A) {}, B) {}, C) {}, D) {}```. Explain your answer, putting your final answer in the form of it's corresponding capitalized letter choice such as (i.e. '(A)') as the last text in your response".format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer

if __name__ == "__main__":
    # DSPy:
    api_key = os.getenv('OPENAI_API_KEY')
    llm = dspy.OpenAI(model='gpt-4o', api_key=api_key, max_tokens = 2048)
    dspy.settings.configure(lm=llm)

    agents = 3
    rounds = 2

    tasks = glob("./data/test/*.csv")

    dfs = [pd.read_csv(task) for task in tasks]

    random.seed(0)
    response_dict = {}

    for i in range(100):
        df = random.choice(dfs)
        ix = len(df)
        idx = random.randint(0, ix-1)

        question, answer = parse_question_answer(df, idx)
        og_question = question

        agent_contexts = [[{"content": question}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2 * round - 1)
                    question = message

                completion = generate_answer(question)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print("------------------------------------------ One of final answer:    ", assistant_message)
        response_dict[og_question] = (agent_contexts, answer)

    json.dump(response_dict, open("mmlu_{}_{}.json".format(agents, rounds), "w"))