import os
import time
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import Graph

# Import agent classes
from . import WriterAgent, CritiqueAgent, ConcluderAgent

class MasterAgent:
    def __init__(self):
        self.output_dir = f"outputs/run_{int(time.time())}"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, answer_context):
        # Initialize agents
        writer_agent = WriterAgent()
        critique_agent = CritiqueAgent()
        concluder_agent = ConcluderAgent()

        # Define a Langchain graph
        workflow = Graph()

        # Add nodes for each agent
        # workflow.add_node("curate", curator_agent.run)
        # workflow.add_node("search", search_agent.run)
        workflow.add_node("write", writer_agent.run)
        workflow.add_node("critique", critique_agent.run)
        workflow.add_node("conclude", concluder_agent.run)  
        # workflow.add_node("design", designer_agent.run)

        # Set up edges
        # workflow.add_edge('curate', 'search')
        # workflow.add_edge('search', 'write')
        workflow.add_edge('write', 'critique')
        workflow.add_conditional_edges(source='critique',
                path=lambda x: "accept" if x['critique'] is None else "revise",
                path_map={"accept": "conclude", "revise": "write"})

        # set up start and end nodes
        workflow.set_entry_point("write")
        workflow.set_finish_point("conclude")

        # compile the graph
        chain = workflow.compile()
        mmlu_question = answer_context[0]["content"]
        print("Running the Langchain graph with question: {}".format(mmlu_question))
        chain.invoke({"question": mmlu_question})
