import dspy
from dspy import Example, Signature, Predict, Module
from dspy.teleprompt import BootstrapFewShot
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import random

random.seed(0)

# Configure DSPy settings
dspy.configure(lm="gpt-4")

# Simple signature for MMLU question answering
class MMLUSignature(Signature):
    """ Answer MMLU questions based on provided context."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# Define the pipeline
class MMLUPipeline(Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict('question -> answer')

    def forward(self, question):
        answer = self.predict(question=question)
        return answer

# Function to parse question and answer
def parse_question_answer(df, ix):
    question_text = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]
    formatted_question = "Can you answer the following question as accurately as possible? ```{}: A) {}, B) {}, C) {}, D) {}```. Put your final answer in the form of it's corresponding capitalized letter choice such as (i.e. '(A)') as the last text in your response".format(question_text, a, b, c, d)
    answer = df.iloc[ix, 5]
    return formatted_question, answer

# Load the MMLU CSV files
all_files = glob.glob("./data/test/*.csv")

# Concatenate all CSV files into a single DataFrame
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# Prepare examples
examples = [
    Example(question=parse_question_answer(df, i)[0], answer=parse_question_answer(df, i)[1]).with_inputs("question")
    for i in range(len(df))
]

# Split the data
train_examples, test_examples = train_test_split(examples, test_size=0.2, random_state=0)
train_examples, dev_examples = train_test_split(train_examples, test_size=0.1, random_state=0)

# Reduce the size of the training and development sets for a manageable cost
train_examples = train_examples[:100]
dev_examples = dev_examples[:20]

# Define a simple validation logic
def validate_mmlu_answer(example, pred, trace=None):
    return example.answer.strip().lower() == pred.answer.strip().lower()

# Set up the optimizer
teleprompter = BootstrapFewShot(
    metric=validate_mmlu_answer,
    max_bootstrapped_demos=8,
    max_labeled_demos=8,
)

# Compile the pipeline with the training examples
compiled_pipeline = teleprompter.compile(MMLUPipeline(), trainset=train_examples, valset=dev_examples)

# Function to compute accuracy
def compute_accuracy(ground_truth, predicted_answer):
    return ground_truth.strip().lower() == predicted_answer.strip().lower()

# Evaluate the compiled pipeline
accuracies = []

for example in test_examples:
    question = example.question
    ground_truth = example.answer

    predicted_answer = compiled_pipeline(question).answer
    is_correct = compute_accuracy(ground_truth, predicted_answer)

    accuracies.append(is_correct)
    if not is_correct:
        print(f"Question: {question}")
        print(f"Expected {ground_truth}, but got: {predicted_answer}")

# Compute overall accuracy
accuracy = sum(accuracies) / len(accuracies)
print(f"Overall Accuracy: {accuracy:.2f}")
