import json
import re
import numpy as np

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def number_to_letter(number):
    # Assuming 'A' corresponds to 1, 'B' to 2, etc.
    return chr(ord('A') + number - 1)

def parse_answer(input_str):
    pattern = r'\((\w)\)\s*$'
    match = re.search(pattern, input_str.strip())

    if match:
        print(f"Matched: {match.group(1)}")
        return match.group(1).upper()
    
    print(f"No match in: {input_str}")
    return None
    # # print("predicted solution")
    # # print(input_str)
    # # print("matches")
    # # print(matches)

    # for match_str in matches[::-1]:
    #     solution = match_str.upper()
    #     if solution:
    #         break

    # return solution

def compute_accuracy(gt, pred_solutions):
    print(f"\nGround Truth: {gt}")
    print(f"Predicted Solutions: {pred_solutions}")
    if isinstance(pred_solutions, list):
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solution)
                # if pred_answer is not None:
                #     # Convert number to letter if necessary
                #     try:
                #         pred_answer = number_to_letter(int(pred_answer))
                #     except ValueError:
                #         pass  # Handle or log the error as appropriate

            if pred_answer is not None:
                pred_answers.append(pred_answer)

        if not pred_answers:
            return 0, pred_solutions
        pred_answer = most_frequent(pred_answers)
        # pred_answer = pred_answers[0]
    else:
        pred_answer = parse_answer(pred_solutions)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solutions)
            # if pred_answer is not None:
            #     # Convert number to letter if necessary
            #     try:
            #         pred_answer = number_to_letter(int(pred_answer))
            #     except ValueError:
            #         pass  # Handle or log the error as appropriate

    print(f"Final Predicted Answer: {pred_answer}")
    
    if gt == pred_answer:
        return 1, pred_solutions
    else:
        return 0, pred_solutions

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

if __name__ == "__main__":
    response_dict = json.load(open("mmlu_3_2_debate+dspy_temp0.5to1_gpt4o.json", "r"))
    questions = list(response_dict.keys())

    accuracies = []

    for i, question in enumerate(questions):
        responses, gt = response_dict[question]

        pred_solutions = []
        for response in responses:
            pred_solution = response[-1]
            pred_solutions.append(pred_solution)

        accurate, pred_solutions = compute_accuracy(gt, pred_solutions)
        if accurate == 0:
            for pred_solution in pred_solutions:
                print(f"\n********************************************************************************\n******************** Question {i + 1}: {question} ******************** Expected: {gt}, but got:\n\n {pred_solution}")

        if accurate is not None:
            accuracies.append(float(accurate))
        else:
            import pdb
            pdb.set_trace()
            print(gt)

        print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
