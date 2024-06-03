import dspy
from dspy.predict import Predict
from dspy.primitives.program import Module
import dsp

class MultiChainComparison(Module):
    def __init__(self, signature, M=5, temperature=0.7, **config):
        super().__init__()

        self.M = M
        signature = dspy.Predict(signature).signature
        *keys, last_key = signature.kwargs.keys()

        extended_kwargs = {key: signature.kwargs[key] for key in keys}

        for idx in range(M):
            candidate_type = dsp.Type(prefix=f"Student Attempt #{idx+1}:", desc="${reasoning attempt}")
            extended_kwargs.update({f'reasoning_attempt_{idx+1}': candidate_type})
        
        rationale_type = dsp.Type(prefix="Accurate Reasoning: Thank you everyone. Let's now holistically", desc="${corrected reasoning}")
        extended_kwargs.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})

        signature = dspy.Template(signature.instructions, **extended_kwargs)
        self.predict = dspy.Predict(signature, temperature=temperature, **config)
        self.last_key = last_key

class BasicQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="1-5 words") #1 uppercase letter corresponding to the correct answer
