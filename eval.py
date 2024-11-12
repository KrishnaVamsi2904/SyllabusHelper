from pydantic import BaseModel
import google.generativeai as genai
import instructor
from pydantic import BaseModel
from typing import List
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric
)
from deepeval import evaluate

class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self):
        self.model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5 Flash"
    
class ResponseSchema(BaseModel):
    answer: str

custom_llm = CustomGeminiFlash()

contextual_precision = ContextualPrecisionMetric(model=custom_llm)
contextual_recall = ContextualRecallMetric(model=custom_llm)
contextual_relevancy = ContextualRelevancyMetric(model=custom_llm)
answer_relevancy = AnswerRelevancyMetric(model = custom_llm)
faithfulness = FaithfulnessMetric(model = custom_llm)

tc1 = LLMTestCase(input = "what are the topics included in unit 3 of 'engineering chemistry'?", 
                  actual_output = "Unit III (Electrochemistry and Corrosion) covers: electrode potential; types of electrodes (calomel and glass electrodes – construction and working); electrochemical series and applications; electrochemical cells (galvanic & electrolytic cells); Nernst equation – applications, numerical problems; batteries (primary and secondary types, lithium metal, lithium ion and lead acid batteries); types of fuel cells (hydrogen-oxygen fuel cell - applications and advantages, microbial fuel cell); corrosion (definition, causes and effects, theories of chemical and electrochemical corrosion with mechanism, types of corrosion - galvanic, concentration cell and pitting corrosions, factors affecting corrosion (nature of metal & nature of environment), corrosion control methods (proper designing, cathodic protection (sacrificial anodic and impressed current cathodic protection), metallic coatings: hot dipping - galvanization and tinning, electroplating, electroless plating of nickel.", 
                  expected_output = "Electrochemistry : Electrode potential, types of electrodes: calomel and glass electrodes - construction and working, electrochemical series and applications, electrochemical cells: Galvanic & electrolytic cells, Nernst equation - applications, numerical problems, Batteries: primary and secondary types, lithium metal, lithium ion and lead acid batteries. Types of Fuel cells: hydrogen -oxygen fuel cell - applications and advantages, microbial fuel cell.""",
                  retrieval_context = ["""UNIT III Electrochemistry and Corrosion: (12 Lectures)
Electrochemistry : Electrode potential, types of electrodes: calomel and glass electrodes - construction and working, electrochemical series and applications, electrochemical cells:
Galvanic & electrolytic cells, Nernst equation - applications, numerical problems, Batteries:
primary and secondary types, lithium metal, lithium ion and lead acid batteries. Types of
Fuel cells: hydrogen -oxygen fuel cell - applications and advantages, microbial fuel cell.
Corrosion: Definition ,causes and effects of corrosion, The ories of chemical and electro
chemical corrosion with mechanism, Types of corrosion - Galvanic, concentration cell and pitting corrosions, factors affecting corrosion (Nature of metal & Nature of Environment),
corrosion control methods: Proper designing, cathodic protection (sacrificial anodic and
impressed current cathodic protection), Metallic coatings: Hot dipping - Galvanization and
tinning, electroplating, electroless plating of nickel."""])

tc2 = LLMTestCase(input = "what are the prerequisities for 'software engineering' subject", 
                actual_output= "Basic knowledge of programming language. Idea about Database systems. Design of flow charts.", 
                expected_output = """1. Basic knowledge of programming language. 2. Idea about Database systems. 3. Design of flow charts.""", 
                retrieval_context = [
                    """Prerequisites:

                        Basic knowledge of programming language
                        Idea about Data base systems
                        Design of flow charts"""
    ])

tc3 = LLMTestCase(input = "what are the recommended books for artificial intelligence subject?", 
                actual_output= "Artificial Intelligence - A modern approach by Stuart Russel, Peter Norvig, 2nd edition, PHI/Pearson Artificial Intelligence by Riche & K. Night, 2nd edition, TMH.", 
                expected_output = "Text Books:  1. Artificial Intelligence-A modern approach-by Staurt Russel, Peter Norvig, 2nd edition, PHI/Pearson  References: 1. Artificial Intelligence – Riche &K.Night , 2nd edition, TMH. 2. Paradigms of Artificial intelligence programming, case studies in common lisp-Peter. Norvig, Morgan Kaufmann.ISBN-13:978-1558601918. 3. Robotics: Fundamental Concepts and Analysis –Ashitava Goshal, oxford. 4. A Textbook of Robotics 1-Basic Concepts-M. Shoham-Springer US.", 
                retrieval_context= [
                    """Text Books:
Artificial Intelligence -A modern approach -by Staurt Russel, Peter Norvig, 2nd edition,
PHI/Pearson
References:
Artificial Intelligence – Riche &K.Night , 2nd edition, TMH."""
                ])


#TC - 3
contextual_precision.measure(tc3)
print("Contextual Precision Score: ", contextual_precision.score)
print("Reason: ", contextual_precision.reason)

contextual_recall.measure(tc3)
print("Contextual Recall Score: ", contextual_recall.score)
print("Reason: ", contextual_recall.reason)

contextual_relevancy.measure(tc3)
print("Contextual Relevancy Score: ", contextual_relevancy.score)
print("Reason: ", contextual_relevancy.reason)

answer_relevancy.measure(tc3)
print("Answer Relevancy Score: ", answer_relevancy.score)
print("Reason: ", answer_relevancy.reason)

faithfulness.measure(tc3)
print("Faithfulness Score: ", faithfulness.score)
print("Reason: ", faithfulness.reason)