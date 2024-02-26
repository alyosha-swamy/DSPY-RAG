#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st

import openai
openai.api_key = st.secrets["OPENAI_API_KEY"]


# In[7]:


# Connect to Weaviate Retriever and configure LLM
import dspy
from dspy.retrieve.weaviate_rm import WeaviateRM
import weaviate
import openai


llm = dspy.OpenAI(model="gpt-3.5-turbo")

# ollamaLLM = dspy.OpenAI(api_base="http://localhost:11434/v1/", api_key="ollama", model="mistral-7b-instruct-v0.2-q6_K", stop='\n\n', model_type='chat')
# Thanks Knox! https://twitter.com/JR_Knox1977/status/1756018720818794916/photo/1

weaviate_client = weaviate.Client("http://localhost:8080")
retriever_model = WeaviateRM("WeaviateBlogChunk", weaviate_client=weaviate_client)
# Assumes the Weaviate collection has a text key `content`
dspy.settings.configure(lm=llm, rm=retriever_model)


# In[2]:


# Load FAQs
import re

f = open("faq.md")
markdown_content = f.read()

def parse_questions(markdown_content):
    # Regular expression pattern for finding questions
    question_pattern = r'#### Q: (.+?)\n'

    # Finding all questions
    questions = re.findall(question_pattern, markdown_content, re.DOTALL)

    return questions

# Parsing the markdown content to get only questions
questions = parse_questions(markdown_content)

# Displaying the first few extracted questions
questions[:5]  # Displaying only the first few for brevity


# In[2]:



# In[3]:


import dspy

# ToDo, add random splitting -- maybe wrap this entire thing in a cross-validation loop
trainset = questions[:20] # 20 examples for training
devset = questions[20:30] # 10 examples for development
testset = questions[30:] # 14 examples for testing

trainset = [dspy.Example(question=question).with_inputs("question") for question in trainset]
devset = [dspy.Example(question=question).with_inputs("question") for question in devset]
testset = [dspy.Example(question=question).with_inputs("question") for question in testset]


# In[4]:


devset[0]


# In[5]:


# This is a WIP, the next step is to optimize this metric as itself a DSPy module (pretty meta)

# Reference - https://github.com/stanfordnlp/dspy/blob/main/examples/tweets/tweet_metric.py

metricLM = dspy.OpenAI(model='gpt-4', max_tokens=1000, model_type='chat')

# Signature for LLM assessments.

class Assess(dspy.Signature):
    """Assess the quality of an answer to a question."""
    
    context = dspy.InputField(desc="The context for answering the question.")
    assessed_question = dspy.InputField(desc="The evaluation criterion.")
    assessed_answer = dspy.InputField(desc="The answer to the question.")
    assessment_answer = dspy.OutputField(desc="A rating between 1 and 5. Only output the rating and nothing else.")

def llm_metric(gold, pred, trace=None):
    predicted_answer = pred.answer
    question = gold.question
    
    print(f"Test Question: {question}")
    print(f"Predicted Answer: predicted_answer")
    
    detail = "Is the assessed answer detailed?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    overall = f"Please rate how well this answer answers the question, `{question}` based on the context.\n `{predicted_answer}`"
    
    with dspy.context(lm=metricLM):
        context = dspy.Retrieve(k=5)(question).passages
        detail = dspy.ChainOfThought(Assess)(context="N/A", assessed_question=detail, assessed_answer=predicted_answer)
        faithful = dspy.ChainOfThought(Assess)(context=context, assessed_question=faithful, assessed_answer=predicted_answer)
        overall = dspy.ChainOfThought(Assess)(context=context, assessed_question=overall, assessed_answer=predicted_answer)
    
    print(f"Faithful: {faithful.assessment_answer}")
    print(f"Detail: {detail.assessment_answer}")
    print(f"Overall: {overall.assessment_answer}")
    
    
    total = float(detail.assessment_answer) + float(faithful.assessment_answer)*2 + float(overall.assessment_answer)
    
    return total / 5.0


# In[8]:


test_example = dspy.Example(question="What drives Tesla's innovation in electric vehicles?")
test_pred = dspy.Example(answer="Pushing boundaries for sustainable transport solutions")

type(llm_metric(test_example, test_pred))


# In[10]:


metricLM.inspect_history(n=3)


# In[11]:


class GenerateAnswer(dspy.Signature):
    """Answer questions based on the context."""
    
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField()


# In[21]:


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.rerank = dspy.Predict("question, context -> reranked_context")
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        context = self.rerank(question=question, context=context).reranked_context
        prediction = self.generate_answer(context=context, question=question)

        return dspy.Prediction(answer=prediction.answer)


# In[22]:


dspy.Predict(GenerateAnswer)(question="What drives Tesla's innovation in electric vehicles?")
llm.inspect_history(n=1)


# In[14]:


dspy.ChainOfThought(GenerateAnswer)(question="What drives Tesla's innovation in electric vehicles?")
llm.inspect_history(n=1)


# In[15]:


dspy.ReAct(GenerateAnswer, tools=[dspy.settings.rm])(question="What drives Tesla's innovation in electric vehicles?")


# In[16]:


llm.inspect_history(n=1)


# In[17]:


uncompiled_rag = RAG()


# In[19]:


print(uncompiled_rag("What drives Tesla's innovation in electric vehicles ").answer)


# In[20]:


from dspy.evaluate.evaluate import Evaluate

evaluate = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)

evaluate(RAG(), metric=llm_metric)


# In[26]:


from dspy.teleprompt import BootstrapFewShot

teleprompter = BootstrapFewShot(metric = llm_metric, max_labeled_demos=8, max_rounds=3)
compiled_rag = teleprompter.compile(uncompiled_rag, trainset=trainset)


# In[30]:


from dspy.teleprompt import BayesianSignatureOptimizer

llm_prompter = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=1000, model_type="chat")
teleprompter = BayesianSignatureOptimizer(task_model=dspy.settings.lm,
                                        metric = llm_metric,
                                        prompt_model=llm_prompter,
                                        n=5,
                                        verbose=False)
kwargs = dict(num_threads=1, display_progress=True, display_table=0)
third_compiled_rag = teleprompter.compile(RAG(), devset=devset,
                                optuna_trials_num=3,
                                max_bootstrapped_demos=4,
                                max_labeled_demos=4,
                                eval_kwargs=kwargs)


# In[33]:


get_ipython().run_line_magic('pip', 'install streamlit')


# In[31]:


import streamlit as st

# Add a title
st.title("My Jupyter Notebook")

# Import the converted Python script here
import test.ipynb

# Run the content of the converted Python script
converted_script.run()

