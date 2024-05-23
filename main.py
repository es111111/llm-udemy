from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# 다른체인에 인풋하기위해
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task",default="return a list of numbers")
parser.add_argument("--language",default="python")
args = parser.parse_args()
## python manin.py --language javascript --task 'print hello';

llm = OpenAI()
# prompt 템플릿은 한 파일에서 관리해줘야 나중에 유지보수가 쉬움
code_propmt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language","task"]
)
test_propmt = PromptTemplate(
    input_variables=["language","code"],
    template="write a test for the following {language} code:\n{code}"
) 


code_chain = LLMChain(
    llm=llm,
    prompt=code_propmt,
    output_key="code"
)
# test-chain
test_chain = LLMChain(
    llm=llm,
    prompt=test_propmt,
    output_key="test"
)
#chain a 를 chain b에 꾸겨넣기 chain wired input 과 output 나열
wired_chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task","language"],
    output_variables=["test","code"],
)

result = wired_chain({
    "language": args.language,
    "task": args.task
})

print(">>>>>> GENERATED CODE:")
print(result["code"])

print(">>>>>> GENERATED TEST:")
print(result["test"])
