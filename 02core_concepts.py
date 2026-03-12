"""
LangChain Core Concepts - LCEL and Runnables
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
import os

load_dotenv()

openai_chat_model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    #To parse output into string 
    parser = StrOutputParser()

    # Compose with pipe operator
    #use pipe operator to run from left to rigjt
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}")

    return chain


def demo_batch_exectution():
    """Demonstrate batch execution for multiple inputs."""
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Batch - run with multiple inputs
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]
    
    #excute LLM against batch of input prompts
    results = chain.batch(inputs)

    for item in zip(inputs, results):
        print(f"Input: {item[0]['text']} => Output: {item[1]}")


def demo_streaming():
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming


def demo_schema_inspection():
    """Demonstrate input/output schema inspection."""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


def exercise_first_chain1():
    prompt = ChatPromptTemplate.from_template("write a marketing tagline for product '{product_name}' for the target audience '{target_audience}'")
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens= 10, timeout= 30, max_retries=3 )

    #provide specific parameters can be provided using model_kwargs=
    # model = AzureChatOpenAI(
    #     model_name=openai_chat_model_name,
    #     temperature=0.0,
    #     max_tokens= 10,
    #     streaming=False,
    #     cache=False,
    #     timeout= 30, 
    #     max_retries=3
    # )
    
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    
    result = chain.invoke({'product_name': "AI Course", 'target_audience': "developers"})
    
    print(f"response {result}")
    
# ------- Exercise the demos -------#
# Exercise: Build your first chain
def exercise_first_chain():
    """
    EXERCISE: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """

    # YOUR CODE HERE
    prompt = ChatPromptTemplate.from_template(
        "Create a marketing tagline for a product named '{product}' targeting '{audience}'."
    )
    # model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens= 10, timeout= 30, max_retries=3 )
    
    model =AzureChatOpenAI(
        model_name=openai_chat_model_name,
        temperature=0.0,
        max_tokens= 10,
        streaming=False,
        cache=False,
        timeout= 30, 
        max_retries=3
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Test the chain
    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Marketing Tagline: {result}")


def new_way():
    # the univeral way to initialize a model
    model = init_chat_model("gemini-2.5-flash", temperature=0.7, max_tokens=1500)
    
    
    # Or provider-specific (still works)

    model.invoke("Hi new universal way to initilized model")
    # from langchain_openai import ChatOpenAI
    # from langchain_anthropic import ChatAnthropic

    # openai_model = ChatOpenAI(model="gpt-4o-mini",
    #                           temperature=0.7,
    #                           max_tokens=1500,
    #                           timeout=30,
    #                           max_retries=3)
    
    # anthropic_model = ChatAnthropic(model="claude-sonnet-4-5-20250929")


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_exectution()
    # demo_streaming()
    # demo_schema_inspection()
    exercise_first_chain1()
    # new_way()
