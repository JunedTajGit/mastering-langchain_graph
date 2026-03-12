from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

openai_chat_model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

def main():
    print("Hello from langc-course!")

    azure_llm =AzureChatOpenAI(
        model_name=openai_chat_model_name,
        temperature=0.0,
        streaming=False,
        cache=False,
    )
        
    result1 = azure_llm.invoke("Hi, Azure Open Ai")
    print(f"Azure: {result1.content}")


if __name__ == "__main__":
    main()
