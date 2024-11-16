from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
import os

template = """
You are an expert summarization AI. Your role is to create well-structured HTML summaries that are clean and semantic. 

## Instructions
1. Create a summary with the following sections:
   - Main Summary (brief overview)
   - Key Points
   - Important Details
   - Takeaways

2. Use semantic HTML elements for structure. The response should follow this format:

<article>
    <section>
        <h2>Main Summary</h2>
        <p>[Concise overview here]</p>
    </section>

    <section>
        <h2>Key Points</h2>
        <ul>
            <li>[Key point 1]</li>
            <li>[Key point 2]</li>
        </ul>
    </section>

    <section>
        <h2>Important Details</h2>
        <h3>Context</h3>
        <ul>
            <li>[Detail 1]</li>
        </ul>
    </section>

    <section>
        <h2>Key Takeaways</h2>
        <ul>
            <li>[Takeaway 1]</li>
        </ul>
    </section>
</article>

## Input Content to Summarize:
{content}

Ensure the summary is comprehensive yet concise, with proper semantic HTML structure throughout.
"""

llm = AzureChatOpenAI(
    openai_api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API"),
    deployment_name="gpt-4",
    temperature=0.3,
    max_retries=1,
    max_tokens=900,
    model_version="turbo-2024-04-09",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI specialized in creating structured HTML summaries. Your task is to transform complex information into clear, concise, and well-organized summaries using semantic HTML. Maintain a neutral and informative tone, ensuring that each section is distinct and logically ordered. Prioritize clarity and coherence in your summaries."),
    ("human", template),
])

def get_summary(content: str) -> str:
    try:
        input_data = {"content": content}
        response = (
            prompt
            | llm.bind(stop=["\nsummarization"])
            | StrOutputParser()
        )
        return response.invoke(input_data)
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return None