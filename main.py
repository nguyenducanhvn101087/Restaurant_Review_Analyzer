from langchain_ollama.llms import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever 

model = OllamaLLM(model="llama3.2", temperature=0.5)

template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain  = prompt | model # invoke the entire chain combining prompt and model here pass the prompt to the model

while True:
    print("\n\n---------------------------------------------------------------------------")
    question = input("Enter your question about a restaurant (or 'exit' to quit): ")
    print("\n\n")

    if question.lower() == 'exit':
        print('Exiting the program.')
        break

    reviews = retriever.invoke(question) # retrieve relevant reviews based on the question
    result = chain.invoke({"reviews": reviews, "question": question}) # pass reviews and question to the chain to get the answer
    print(result)