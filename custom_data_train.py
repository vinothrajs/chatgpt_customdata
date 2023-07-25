import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def ask(user_question ,db):
    results = db.similarity_search(query=user_question,n_results=5)
    template = """
                You are a chat bot who loves to help people! Given the following context sections, answer the
                question using only the given context. If you are unsure and the answer is not
                explicitly writting in the documentation, say "Sorry, I don't know how to help with that."

                Context sections:
                {context}

                Question:
                {users_question}

                Answer:
                """

    prompt = PromptTemplate(template=template, input_variables=["context", "users_question"])
    prompt_text = prompt.format(context = results, users_question = user_question)
    chatgpt_api(prompt_text)

def chatgpt_api(prompt_text):
    openai.api_key  = "sk-xxxxxxxxxxxxxxxxxxx"
    for chunk in openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": prompt_text }],
      temperature=0,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stream=True,
      ## stop=[]
    ):
      content = chunk["choices"][0].get("delta", {}).get("content")
      if content is not None:
          print(content, end='')
          
def train_custom_data():
    
    # load the document
    with open('learning_data.text', encoding='utf-8') as f:
        text = f.read()

    # define the text splitter
    text_splitter = RecursiveCharacterTextSplitter(    
        chunk_size = 500,
        chunk_overlap  = 100,
        length_function = len,
    )
    data = text_splitter.create_documents([text])
    print("Training Started..")
    api_key  = "sk-xxxxxxxxxxxxxxxxxxxx"
    # define the embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # use the text chunks and the embeddings model to fill our vector store
    print("Creating Vector Store..")
    vectors  = FAISS.from_documents(data, embeddings)
    print("Training Completed..")
    return vectors 
          
if __name__ == "__main__":
    vectors  = train_custom_data()
    while True:
        user_question = input("Enter your question: ")
        print(user_question)
        ask(user_question, vectors)
        print('\n')