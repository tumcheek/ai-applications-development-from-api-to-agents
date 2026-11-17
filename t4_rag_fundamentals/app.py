import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from commons.constants import OPENAI_API_KEY, OPENAI_EMBEDDINGS_MODEL, OPENAI_TERRA_MODEL

#TODO:
# Create system prompt with:
# - role: explains the role for LLM and what it should do
# - Structure of User message, consists of 2 blocks:
#   - `RAG CONTEXT`: information retrieved on the Retrieval step based on user request
#   - `USER QUESTION`: The user's actual question
# - Instructions:
#   - Model must use only information from conversation
#   - Strictly forbid to answer questions that are not in the conversation or not present in `RAG CONTEXT`
_SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

_USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """
        Load existing FAISS index from disk or create a new one.
        Returns:
              VectorStore: Initialized FAISS vectorstore.
        """
        #TODO:
        # - Print a startup message
        # - Check if 'microwave_faiss_index' folder already exists
        # - If yes, load the index from disk using FAISS.load_local()
        # - If no, call _create_new_index() to build and save a fresh index
        # - Return the vectorstore
        print("Setup vector store")
        index_name = "microwave_faiss_index"
        index_path = f"t4_rag_fundamentals\{index_name}"
        if os.path.isdir(index_path):
            print("Index exists")
            index = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Create new index")
            index = self._create_new_index()
        return index

    def _create_new_index(self) -> VectorStore:
        """
        Load the manual, split into chunks, embed, and save a new FAISS index.
        Returns:
              VectorStore: Newly created and saved FAISS vectorstore.
        """
        #TODO:
        # - Load 'microwave_manual.txt' using TextLoader
        # - Split documents into chunks using RecursiveCharacterTextSplitter
        #   (chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "."])
        # - Create a FAISS vectorstore from chunks and self.embeddings using FAISS.from_documents()
        # - Save the index locally using vectorstore.save_local("microwave_faiss_index")
        # - Return the vectorstore
        path = "t4_rag_fundamentals\\microwave_manual.txt"
        index_name = "microwave_faiss_index"
        index_path = f"t4_rag_fundamentals\{index_name}"
        loader = TextLoader(path, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", "."])
        chunks = text_splitter.split_documents(documents)
        for c in chunks:
            c.page_content = f"Microwave Oven DW 395 HCG manual\n{c.page_content}"

        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(index_path)
        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.2):
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        #TODO:
        # - Search the vectorstore using similarity_search_with_relevance_scores() with k and score_threshold parameters
        # - Iterate over results, collect each doc's page_content, and print its relevance score
        # - Return all collected chunks joined with "\n\n" as a single context string
        searches = self.vectorstore.similarity_search_with_relevance_scores(query, k, score_threshold=score)
        contetn = []
        for document, doc_score in searches:
            print("Score:", doc_score)
            print(document.page_content)
            contetn.append(document.page_content)
        return "\n\n".join(contetn)

    def augment_prompt(self, query: str, context: str):
        """
        Inject retrieved context and user query into the prompt template.
        Args:
              query (str): The user's question.
              context (str): Retrieved context from the vectorstore.
        Returns:
              str: Formatted prompt ready for the LLM.
        """
        #TODO:
        # - Format _USER_PROMPT template substituting {context} and {query}
        # - Print the resulting augmented prompt
        # - Return the formatted string
        augmented_prompt = _USER_PROMPT.format(query=query, context=context)
        print(augmented_prompt)
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str):
        """
        Send the augmented prompt to the LLM and return its response.
        Args:
              augmented_prompt (str): The prompt with injected context and query.
        Returns:
              str: The LLM-generated answer.
        """
        #TODO:
        # - Build a messages list: [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=augmented_prompt)]
        # - Invoke self.llm_client with the messages list
        # - Print the response content
        # - Return the response content string
        messages =[SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=augmented_prompt)]
        response = self.llm_client.invoke(messages)

        return response.content


def main(rag: MicrowaveRAG):
    #TODO:
    # - Print a welcome message
    # - Run an infinite loop that reads user input with input()
    # - For each question execute the 3-step RAG pipeline:
    #   - Step 1 (Retrieval):   call rag.retrieve_context() to fetch relevant chunks
    #   - Step 2 (Augmentation): call rag.augment_prompt() to build the prompt
    #   - Step 3 (Generation):  call rag.generate_answer() to get the LLM answert
    print("Welcome to rag")
    while True:
        user_input = input("User:")
        if user_input == "exit":
            break
        retrived_text = rag.retrieve_context(user_input)
        augmented_prompt = rag.augment_prompt(user_input, retrived_text)
        response = rag.generate_answer(augmented_prompt)
        print("AI:", response)


#TODO:
# Start the application by calling main() and passing a MicrowaveRAG instance:
# - Create OpenAIEmbeddings with model=OPENAI_EMBEDDINGS_MODEL and api_key=OPENAI_API_KEY
# - Create ChatOpenAI with temperature=0.0, reasoning_effort="none", model=OPENAI_TERRA_MODEL and api_key=OPENAI_API_KEY
# - Wrap both in a MicrowaveRAG instance and pass it to main()

embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL, api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.0, reasoning_effort="none", model=OPENAI_TERRA_MODEL, api_key=OPENAI_API_KEY)
rag = MicrowaveRAG(embeddings, llm)
main(rag)