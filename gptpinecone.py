# pip install -qU langchain tiktoken tqdm
# pip install -qU openai pinecone-client datasets
# pip install langchain
# pip install unstructured
# pip install unstructured[local-inference]
# pip install pymupdf
# pip install openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from tqdm.autonotebook import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma, Pinecone
import pinecone
import os


loader = PyMuPDFLoader("C:/Users/aaron/Documents/Uni/contractlaw.pdf")
data = loader.load()
print(data[1].page_content)


print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[1].page_content)} characters in your document')


os.environ["OPENAI_API_KEY"] = "..."

print (f'Now you have {len(texts)} documents')

OPENAI_API_KEY = '...'
PINECONE_API_KEY = '...'
PINECONE_API_ENV = '...'

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "pdf-test"

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

query = "I have a breach in my contract specifically regarding consideration not being performed by one party. What would apply to my situation? Can you give me a very detailed answer?"
docs = docsearch.similarity_search(query, include_metadata=True)


chain.run(input_documents=docs, question=query)

# Output
# ' In this situation, the person seeking to enforce the contractual obligation
# may be able to sound in damages at the suit of the promise. This means that
# the person seeking to enforce the contractual obligation may be able to sue
# the other party for damages due to the breach of the contract. If damages are
# an inadequate remedy, the promisee may be able to obtain a decree of specific
# performance of the contract. However, the third party cannot institute an action
# for breach of contract or for specific performance unless they can bring
# themselves within one of the recognized exceptions or qualifications of the rules.
# In some cases, a breach of a given term may entitle the other party to terminate
# the contract, depending on the nature of the events to which the breach gives rise.'

