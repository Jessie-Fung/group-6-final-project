
**Key takeaways**
- use langchain workflow makes things much cleaner
- [use hybrid search ](https://python.langchain.com/docs/how_to/hybrid/)
- Sub-Queries: There are different query strategies such as tree queries or sequential querying of chunks that can be used for different scenario
- Try out **Hypothetical Document Embedding
- Use advanced evaluation (see bottom of note)



**Use Langchain or Lamaindex framework**
https://medium.com/@tam.tamanna18/langchain-vs-llamaindex-a-comprehensive-comparison-for-retrieval-augmented-generation-rag-0adc119363fe
```
# IN langchain Chain them together  
chain = SimpleSequentialChain(chains=[gpt_chain, bert_chain])  
result = chain.run("What is the future of AI?")  
print(result)

from langchain.retrievers.multi_retriever import MultiRetriever  
from langchain.retrievers import FAISSRetriever, KeywordRetriever  
  
# Define keyword and embedding-based retrieval systems  
keyword_retriever = KeywordRetriever(documents=legal_documents)  
embedding_retriever = FAISSRetriever(index=scientific_index)  
  
# Combine them in a MultiRetriever  
retriever = MultiRetriever(retrievers={  
'legal': keyword_retriever,  
'science': embedding_retriever  
})  
  
# Query the retriever with a legal question  
response = retriever.retrieve("What are the recent changes in contract law?")  
print(response)
```


**Advanced vs. Nieve Rag**
https://www.promptingguide.ai/research/rag
Optimizing post-retrieval focuses on avoiding context window limits and dealing with noisy or potentially distracting information. A common approach to address these issues is re-ranking which could involve approaches such as relocation of relevant context to the edges of the prompt or recalculating the semantic similarity between the query and relevant text chunks. Prompt compression may also help in dealing with these issues.

Modular RAG
- **Hybrid Search Exploration:** This approach leverages a combination of search techniques like keyword-based search and semantic search to retrieve relevant and context-rich information; this is useful when dealing with different query types and information needs.
- - **Recursive Retrieval and Query Engine:** Involves a recursive retrieval process that might start with small semantic chunks and subsequently retrieve larger chunks that enrich the context; this is useful to balance efficiency and context-rich information.
- - **StepBack-prompt:** [A prompting technique(opens in a new tab)](https://arxiv.org/abs/2310.06117) that enables LLMs to perform abstraction that produces concepts and principles that guide reasoning; this leads to better-grounded responses when adopted to a RAG framework because the LLM moves away from specific instances and is allowed to reason more broadly if needed.
- - **Sub-Queries:** There are different query strategies such as tree queries or sequential querying of chunks that can be used for different scenarios. LlamaIndex offers a [sub question query engine(opens in a new tab)](https://docs.llamaindex.ai/en/latest/understanding/putting_it_all_together/agents.html#) that allows a query to be broken down into several questions that use different relevant data sources.
- - **Hypothetical Document Embeddings:** [HyDE(opens in a new tab)](https://arxiv.org/abs/2212.10496) generates a hypothetical answer to a query, embeds it, and uses it to retrieve documents similar to the hypothetical answer as opposed to using the query directly.
- **Iterative retrieval** enables the model to perform multiple retrieval cycles to enhance the depth and relevance of information. Notable approaches that leverage this method include [RETRO(opens in a new tab)](https://arxiv.org/abs/2112.04426) and [GAR-meets-RAG](https://arxiv.org/abs/2310.20158)

##Rag Evaluation
Evaluating a RAG framework focuses on three primary quality scores and four abilities. Quality scores include measuring context relevance (i.e., the precision and specificity of retrieved context), answer faithfulness (i.e., the faithfulness of answers to the retrieved context), and answer relevance (i.e., the relevance of answers to posed questions). In addition, there are four abilities that help measure the adaptability and efficiency of a RAG system: noise robustness, negative rejection, information integration, and counterfactual robustness. Below is a summary of metrics used for evaluating different aspects of a RAG system:
![[Pasted image 20250521162755.png]]