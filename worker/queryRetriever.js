import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";

const queryRetriver = async () => {
  const userQuery = "burning";

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_KEY,
  });

  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: "http://localhost:6333",
      // collection is a set of points where each point consists of a vector and its associated metadata
      collectionName: "langchainjs-testing", // this is the collection (equivalent to a traditional db table)
    },
  );

  // This method transforms the QdrantVectorStore instance into a retriever. A retriever is a component in Langchain that can fetch relevant documents based on a query.
  const ret = vectorStore.asRetriever({
    // k specifies the number of top relevant documents (or "chunks") that the retriever should return when it's invoked with a query. In this case, it's
    // set to 2, meaning it will retrieve the 2 most similar documents from the vector store.
    k: 2,
  });

  const result = await ret.invoke(userQuery);

  console.log(result);
};

queryRetriver();
