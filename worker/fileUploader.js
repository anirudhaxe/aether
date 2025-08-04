import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// Documents (unit of text and associated metadata) and document loaders;
// Text splitters;
// Embeddings;
// Vector stores and retrievers.
const uploadFile = async () => {
  // load the pdf
  // create the embeddings
  // store the embeddings in the vector DB

  // load the PDF with document loader
  const loader = new PDFLoader("./data/brain.pdf");

  // .load() returns documents of type:
  // const documents = [
  //   new Document({
  //     pageContent:
  //       "Dogs are great companions, known for their loyalty and friendliness.",
  //     metadata: { source: "mammal-pets-doc" },
  //   }),
  //   new Document({
  //     pageContent: "Cats are independent pets that often enjoy their own space.",
  //     metadata: { source: "mammal-pets-doc" },
  //   }),
  // ];
  const docs = await loader.load();

  // This splitter will recursively split the document using common separators like new lines until each chunk is of appropriate size.
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000, // split in chunks of 1000 characters
    chunkOverlap: 200, // 200 characters overlap between the chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it.
  });

  const allSplitedDocs = await textSplitter.splitDocuments(docs);

  // initialize embeddings model instance
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_KEY,
  });

  // initialize the vector store by passing the embedding model instance
  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: "http://localhost:6333",
      collectionName: "brain-rust-data",
    },
  );

  // add the vector embeddings to the vector store
  await vectorStore.addDocuments(docs);

  console.log("Document added to vector store, semantic search engine ready");
};

uploadFile();
