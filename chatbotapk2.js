import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { HumanMessage } from "@langchain/core/messages";
import express from "express";
import cors from "cors";
import fs from "fs";
import dotenv from "dotenv";

dotenv.config();

const DB_FAISS_PATH = "./vectorstore/db_faiss";

async function loadRetriever() {
  if (!fs.existsSync(DB_FAISS_PATH)) {
    throw new Error(`Vector database not found at ${DB_FAISS_PATH}`);
  }
  const embeddings = new OpenAIEmbeddings({ modelName: "text-embedding-3-small" });
  const vectorStore = await FaissStore.load(DB_FAISS_PATH, embeddings);
  return vectorStore.asRetriever({ k: 5 });
}

async function createChatbot() {
  const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });
  const retriever = await loadRetriever();

  async function getAnswer(question) {
    if (!question.trim()) return "";

    // Retrieve relevant documents
    const docs = await retriever.getRelevantDocuments(question);

    const contextText = docs.map(d => d.pageContent).join("\n\n");

    const systemPrompt = `You are an expert Q&A assistant. Your task is to answer the user's question accurately based on the provided document context. Keep the answer concise and relevant.

IMPORTANT GUIDELINES:
- Use only the context retrieved from the documents.
- If the retrieved context does not contain the answer, you MUST state that the information is not available in the provided documents.
- Do not use any outside knowledge or prior conversational context to answer the question.
- Greet the user warmly whenever they say hello, and follow up by asking what they're looking for.
- Format your answer clearly and concisely, using bullet points or short paragraphs if helpful.

Answer based solely on the document context provided.
Context:
${contextText}`;

    const messages = [
      new HumanMessage(systemPrompt),
      new HumanMessage(question)
    ];

    const response = await llm.invoke(messages);
    return response.content;
  }

  return { getAnswer };
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    console.error("Missing OPENAI_API_KEY");
    process.exit(1);
  }

  const chatbot = await createChatbot();

  const app = express();
  app.use(cors());
  app.use(express.json());

  app.post("/chat", async (req, res) => {
    const { question } = req.body;
    if (!question) {
      return res.status(400).json({ error: "Question is required." });
    }
    try {
      const answer = await chatbot.getAnswer(question);
      res.json({ answer });
    } catch (err) {
      console.error("Error getting answer:", err);
      res.status(500).json({ error: "Internal error" });
    }
  });

  const port = process.env.PORT || 3000;
  app.listen(port, () => {
    console.log(`Chatbot running at http://localhost:${port}`);
  });
}

main().catch(console.error);
