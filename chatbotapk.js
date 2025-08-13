import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { createRetrieverTool } from "langchain/tools/retriever";
import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { StateGraph, END } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import * as fs from "fs";
import dotenv from "dotenv";
import { fileURLToPath } from 'url';
import { resolve } from 'path';
import express from 'express';
import cors from 'cors';

// Load environment variables from .env file
dotenv.config();

// Path to the vector DB
const DB_FAISS_PATH = "./vectorstore/db_faiss";

/**
 * RAG Chatbot API Class
 */
class RAGChatbot {
    constructor(dbPath) {
        this.llm = new ChatOpenAI({
            modelName: "gpt-4o-mini",
            temperature: 0,
        });
        this.dbPath = dbPath;
        this.app = null;
    }

    async init() {
        try {
            this.app = await this._buildGraph(this.dbPath);
        } catch (error) {
            console.error("Error initializing chatbot:", error);
            throw error;
        }
    }

    async _buildGraph(dbPath) {
        if (!fs.existsSync(dbPath)) {
            throw new Error(`Vector database not found at ${dbPath}.`);
        }

        console.log("Loading vector database...");
        const embeddings = new OpenAIEmbeddings({ modelName: "text-embedding-3-small" });
        const vectorStore = await FaissStore.load(dbPath, embeddings);
        const retriever = vectorStore.asRetriever({ k: 5 });

        const tool = createRetrieverTool(retriever, {
            name: "retrieve_document_context",
            description: "Searches and returns relevant document context for a given query.",
        });

        const llmWithTools = this.llm.bindTools([tool]);

        const callModel = async (state) => {
            const systemPrompt = `You are an expert Q&A assistant. Your task is to answer the user's question accurately based on the provided document context.

IMPORTANT GUIDELINES:
- Use only the context retrieved from the documents provided in the tool's output.
- If the retrieved context does not contain the answer, you MUST state that the information is not available in the provided documents.
- Do not use any outside knowledge or prior conversational context to answer the question.
- Greet the user warmly whenever they say hello, and follow up by asking what they're looking for.
- Format your answer clearly and concisely, using bullet points or short paragraphs if helpful.

Answer based solely on the document context provided.`;

            const messagesWithPrompt = [new HumanMessage(systemPrompt), ...state.messages];
            const response = await llmWithTools.invoke(messagesWithPrompt);
            return { messages: [response] };
        };

        const callToolNode = async (state) => {
            const lastMessage = state.messages[state.messages.length - 1];
            const toolCall = lastMessage.tool_calls[0];
            const toolOutput = await tool.invoke(toolCall.args);
            return { messages: [new ToolMessage({ content: String(toolOutput), tool_call_id: toolCall.id })] };
        };

        const workflow = new StateGraph({
            channels: {
                messages: { reducer: (x, y) => x.concat(y), default: () => [] },
            },
        });

        workflow.addNode("agent", callModel);
        workflow.addNode("action", callToolNode);

        workflow.setEntryPoint("agent");
        workflow.addConditionalEdges("agent",
            (state) => {
                const lastMessage = state.messages[state.messages.length - 1];
                return !(lastMessage instanceof AIMessage) || !lastMessage.tool_calls?.length ? "end" : "continue";
            },
            { continue: "action", end: END }
        );
        workflow.addEdge("action", "agent");

        return workflow.compile({ checkpointer: new MemorySaver() });
    }

    async getAnswer(question, threadId = "api-conversation") {
        if (!question.trim()) return "";

        const config = { configurable: { thread_id: threadId } };
        const inputMessage = new HumanMessage(question);
        let finalAnswer = "";

        try {
            const stream = await this.app.stream({ messages: [inputMessage] }, config);

            // Improved streaming handling (same as chatbot.js)
            for await (const chunk of stream) {
                for (const [key, value] of Object.entries(chunk)) {
                    if (key === "agent" && value.messages?.length > 0) {
                        const aiMessage = value.messages[0];
                        if (aiMessage instanceof AIMessage && aiMessage.content) {
                            finalAnswer += aiMessage.content;
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Error during getAnswer:", error);
        }

        return finalAnswer;
    }
}

/**
 * Main function to start API server
 */
async function main() {
    if (!process.env.OPENAI_API_KEY) {
        console.error("Error: OPENAI_API_KEY environment variable not set.");
        process.exit(1);
    }

    try {
        const chatbot = new RAGChatbot(DB_FAISS_PATH);
        await chatbot.init();
        console.log("Chatbot initialized successfully.");

        const app = express();
        const port = process.env.PORT || 3000;

        app.use(cors());
        app.use(express.json());

        app.post('/chat', async (req, res) => {
            const { question, threadId } = req.body;

            if (!question) {
                return res.status(400).json({ error: 'Question is required.' });
            }

            console.log(`Received question: "${question}"`);

            try {
                const answer = await chatbot.getAnswer(question, threadId);
                res.json({ answer: answer });
            } catch (error) {
                console.error("API Chat Error:", error);
                res.status(500).json({ error: 'Failed to get a response from the chatbot.' });
            }
        });

        app.listen(port, () => {
            console.log(`Server is running at http://localhost:${port}`);
        });

    } catch (error) {
        console.error("Failed to start chatbot server:", error.message);
        process.exit(1);
    }
}

const __filename = fileURLToPath(import.meta.url);
if (resolve(process.argv[1]) === __filename) {
    main().catch(console.error);
}
