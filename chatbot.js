import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { createRetrieverTool } from "langchain/tools/retriever";
import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { StateGraph, END } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import * as fs from "fs";
import * as readline from "readline";
import dotenv from "dotenv";
import { fileURLToPath } from 'url';
import { resolve } from 'path';

// Load environment variables from .env file
dotenv.config();

// Define the path where the vector database is saved
const DB_FAISS_PATH = "./vectorstore/db_faiss";

/**
 * RAG Chatbot class using LangGraph and OpenAI
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
            // Re-throw the error to be caught by the main execution block
            throw error;
        }
    }

    /**
     * Builds the conversational graph for the chatbot
     */
    async _buildGraph(dbPath) {
        // Check if database exists
        if (!fs.existsSync(dbPath)) {
            throw new Error(`Vector database not found at ${dbPath}. Please run create-db.js first.`);
        }

        // --- 1. Load the Vector DB ---
        console.log("Loading vector database...");
        const embeddings = new OpenAIEmbeddings({
            modelName: "text-embedding-3-small",
        });

        const vectorStore = await FaissStore.load(dbPath, embeddings);
        const retriever = vectorStore.asRetriever({
            k: 5, // Return top 5 most relevant chunks
        });

        // --- 2. Create the Retriever Tool ---
        const tool = createRetrieverTool(retriever, {
            name: "retrieve_document_context",
            description: "Searches and returns relevant document context for a given query.",
        });

        const tools = [tool];
        const llmWithTools = this.llm.bindTools(tools);

        // --- 3. Define Graph Nodes ---
        const shouldContinue = (state) => {
            const lastMessage = state.messages[state.messages.length - 1];
            if (!(lastMessage instanceof AIMessage) || !lastMessage.tool_calls?.length) {
                return "end";
            }
            return "continue";
        };

        const callModel = async (state) => {
            const messages = state.messages;
            const systemPrompt = `You are an expert Q&A assistant. Your task is to answer the user's question accurately based on the provided document context.

IMPORTANT GUIDELINES:
- Use only the context retrieved from the documents provided in the tool's output. If there is no relevant context for the user's query, simply respond that there is not enough information available.
- Do not use any outside knowledge or prior conversational context to answer the question. 
- Format your answer clearly and concisely. 
- If the retrieved context does not contain the answer, you MUST state that the information is not available in the provided documents.
- Greet the user warmly whenever they say hello, and follow up by asking what they're looking for.

Answer based solely on the document context provided.`;

            const messagesWithPrompt = [
                new HumanMessage(systemPrompt),
                ...messages
            ];

            const response = await llmWithTools.invoke(messagesWithPrompt);
            return { messages: [response] };
        };

        const callToolNode = async (state) => {
            const lastMessage = state.messages[state.messages.length - 1];
            const toolCall = lastMessage.tool_calls[0];
            const toolOutput = await tool.invoke(toolCall.args);
            
            return {
                messages: [
                    new ToolMessage({
                        content: String(toolOutput),
                        tool_call_id: toolCall.id,
                    }),
                ],
            };
        };

        // --- 4. Construct the Graph ---
        const workflow = new StateGraph({
            channels: {
                messages: {
                    reducer: (x, y) => x.concat(y),
                    default: () => [],
                },
            },
        });

        workflow.addNode("agent", callModel);
        workflow.addNode("action", callToolNode);

        workflow.setEntryPoint("agent");
        workflow.addConditionalEdges("agent", shouldContinue, {
            continue: "action",
            end: END,
        });
        workflow.addEdge("action", "agent");

        // --- 5. Add Memory ---
        const memory = new MemorySaver();
        return workflow.compile({ checkpointer: memory });
    }

    /**
     * Chat with the RAG chatbot
     */
    async chat(question, threadId = "main-conversation") {
        if (!question.trim()) {
            return;
        }

        const config = {
            configurable: { thread_id: threadId },
        };

        const inputMessage = new HumanMessage(question);

        try {
            let finalAnswer = "";
            
            // Stream the response from the chatbot
            const stream = await this.app.stream(
                { messages: [inputMessage] },
                config
            );

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

            if (finalAnswer) {
                console.log(`\n AI: ${finalAnswer}\n`);
            }
        } catch (error) {
            console.error("Error during chat:", error);
            console.log(" Please try again or check your question.\n");
        }
    }

    /**
     * Start interactive chat session
     */
    async startChat() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        const conversationThreadId = "main-conversation";
        
        console.log("--- RAG Chatbot ---");
        console.log("Ask questions based on your documents. Type 'exit' to end.\n");

        const askQuestion = () => {
            rl.question("You: ", async (userInput) => {
                if (userInput.toLowerCase() === "exit") {
                    console.log("Ending conversation. Goodbye!");
                    rl.close();
                    return;
                }

                await this.chat(userInput, conversationThreadId);
                askQuestion();
            });
        };

        askQuestion();
    }
}

/**
 * Main function to run the chatbot
 */
async function main() {
    // Check if API key is set
    if (!process.env.OPENAI_API_KEY) {
        console.error("Error: OPENAI_API_KEY environment variable not set.");
        console.log("Please set your OpenAI API key in the .env file or environment variables.");
        process.exit(1);
    }

    try {
        // 1. Create an instance of the chatbot. This is now a synchronous operation.
        const chatbot = new RAGChatbot(DB_FAISS_PATH);
        
        // 2. Await its asynchronous initialization. This ensures the DB is loaded
        //    and the graph is compiled before proceeding.
        await chatbot.init();

        // 3. Start the interactive chat session. This runs only after init() is successful.
        await chatbot.startChat();

    } catch (error) {
        // Errors from init() or startChat() will be caught here.
        console.error("Error starting chatbot:", error.message);
        process.exit(1);
    }
}

// Run the main function if this file is executed directly
const __filename = fileURLToPath(import.meta.url);
if (resolve(process.argv[1]) === __filename) {
    main().catch(console.error);
}

export { RAGChatbot };