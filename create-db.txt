import { OpenAIEmbeddings } from "@langchain/openai";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import * as path from "path";
import mammoth from "mammoth";
import dotenv from "dotenv";

// Load environment variables from .env file
dotenv.config();

// --- CONFIGURATION ---
const DOCS_PATH = "./docs"; // Folder containing your .docx files
const DB_FAISS_PATH = "./vectorstore/db_faiss"; // Folder to save the vector store

/**
 * Extracts raw text from a .docx file.
 * @param {string} filePath - The path to the .docx file.
 * @returns {Promise<string>} The extracted text content.
 */
async function extractTextFromDocx(filePath) {
    try {
        const result = await mammoth.extractRawText({ path: filePath });
        return result.value;
    } catch (error) {
        console.error(`Error extracting text from ${filePath}:`, error);
        return ""; // Return empty string on error
    }
}

/**
 * Main function to generate and save the vector store.
 */
async function createVectorStore() {
    // --- 1. CHECK FOR API KEY ---
    if (!process.env.OPENAI_API_KEY) {
        console.error(" Error: OPENAI_API_KEY environment variable not set.");
        console.log("Please set your OpenAI API key in the .env file.");
        process.exit(1);
    }

    // --- 2. LOAD DOCUMENTS ---
    console.log("Loading documents from:", DOCS_PATH);
    const documents = [];
    try {
        const files = fs.readdirSync(DOCS_PATH).filter(file => file.endsWith('.docx'));

        if (files.length === 0) {
            console.error(` No .docx files found in the '${DOCS_PATH}' directory.`);
            process.exit(1);
        }

        console.log(`Found ${files.length} .docx file(s).`);

        for (const file of files) {
            const filePath = path.join(DOCS_PATH, file);
            const text = await extractTextFromDocx(filePath);
            if (text) {
                // We store the source file name in the metadata
                documents.push({
                    pageContent: text,
                    metadata: { source: file },
                });
            }
        }
        console.log("Documents loaded successfully.");
    } catch (error) {
        console.error(" Error loading documents:", error.message);
        process.exit(1);
    }


    // --- 3. SPLIT DOCUMENTS INTO CHUNKS ---
    console.log("\nSplitting documents into smaller chunks...");
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000, // The maximum size of each chunk
        chunkOverlap: 200, // The number of characters to overlap between chunks
    });
    const docs = await textSplitter.splitDocuments(documents);
    console.log(`Documents split into ${docs.length} chunks.`);


    // --- 4. CREATE EMBEDDINGS ---
    console.log("\n Creating embeddings for the document chunks...");
    const embeddings = new OpenAIEmbeddings({
        modelName: "text-embedding-3-small",
    });


    // --- 5. CREATE AND SAVE FAISS VECTOR STORE ---
    try {
        console.log("\n Creating and saving the FAISS vector store...");
        // Ensure the target directory exists
        if (fs.existsSync(DB_FAISS_PATH)) {
            console.warn(`Warning: Existing vector store found at ${DB_FAISS_PATH}. It will be overwritten.`);
            fs.rmSync(DB_FAISS_PATH, { recursive: true, force: true });
        }
        
        fs.mkdirSync(path.dirname(DB_FAISS_PATH), { recursive: true });

        // The static `fromDocuments` method creates a new FaissStore from the documents.
        const vectorStore = await FaissStore.fromDocuments(docs, embeddings);
        
        // Save the vector store to the specified path.
        await vectorStore.save(DB_FAISS_PATH);
        
        console.log("\n✨ Vector store created and saved successfully at:", DB_FAISS_PATH);

    } catch (error) {
        console.error(" Error creating or saving the vector store:", error);
        process.exit(1);
    }
}

// Run the main function
createVectorStore();
