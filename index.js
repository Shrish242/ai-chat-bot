import { GoogleGenAI, Type } from "@google/genai";
import express from 'express';
import bodyParser from 'body-parser';
import * as dotenv from 'dotenv';

// --- SETUP ---
dotenv.config();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY not found. Please check your .env file.");
}

// NOTE: Using the genai library now, which handles the correct API structure.
const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
const app = express();
const PORT = 8000;
const MODEL = 'gemini-2.5-flash';

// Middleware
app.use(bodyParser.json());
// CORS setup for your future frontend widget
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*'); // Change * to your frontend domain in production
    res.header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    next();
});

// --- MOCK KNOWLEDGE BASE (RAG) ---
// This simulates your Vector Database and keyword-tagged documents.
const KNOWLEDGE_CHUNKS = [
    {
        text: "Our standard shipping takes 5-7 business days within the country. Express shipping is available for a flat rate of $15.",
        keywords: ["shipping", "delivery", "cost", "express"]
    },
    {
        text: "Returns are accepted within 30 days of purchase, provided the item is unworn and has original tags. Refunds are processed within 10 days.",
        keywords: ["returns", "refunds", "policy", "30 days"]
    },
    {
        text: "The main support hours are Mon-Fri, 9 AM to 5 PM EST. Outside of these hours, the AI agent handles all inquiries.",
        keywords: ["hours", "support", "staff"]
    },
];

/**
 * Mocks RAG retrieval by finding relevant chunks based on query keywords.
 * @param {string} query The user's message.
 * @returns {string} Combined relevant context chunks.
 */
function retrieveContext(query) {
    const queryLower = query.toLowerCase();
    const relevantChunks = KNOWLEDGE_CHUNKS.filter(chunk =>
        chunk.keywords.some(kw => queryLower.includes(kw)) ||
        chunk.text.toLowerCase().includes(queryLower)
    ).map(chunk => chunk.text);

    if (relevantChunks.length > 0) {
        console.log(`[RAG] Retrieved ${relevantChunks.length} context chunks.`);
        return relevantChunks.join("\n---\n");
    }
    return "No specific internal knowledge found. Answer using only general knowledge if possible.";
}

// --- MOCK EXTERNAL TOOLS (FUNCTION CALLING) ---

// 1. Tool Declaration (Schema for the model)
const TOOL_DECLARATIONS = [
    {
        name: 'book_appointment',
        description: 'Books a service appointment for the customer. Requires a date and the type of service.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                date: { type: Type.STRING, description: 'The desired date for the appointment (e.g., "next Tuesday").' },
                service: { type: Type.STRING, description: 'The specific service the customer wants (e.g., "haircut", "plumbing repair").' }
            },
            required: ["date", "service"],
        },
    },
    {
        name: 'check_order_status',
        description: 'Retrieves the current shipping status of a customer\'s order. Requires an order ID.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                order_id: { type: Type.STRING, description: 'The customer\'s unique order identification number.' },
            },
            required: ["order_id"],
        },
    },
];

// 2. Tool Mapping (Actual execution function)
const TOOL_FUNCTIONS = {
    'book_appointment': ({ date, service }) => {
        // MOCK API CALL: In production, this would use a library like 'axios'
        // to call an external scheduling API (e.g., Calendly, internal CRM).
        if (service.toLowerCase().includes("urgent")) {
            return `Sorry, all urgent slots are full. Please try booking for a later date than ${date}.`;
        }
        return `✅ Appointment for '${service}' successfully confirmed on ${date}. A text alert has been sent.`;
    },
    'check_order_status': ({ order_id }) => {
        // MOCK API CALL: In production, this would call an e-commerce API (e.g., Shopify, WooCommerce).
        if (order_id === "ABC-123") {
            return "Order ABC-123 is currently 'In Transit' and expected to arrive tomorrow, December 8, 2025.";
        }
        return `❌ Sorry, order ${order_id} was not found in our system. Please double-check the ID.`;
    },
};

// --- CORE CHAT ENDPOINT ---

app.post('/chat', async (req, res) => {
    const { user_query } = req.body;
    if (!user_query) {
        return res.status(400).json({ error: "Missing 'user_query' in request body." });
    }

    try {
        // 1. Retrieval (RAG)
        const context = retrieveContext(user_query);

        const systemInstruction = (
            "You are a helpful and professional customer support agent for a small business. " +
            "Your goal is to answer questions accurately and quickly. " +
            "Always prioritize the information provided in the 'CONTEXT' section. " +
            "If the user asks about an appointment or order status, use the available tools."
        );

        const fullPrompt = (
            `CONTEXT:\n---\n${context}\n---\n\n` +
            `CUSTOMER QUESTION: ${user_query}`
        );
        
        // 2. First API Call: Request generation with tools and the system instruction correctly placed in config
        let chatHistory = [
            // The system instruction is moved to the config object below.
            { role: "user", parts: [{ text: fullPrompt }] }
        ];

        let response = await ai.models.generateContent({
            model: MODEL,
            contents: chatHistory,
            config: {
                tools: [{ functionDeclarations: TOOL_DECLARATIONS }],
                // === FIX APPLIED HERE ===
                systemInstruction: systemInstruction, 
            }
        });

        let finalResponseText = response.text;
        let actionTaken = null;

        // 3. Handle Function Calls
        if (response.functionCalls && response.functionCalls.length > 0) {
            console.log("Function call detected. Executing tool...");
            
            // Handle the first function call (or parallel calls in a loop for advanced use)
            const call = response.functionCalls[0];
            const funcName = call.name;
            const funcArgs = call.args;

            const funcToCall = TOOL_FUNCTIONS[funcName];
            
            if (funcToCall) {
                // Execute the local tool function (MOCK API call)
                const toolOutput = funcToCall(funcArgs);
                actionTaken = `${funcName}(${JSON.stringify(funcArgs)}) -> ${toolOutput}`;
                
                // Add the function call and its result to history
                // Note: The model's response structure in the SDK may vary slightly;
                // ensuring the role is 'model' and parts include the functionCall.
                chatHistory.push(response.candidates[0].content); // AI's call
                chatHistory.push({
                    role: "function", 
                    parts: [{ 
                        functionResponse: { 
                            name: funcName, 
                            response: { result: toolOutput } 
                        } 
                    }] 
                });

                // 4. Second API Call: Get the final, human-readable response
                const secondResponse = await ai.models.generateContent({
                    model: MODEL,
                    contents: chatHistory,
                    config: {
                        tools: [{ functionDeclarations: TOOL_DECLARATIONS }],
                        // === FIX APPLIED HERE ===
                        systemInstruction: systemInstruction, 
                    }
                });

                finalResponseText = secondResponse.text;
            } else {
                finalResponseText = `Error: The model requested an unknown function: ${funcName}.`;
            }
        }

        // 5. Send final response back to the client
        res.json({ 
            ai_response: finalResponseText, 
            action_taken: actionTaken
        });

    } catch (error) {
        // Added check to distinguish between API errors and other errors
        if (error.status === 400 && error.message.includes("valid role")) {
             console.error("Gemini/Server Error: Role Error (This should now be fixed):", error);
        } else {
            console.error("Gemini/Server Error:", error);
        }
        
        res.status(500).json({ 
            ai_response: "I'm sorry, I'm currently having trouble connecting to my service. Please try again shortly.",
            action_taken: `Server Error: ${error.message}`
        });
    }
});

// --- START SERVER ---
app.listen(PORT, () => {
    console.log(`\n\n🚀 SME AI Agent Backend running at http://localhost:${PORT}`);
    console.log(`\nTo test, use a tool like cURL or Postman to send a POST request to http://localhost:${PORT}/chat`);
    console.log(`\nExample Payload: {"user_query": "I want to check order ABC-123"}\n`);
});