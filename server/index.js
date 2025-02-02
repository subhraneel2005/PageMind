import axios from "axios";
import * as cheerio from "cheerio";
import OpenAI from "openai";
import dotenv from "dotenv";
import { ChromaClient } from "chromadb";

dotenv.config();

const openai = new OpenAI({});
const chromaClient = new ChromaClient({
  path: "http://localhost:8000",
});
chromaClient.heartbeat();
const WEB_COLLECTION = "PAGEMIND_COLLECTION_1";

async function scrapeWebpage(url = "") {

  const { data } = await axios.get(url);
  const $ = cheerio.load(data);

  const pageMetadata = $("head").html();
  const pageBody = $("body").html();

  const internalLinks = new Set;
  const externalLinks = new Set;

  $("a").each((_, el) => {
    const link = $(el).attr("href");
    if (link === "/") return;

    if (link.startsWith("https") || link.startsWith("http")) {
      externalLinks.add(link);
    } else {
      internalLinks.add(link);
    }
  });

  return {
    metaData: pageMetadata,
    body: pageBody,
    internalLinks: Array.from(internalLinks),
    externalLinks: Array.from(externalLinks),
  };
}

async function generateVectorEmbeddings({ text }) {
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  return embedding.data[0].embedding;
}

async function ingest(url = "") {
  try {
    console.log(`✨Ingesting url = ${url}`);

    // Delete existing embeddings for this URL (if any)
    await deleteEmbeddingsForUrl(url);

    const { body, metaData, internalLinks } = await scrapeWebpage(url);
    if (!body) {
      console.error(`❌ Body content is empty for ${url}`);
      return;
    }

    const bodyChunks = splitIntoChunks(body, 100);
    console.log(`📦 Split body into ${bodyChunks.length} chunks`);
    for (let i = 0; i < bodyChunks.length; i++) {
      const chunk = bodyChunks[i];
      const chunkedBodyEmbeddings = await generateVectorEmbeddings({
        text: chunk,
      });
      await addToVectorDB(
        { embedding: chunkedBodyEmbeddings, url, metaData, body: chunk },
        i
      );
    }

    console.log(`🎉Ingesting successful = ${url}`);
  } catch (error) {
    console.error(`❌ Error ingesting ${url}:`, error);
  }
}

function splitIntoChunks(text, numChunks) {
  if (numChunks <= 0) {
    throw new Error("Number of chunks must be greater than zero");
  }

  const chunkSize = Math.ceil(text.length / numChunks);
  const chunks = [];

  for (let i = 0; i < text.length; i += chunkSize) {
    chunks.push(text.slice(i, i + chunkSize));
  }

  return chunks;
}

async function addToVectorDB({ embedding, url, body = '', head }, chunkIndex = 0) {
  const collection = await chromaClient.getOrCreateCollection({
    name: WEB_COLLECTION,
  });

  const uniqueId = `${url}_chunk_${chunkIndex}`;

  // Check if the embedding already exists
  const existingEmbedding = await collection.get({ ids: [uniqueId] });

  if (existingEmbedding.ids.length > 0) {
    console.log(`🟡 Embedding already exists for ID: ${uniqueId}, skipping...`);
    return; // Skip adding this embedding
  }

  // Add the embedding if it doesn't exist
  await collection.add({
    ids: [uniqueId],
    embeddings: [embedding],
    metadatas: [
      {
        url,
        body,
        head,
        chunkIndex,
      },
    ],
  });

  console.log(`✅ Added embedding for ID: ${uniqueId}`);
}

async function isUrlProcessed(url) {
  const collection = await chromaClient.getOrCreateCollection({
    name: WEB_COLLECTION,
  });
  const embeddings = await collection.get();
  return embeddings.ids.some((id) => id.startsWith(url));
}

async function deleteEmbeddingsForUrl(url) {
  const collection = await chromaClient.getOrCreateCollection({
    name: WEB_COLLECTION,
  });
  const embeddings = await collection.get();
  const idsToDelete = embeddings.ids.filter((id) => id.startsWith(url));
  if (idsToDelete.length > 0) {
    await collection.delete({ ids: idsToDelete });
    console.log(`🗑️ Deleted ${idsToDelete.length} embeddings for ${url}`);
  }
}

async function chat(question = '') {
  const questionEmbedding = await generateVectorEmbeddings({text: question});

  const collection = await chromaClient.getOrCreateCollection({
    name: WEB_COLLECTION,
  });

  const collectionResult = await collection.query({
    nResults: 3,
    queryEmbeddings: questionEmbedding
  });

  const body = collectionResult.metadatas[0].map((e) => e.body).filter(e => e.trim() !== '' && !!e);

  const url = collectionResult.metadatas[0].map((e) => e.url).filter(e => e.trim() !== '' && !!e);

  const openAIResponse = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      {role: 'system', content: 'You are an AI support agent, expert in providing support to users on behalf of a webpage. Given the context about page content, reply the user accordingly.'},
      {
        role: 'user',
        content:`Query: ${question}\n\n
                 URl: ${url.join(', ')}\n\n
                 Retrieved conext: ${body.join(', ')}`
      }
    ]
  });

  console.log({
    message: `🤖: ${openAIResponse.choices[0].message.content}`,
    url: url[0]
  });
  
}

chat('List all his projects?')
