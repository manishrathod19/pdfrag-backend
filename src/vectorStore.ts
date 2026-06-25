/**
 * @file vectorStore.ts
 * @description Interface to the Qdrant vector database. Handles collection
 * initialization, storing embedded chunks, and semantic similarity search.
 * Qdrant runs locally via Docker on port 6333.
 */

import { QdrantClient } from '@qdrant/js-client-rest';
import { embedText } from './embedder';
import { TextChunk } from './chunker';
import * as crypto from 'crypto';

/**
 * Collection name in Qdrant for all PDF embeddings.
 * Kept as a constant so the same name is used by init, store, and search.
 */
const COLLECTION_NAME = 'pdf_knowledge';

/**
 * Vector size MUST match the all-MiniLM-L6-v2 output dimensions exactly.
 * Mismatching this with the embedder will cause Qdrant to reject upserts.
 */
const VECTOR_SIZE = 384;

/**
 * Qdrant default REST URL. Configurable via env so this works in Docker too.
 */
const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;


console.log(`[qdrant] Using Qdrant at ${QDRANT_URL}`);
/**
 * Single shared Qdrant client. The REST client is stateless — one instance is fine.
 */
const client = new QdrantClient({ url: QDRANT_URL, apiKey: QDRANT_API_KEY });

/**
 * @method initCollection
 * @description Checks if the Qdrant collection exists and creates it
 * if not. Uses Cosine distance which is best for normalized text embeddings.
 * Called once at server startup before any requests are handled.
 * @returns Promise<void>
 */
export async function initCollection(): Promise<void> {
  // List existing collections so re-running the server does not throw "already exists".
  const collections = await client.getCollections();
  console.log(`[qdrant] Existing collections: ${collections.collections.map((c) => c.name).join(', ')}`);
  const exists = collections.collections.some((c) => c.name === COLLECTION_NAME);

  if (!exists) {
    // Cosine distance is the standard choice for normalized text embeddings.
    // Vectors out of all-MiniLM-L6-v2 are already L2-normalized, so cosine == dot product.
    await client.createCollection(COLLECTION_NAME, {
      vectors: { size: VECTOR_SIZE, distance: 'Cosine' },
    });
    console.log(`[qdrant] Created collection "${COLLECTION_NAME}"`);
  } else {
    console.log(`[qdrant] Collection "${COLLECTION_NAME}" already exists`);
  }
}

/**
 * @method storeChunks
 * @description Embeds each TextChunk and upserts it into Qdrant with
 * its payload (text, source filename, index). Uses upsert so re-ingesting
 * the same PDF does not create duplicates — points with the same ID are overwritten.
 * @param chunks - array of TextChunk to embed and store
 * @returns Promise<void>
 */
export async function storeChunks(chunks: TextChunk[]): Promise<void> {
  if (chunks.length === 0) {
    return;
  }

  // Build the array of Qdrant points. We embed in series rather than Promise.all
  // because the embedder pipeline is single-threaded — parallel calls are queued
  // internally anyway and would just balloon memory.
  const points: { id: string; vector: number[]; payload: Record<string, unknown> }[] = [];

  for (const chunk of chunks) {
    const vector = await embedText(chunk.text);

    // Deterministic UUID derived from (source + index). Using the same input twice
    // produces the same UUID, so re-ingesting the same PDF overwrites old points
    // instead of creating duplicates. createHash + format → RFC 4122 UUID v5-ish.
    const id = deterministicUuid(`${chunk.source}::${chunk.index}`);

    points.push({
      id,
      vector,
      payload: {
        text: chunk.text,
        source: chunk.source,
        index: chunk.index,
      },
    });
  }

  // upsert (vs. insert): inserts if new, overwrites if id already exists.
  // This is exactly what we want when a user re-uploads a PDF.
  await client.upsert(COLLECTION_NAME, { points, wait: true });
  console.log(`[qdrant] Upserted ${points.length} points`);
}

/**
 * @method deleteChunksBySource
 * @description Removes every Qdrant point belonging to a single source PDF.
 * Points are matched on their `source` payload field (the original filename),
 * which is the same value written by storeChunks. Used when a document is
 * deleted so its chunks no longer surface in search results.
 * @param source - the source filename whose chunks should be removed
 * @returns Promise<void>
 */
export async function deleteChunksBySource(source: string): Promise<void> {
  // Delete by filter rather than by ID: we don't track how many chunks a PDF
  // produced, so matching on the `source` payload removes them all in one call.
  // wait: true so the response only returns once the deletion is durable.
  await client.delete(COLLECTION_NAME, {
    filter: {
      must: [{ key: 'source', match: { value: source } }],
    },
    wait: true,
  });
  console.log(`[qdrant] Deleted points for source "${source}"`);
}

/**
 * @method searchChunks
 * @description Embeds the user's question and finds the topK most
 * semantically similar chunks stored in Qdrant using cosine similarity.
 * These chunks form the context given to the LLM for answering.
 * @param query - the user's question as plain text
 * @param topK - number of chunks to retrieve (default 4)
 * @returns Promise resolving to array of matching text strings
 */
export async function searchChunks(query: string, topK: number = 4): Promise<string[]> {
  // Embed the query with the SAME model as the stored chunks so vectors
  // share an embedding space. Mismatched models produce meaningless similarity.
  const vector = await embedText(query);

  const results = await client.search(COLLECTION_NAME, {
    vector,
    limit: topK,
    // with_payload: true so we get the original text back, not just IDs/scores.
    with_payload: true,
  });

  // Map points back to their stored text. Filter out any points that somehow
  // have no payload (defensive — should not happen with our writer).
  return results
    .map((r) => (r.payload?.text as string | undefined) ?? '')
    .filter((t) => t.length > 0);
}

/**
 * @method deterministicUuid
 * @description Produces a UUID-formatted string from an arbitrary input string
 * using SHA-1. Same input always yields the same UUID, which gives us idempotent
 * upserts without storing a separate ID map.
 * @param input - any string identifier (here we use `source::index`)
 * @returns RFC 4122-formatted UUID string
 */
function deterministicUuid(input: string): string {
  // sha1 → 20 bytes, take first 16 to fit a UUID (128 bits).
  const hash = crypto.createHash('sha1').update(input).digest('hex').slice(0, 32);
  // Splice into 8-4-4-4-12 hex layout. Qdrant accepts any RFC 4122 UUID string.
  return `${hash.slice(0, 8)}-${hash.slice(8, 12)}-${hash.slice(12, 16)}-${hash.slice(16, 20)}-${hash.slice(20, 32)}`;
}
