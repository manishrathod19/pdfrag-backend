/**
 * @file llm.ts
 * @description Handles RAG (Retrieval-Augmented Generation) logic.
 * Retrieves relevant PDF chunks from Qdrant, builds a grounded prompt,
 * and sends it to the locally running Ollama LLM (llama3).
 */

import { Ollama } from 'ollama';
import { searchChunks } from './vectorStore';

/**
 * Model name. llama3 is open-weights (Meta license) and runs locally via Ollama.
 * Override with $env:OLLAMA_MODEL if the user pulled a different tag.
 */
const MODEL_NAME = process.env.OLLAMA_MODEL || 'mistral:7b-instruct-q4_0';

/**
 * Ollama host. Defaults to the remote ngrok tunnel so the deployed backend can
 * reach an Ollama instance running outside the container. Override with
 * $env:OLLAMA_HOST (e.g. "http://localhost:11434" for local dev).
 */
const OLLAMA_HOST = process.env.OLLAMA_HOST || ' https://47b6-45-250-226-215.ngrok-free.app';

/**
 * Single Ollama client pointed at OLLAMA_HOST. The default `ollama` singleton
 * always targets http://localhost:11434, so we instantiate our own to control
 * the host.
 */
const ollama = new Ollama({ host: OLLAMA_HOST });

/**
 * @method buildPrompt
 * @description Constructs the RAG prompt that instructs the LLM to
 * answer strictly from the provided context. Grounding the prompt
 * prevents hallucinations by explicitly forbidding outside knowledge.
 * @param question - the user's original question
 * @param context - concatenated relevant PDF chunks from Qdrant
 * @returns formatted prompt string ready to send to Ollama
 */
export function buildPrompt(question: string, context: string): string {
  // The exact wording matters — small models follow surface patterns more than intent.
  // We include the literal "fallback" sentence the model should produce when context is missing.
  return `You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say exactly:
'I could not find that information in the uploaded documents.'
Do not make up answers. Do not use outside knowledge.

Context:
${context}

Question: ${question}
Answer:`;
}

/**
 * @method ask
 * @description Full RAG pipeline for a single question. Retrieves
 * context from the vector store, builds the grounded prompt, calls Ollama,
 * and returns the complete answer as a string.
 * @param question - user question string
 * @returns Promise resolving to LLM answer string
 */
export async function ask(question: string): Promise<string> {
  // Retrieve top-K most similar chunks. K=4 is a reasonable default —
  // small enough to fit in context, large enough to cover paraphrased matches.
  const chunks = await searchChunks(question, 4);

  // If retrieval found nothing, short-circuit. Calling the LLM with an empty
  // context would just waste compute and likely produce a hallucinated answer.
  if (chunks.length === 0) {
    return 'I could not find that information in the uploaded documents.';
  }

  // Join chunks with double newlines so the model sees them as distinct passages.
  const context = chunks.join('\n\n');
  const prompt = buildPrompt(question, context);

  // ollama.chat returns the full response after the model finishes generating.
  const response = await ollama.chat({
    model: MODEL_NAME,
    messages: [{ role: 'user', content: prompt }],
  });

  return response.message.content;
}

/**
 * @method askStream
 * @description Streaming version of ask(). Same RAG pipeline but
 * calls the onToken callback for each token as it is generated, enabling
 * real-time word-by-word display in the frontend chat UI.
 * @param question - user question string
 * @param onToken - callback fired for each streamed token string
 * @param onDone - callback fired when streaming is complete
 * @returns Promise<void>
 */
export async function askStream(
  question: string,
  onToken: (token: string) => void,
  onDone: () => void
): Promise<void> {
  const chunks = await searchChunks(question, 4);

  // Same empty-retrieval guard as ask(), but we deliver the message via the
  // streaming callbacks so the SSE contract is preserved.
  if (chunks.length === 0) {
    onToken('I could not find that information in the uploaded documents.');
    onDone();
    return;
  }

  const context = chunks.join('\n\n');
  const prompt = buildPrompt(question, context);

  // stream: true makes ollama.chat return an async iterable of partial chunks.
  const stream = await ollama.chat({
    model: MODEL_NAME,
    messages: [{ role: 'user', content: prompt }],
    stream: true,
  });

  // for-await iterates each partial response. Each chunk contains a small piece
  // of generated text in `message.content` — possibly a single word or sub-word.
  for await (const part of stream) {
    const token = part.message?.content ?? '';
    if (token.length > 0) {
      onToken(token);
    }
  }

  onDone();
}
