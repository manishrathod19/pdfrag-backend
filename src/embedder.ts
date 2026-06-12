/**
 * @file embedder.ts
 * @description Generates vector embeddings for text using a locally
 * running transformer model. No API calls or internet required after
 * the first model download (~25MB, cached permanently on disk).
 */

// @xenova/transformers ships ESM; we lazy-import via dynamic import() inside
// getEmbedder() so this CommonJS file stays compatible with `ts-node`.
// The pipeline type is loose (`any`) on purpose — pinning the exact union type
// from the transformers package would force its ESM types into our CJS build.

/**
 * Singleton holder for the loaded pipeline.
 * Reason: the model weights are ~25MB and loading takes seconds. Caching it
 * means we pay that cost once at first request, not on every embed call.
 */
let embedderPipeline: any = null;

/**
 * In-flight promise so concurrent first calls don't each kick off a separate load.
 * Without this guard, a burst of parallel requests during cold start would each
 * spawn their own model download.
 */
let loadingPromise: Promise<any> | null = null;

/**
 * @method getEmbedder
 * @description Returns the singleton embedding pipeline, initializing
 * it on first call. Uses Xenova/all-MiniLM-L6-v2 which produces
 * 384-dimensional vectors and runs entirely on local CPU/GPU.
 * @returns Promise resolving to the transformer pipeline instance
 */
export async function getEmbedder(): Promise<any> {
  // Fast path: already initialized.
  if (embedderPipeline) {
    return embedderPipeline;
  }

  // Concurrent first-call guard — share the same in-flight promise.
  if (loadingPromise) {
    return loadingPromise;
  }

  loadingPromise = (async () => {
    // Dynamic ESM import — required because @xenova/transformers is published as ESM
    // and our backend tsconfig is CommonJS.
    const { pipeline } = await import('@xenova/transformers');

    // 'feature-extraction' is the task name for raw embeddings.
    // Xenova/all-MiniLM-L6-v2: small (25MB), fast, 384-dim, MIT-licensed.
    const created = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    embedderPipeline = created;
    return created;
  })();

  return loadingPromise;
}

/**
 * @method embedText
 * @description Converts a plain text string into a 384-dimensional
 * float vector using mean pooling and L2 normalization.
 * Mean pooling averages token embeddings into one fixed-size vector.
 * Normalization ensures cosine similarity behaves correctly in Qdrant
 * (cosine distance reduces to a dot product on normalized vectors).
 * @param text - the text to embed
 * @returns Promise resolving to number[] of length 384
 */
export async function embedText(text: string): Promise<number[]> {
  const embedder = await getEmbedder();

  // pooling: 'mean' — averages token embeddings into one fixed-size vector.
  // normalize: true — produces unit-length vectors for cosine similarity.
  const output = await embedder(text, { pooling: 'mean', normalize: true });

  // The pipeline returns a Tensor-like object whose `.data` is a TypedArray (Float32Array).
  // Array.from converts it to a plain number[] which Qdrant's REST client accepts.
  return Array.from(output.data as Float32Array);
}
