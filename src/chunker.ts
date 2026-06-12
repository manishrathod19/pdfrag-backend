/**
 * @file chunker.ts
 * @description Splits large PDF text into overlapping chunks suitable
 * for embedding. Overlap ensures context is not lost at chunk boundaries.
 */

/**
 * @interface TextChunk
 * @description Represents a single piece of text extracted from a PDF,
 * ready to be embedded and stored in the vector database.
 */
export interface TextChunk {
  /** The chunk content — a contiguous substring of the source document. */
  text: string;
  /** Original PDF filename, kept so retrieved answers can cite their source. */
  source: string;
  /** Position of this chunk in the document (0-based) for ordering and traceability. */
  index: number;
}

/**
 * @method chunkText
 * @description Splits a full document text into overlapping word-based
 * chunks. Overlap of 50 words prevents context loss at boundaries
 * (a sentence that spans two chunks remains retrievable from either).
 * @param text - full extracted text from PDF
 * @param source - PDF filename for traceability
 * @param chunkSize - number of words per chunk (default 400)
 * @param overlap - words shared between consecutive chunks (default 50)
 * @returns array of TextChunk objects
 */
export function chunkText(
  text: string,
  source: string,
  chunkSize: number = 400,
  overlap: number = 50
): TextChunk[] {
  // Split on any whitespace — robust to tabs/newlines from pdf-parse output.
  const words = text.split(/\s+/).filter((w) => w.length > 0);

  // Result accumulator — built imperatively because window position depends on the previous step.
  const chunks: TextChunk[] = [];

  // Position counter assigned to each emitted chunk (separate from word index `i`).
  let chunkIndex = 0;

  // Sliding window: advance by (chunkSize - overlap) so each chunk shares `overlap`
  // words with the previous one. This preserves context at chunk boundaries.
  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    // Slice a window of up to chunkSize words starting at i.
    const slice = words.slice(i, i + chunkSize);

    // Re-join with single spaces — original whitespace is irrelevant for embedding.
    const chunkContent = slice.join(' ');

    // Skip near-empty fragments — embeddings of <20 chars are noisy and waste storage.
    if (chunkContent.length < 20) {
      continue;
    }

    chunks.push({
      text: chunkContent,
      source,
      index: chunkIndex,
    });

    chunkIndex += 1;

    // If this slice already reached the end of the document, stop early.
    // Without this, the loop would re-emit a tiny tail chunk on the next iteration.
    if (i + chunkSize >= words.length) {
      break;
    }
  }

  return chunks;
}
