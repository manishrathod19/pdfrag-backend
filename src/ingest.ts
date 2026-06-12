/**
 * @file ingest.ts
 * @description Handles reading PDF files from disk, extracting their
 * text content using pdf-parse, and passing them to the chunker.
 */

import * as fs from 'fs';
import * as path from 'path';
// pdf-parse is published as CommonJS; require avoids a default-export typing mismatch.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const pdfParse: (buffer: Buffer) => Promise<{ text: string }> = require('pdf-parse');
import { chunkText, TextChunk } from './chunker';

/**
 * @method extractTextFromPDF
 * @description Reads a PDF file from disk as a binary buffer and
 * extracts its plain text using the pdf-parse library.
 * @param filePath - absolute or relative path to the PDF file
 * @returns Promise resolving to plain text string of PDF content
 */
export async function extractTextFromPDF(filePath: string): Promise<string> {
  // Read the entire file as a Buffer — pdf-parse expects raw bytes, not a stream.
  const buffer = fs.readFileSync(filePath);

  // pdf-parse returns an object with the extracted plain text.
  const data = await pdfParse(buffer);

  return data.text;
}

/**
 * @method ingestFolder
 * @description Scans a directory for all .pdf files, extracts text
 * from each, chunks them, and returns all chunks combined.
 * @param folderPath - path to folder containing PDFs
 * @returns Promise resolving to all TextChunk[] from all PDFs
 */
export async function ingestFolder(folderPath: string): Promise<TextChunk[]> {
  // Defensive: if the folder doesn't exist we return an empty list rather than throwing,
  // because at first startup the user may not have dropped any PDFs in yet.
  if (!fs.existsSync(folderPath)) {
    console.warn(`[ingest] Folder does not exist: ${folderPath}`);
    return [];
  }

  // List all entries, then filter to .pdf files only — case-insensitive
  // because Windows filesystems do not preserve case the same way Linux does.
  const entries = fs.readdirSync(folderPath);
  const pdfFiles = entries.filter((f) => f.toLowerCase().endsWith('.pdf'));

  // Process files sequentially. Parallelism would be faster but pdf-parse can
  // peak memory on large PDFs, and Node's event loop already overlaps I/O.
  const allChunks: TextChunk[] = [];
  for (const filename of pdfFiles) {
    const fullPath = path.join(folderPath, filename);
    try {
      const text = await extractTextFromPDF(fullPath);
      const chunks = chunkText(text, filename);

      // Log per-file progress so the operator can see ingestion happening live.
      console.log(`[ingest] ${filename} → ${chunks.length} chunks`);

      // Accumulate using push(...spread) instead of concat to avoid creating
      // throwaway intermediate arrays on every iteration.
      allChunks.push(...chunks);
    } catch (err) {
      // One bad PDF should not abort ingestion of the rest of the folder.
      console.error(`[ingest] Failed to process ${filename}:`, err);
    }
  }

  return allChunks;
}
