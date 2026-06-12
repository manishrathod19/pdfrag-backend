/**
 * @file watcher.ts
 * @description Watches the /pdfs folder for newly added PDF files
 * and automatically ingests them into the vector store. This means
 * users can drop files into the folder without manual re-ingestion.
 */

import chokidar from 'chokidar';
import * as path from 'path';
import { extractTextFromPDF } from './ingest';
import { chunkText } from './chunker';
import { storeChunks } from './vectorStore';

/**
 * @method watchPDFFolder
 * @description Starts a chokidar file watcher on the given folder.
 * Only responds to "add" events (new files) — not modifications —
 * because most editors save PDFs by replacing them, which fires both
 * remove + add. Reacting to "change" would double-process every save.
 * On detection: extract → chunk → embed → store pipeline runs.
 * @param folderPath - path to the directory to watch for new PDFs
 * @returns void (runs as a background process)
 */
export function watchPDFFolder(folderPath: string): void {
  // ignoreInitial: true so existing PDFs at startup do NOT each trigger an add.
  // Bulk ingestion at startup is handled separately by the /api/ingest route.
  const watcher = chokidar.watch(folderPath, {
    ignoreInitial: true,
    // awaitWriteFinish — wait for the file size to stop changing before firing.
    // Without this, large PDFs being copied in will fire "add" mid-copy and we'd
    // try to parse a half-written file.
    awaitWriteFinish: { stabilityThreshold: 1000, pollInterval: 100 },
  });

  watcher.on('add', async (filePath: string) => {
    // Filter on extension here rather than via chokidar's `glob` option, because
    // matching globs across Windows path separators can be flaky.
    if (!filePath.toLowerCase().endsWith('.pdf')) {
      return;
    }

    const filename = path.basename(filePath);
    console.log(`[watcher] New PDF detected: ${filename}`);

    try {
      // Step 1: extract — read PDF bytes and pull plain text out.
      const text = await extractTextFromPDF(filePath);

      // Step 2: chunk — split into overlapping windows so each chunk is short
      // enough for the embedder and contiguous for retrieval.
      const chunks = chunkText(text, filename);

      // Step 3: embed + store — convert each chunk to a 384-dim vector and
      // upsert into Qdrant. Using upsert means re-adding the same file is safe.
      await storeChunks(chunks);

      console.log(`[watcher] Ingested ${chunks.length} chunks from ${filename}`);
    } catch (err) {
      // Log and keep watching — one failed file should not crash the watcher.
      console.error(`[watcher] Failed to ingest ${filename}:`, err);
    }
  });

  console.log(`[watcher] Watching folder for PDFs: ${folderPath}`);
}
