/**
 * @file server.ts
 * @description Main Express server. Registers all API routes, sets up
 * CORS for the Angular dev server, configures multer for PDF uploads,
 * and initializes Qdrant + the file watcher on startup.
 */

import express, { Request, Response } from 'express';
import cors from 'cors';
import multer from 'multer';
import * as path from 'path';
import * as fs from 'fs';
import { z } from 'zod';
import { initCollection, storeChunks } from './vectorStore';
import { ingestFolder, extractTextFromPDF } from './ingest';
import { chunkText } from './chunker';
import { ask, askStream } from './llm';
import { watchPDFFolder } from './watcher';
process.loadEnvFile()


/**
 * Absolute path to the shared PDF folder. Resolved once so every route
 * uses the same location regardless of where the server is launched from.
 */
const PDF_FOLDER = path.resolve(__dirname, '..', '..', 'pdfs');

/**
 * Port the HTTP server listens on. Configurable via $env:PORT for Docker.
 */
const PORT = Number(process.env.PORT) || 3001;

/**
 * Zod schema for /api/ask body validation. Defined once so we can reuse it
 * for the streaming endpoint too.
 */
const askSchema = z.object({
  question: z.string().min(1, 'question must not be empty'),
});

/**
 * Multer storage configuration.
 * - destination: write straight into PDF_FOLDER so the watcher would also see it.
 * - filename: keep the original filename so the user recognizes uploads.
 *   We do NOT sanitize beyond what multer does because the source is a trusted
 *   local user via the desktop UI; the file watcher reads back what's on disk.
 */
const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => {
      // Ensure the folder exists; Express does not create paths on its own.
      if (!fs.existsSync(PDF_FOLDER)) {
        fs.mkdirSync(PDF_FOLDER, { recursive: true });
      }
      cb(null, PDF_FOLDER);
    },
    filename: (_req, file, cb) => cb(null, file.originalname),
  }),
});

/** The single Express application instance. */
const app = express();

// CORS — allow the Angular dev server at localhost:4200 during development.
// In production behind the same origin this becomes a no-op.
app.use(cors({ origin: true }));

// express.json() — parse JSON request bodies for /api/ask.
app.use(express.json());

/**
 * @route POST /api/ask
 * @description Accepts a question, runs the full RAG pipeline, returns the answer.
 * Validates the request body with zod before processing.
 */
app.post('/api/ask', async (req: Request, res: Response) => {
  // zod safeParse so we return a clean 400 instead of a thrown stack trace.
  const parsed = askSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.issues });
  }

  try {
    const answer = await ask(parsed.data.question);
    return res.json({ answer });
  } catch (err) {
    console.error('[/api/ask] error:', err);
    return res.status(500).json({ error: 'Internal error during RAG pipeline' });
  }
});

/**
 * @route GET /api/ask/stream
 * @description SSE endpoint. Streams LLM tokens one by one as
 * Server-Sent Events. Frontend EventSource connects here for
 * real-time token-by-token chat response rendering.
 * SSE format: data: {"token": "word"}\n\n
 * End signal:  data: {"done": true}\n\n
 *
 * Note: EventSource in browsers can only send GET, so the question is
 * passed as a query-string parameter.
 */
app.get('/api/ask/stream', async (req: Request, res: Response) => {
  const question = (req.query.question as string | undefined) ?? '';
  if (!question.trim()) {
    return res.status(400).json({ error: 'question query param required' });
  }

  // SSE handshake headers — these must be set before any data is written.
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  // Disable Nginx buffering if a reverse proxy is ever introduced.
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  /**
   * Helper to write a single SSE event in the `data: <json>\n\n` shape.
   * Defined inline so it captures `res` without polluting module scope.
   */
  const sendEvent = (payload: object) => {
    res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };

  try {
    await askStream(
      question,
      // onToken — forward each generated token to the browser as its own event.
      (token) => sendEvent({ token }),
      // onDone — emit a sentinel event so the client knows when to close the connection.
      () => {
        sendEvent({ done: true });
        res.end();
      }
    );
  } catch (err) {
    console.error('[/api/ask/stream] error:', err);
    sendEvent({ error: 'stream error' });
    res.end();
  }
});

/**
 * @route POST /api/upload
 * @description Accepts a PDF file via multipart upload, saves it to
 * ../pdfs folder, then immediately ingests it into Qdrant so it
 * is searchable without waiting for the file watcher or manual re-ingestion.
 */
app.post('/api/upload', upload.single('file'), async (req: Request, res: Response) => {
  // multer attaches the file to req.file when storage succeeds.
  const file = req.file;
  if (!file) {
    return res.status(400).json({ error: 'No file uploaded (field name: file)' });
  }

  // Hard reject anything that didn't end up with a .pdf extension. The watcher
  // would skip it anyway, but the user deserves immediate feedback.
  if (!file.originalname.toLowerCase().endsWith('.pdf')) {
    fs.unlinkSync(file.path);
    return res.status(400).json({ error: 'Only PDF files are allowed' });
  }

  try {
    // Run the same extract → chunk → store pipeline as the watcher.
    // We run it inline so the response only returns once the upload is searchable.
    const text = await extractTextFromPDF(file.path);
    const chunks = chunkText(text, file.originalname);
    await storeChunks(chunks);

    return res.json({
      message: `Uploaded and ingested ${file.originalname}`,
      chunks: chunks.length,
    });
  } catch (err) {
    console.error('[/api/upload] error:', err);
    return res.status(500).json({ error: 'Failed to ingest uploaded PDF' });
  }
});

/**
 * @route POST /api/ingest
 * @description Re-ingests all PDFs in ../pdfs folder into Qdrant.
 * Useful when PDFs were added manually or earlier ingestion was incomplete.
 */
app.post('/api/ingest', async (_req: Request, res: Response) => {
  try {
    const chunks = await ingestFolder(PDF_FOLDER);
    await storeChunks(chunks);
    return res.json({ message: `Ingested ${chunks.length} chunks` });
  } catch (err) {
    console.error('[/api/ingest] error:', err);
    return res.status(500).json({ error: 'Ingestion failed' });
  }
});

/**
 * @route GET /api/documents
 * @description Returns the list of all PDF filenames in ../pdfs folder.
 * Used by the Angular sidebar to display uploaded documents.
 */
app.get('/api/documents', (_req: Request, res: Response) => {
  // Defensive: folder may not exist on first launch.
  if (!fs.existsSync(PDF_FOLDER)) {
    return res.json({ documents: [] });
  }
  // Filter to .pdf only so stray non-PDF files don't show up in the UI.
  const documents = fs
    .readdirSync(PDF_FOLDER)
    .filter((f) => f.toLowerCase().endsWith('.pdf'));
  return res.json({ documents });
});

/**
 * @route GET /api/health
 * @description Health check endpoint. Returns server status and
 * the current timestamp. Used by the frontend to verify the backend is reachable.
 */
app.get('/api/health', (_req: Request, res: Response) => {
  return res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

/**
 * @function startServer
 * @description Bootstraps the server: initializes the Qdrant collection,
 * starts the PDF folder watcher, then begins listening on port 3001.
 * Order matters — the collection must exist before any requests are handled,
 * otherwise the first /api/ask would 500.
 */
async function startServer(): Promise<void> {
  // 1. Create the Qdrant collection if it doesn't exist.
  await initCollection();

  // 2. Start watching the shared PDF folder for new uploads/drops.
  watchPDFFolder(PDF_FOLDER);

  // 3. Only now begin accepting HTTP traffic.
  app.listen(PORT, () => {
    console.log(`[server] Listening on http://localhost:${PORT}`);
    console.log(`[server] PDF folder: ${PDF_FOLDER}`);
  });
}

// Top-level await is unavailable in CommonJS; use .catch on the bootstrap promise.
startServer().catch((err) => {
  console.error('[server] Failed to start:', err);
  process.exit(1);
});
