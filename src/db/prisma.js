/**
 * src/db/prisma.js
 *
 * Singleton PrismaClient for Prisma 7 with the pg adapter.
 * In Prisma 7, the DATABASE_URL is not declared in schema.prisma —
 * it is passed here via PrismaPg and handed to PrismaClient({ adapter }).
 *
 * The module exports:
 *   prisma        — the singleton client (null when DATABASE_URL is missing)
 *   connectDB()   — call once at server startup; logs and resolves quietly
 *   disconnectDB()— call on process exit / SIGTERM
 *   isDBReady()   — returns true when client is connected
 */

import dotenv from 'dotenv';
dotenv.config();

import pg from 'pg';
import { PrismaPg } from '@prisma/adapter-pg';

// ─── Lazy import of generated client ────────────────────────────────────────
// We import dynamically so the file is still loadable before `prisma generate`
// has run (the server functions normally without DB when no DATABASE_URL is set).
let PrismaClient;
let prisma = null;
let _ready  = false;

async function loadPrismaClient() {
  if (PrismaClient) return PrismaClient;
  try {
    const mod = await import('../../generated/prisma/client.js');
    PrismaClient = mod.PrismaClient;
    return PrismaClient;
  } catch (err) {
    console.error('⚠️  DB: generated Prisma client not found — run `npx prisma generate`');
    console.error('    DB features will be disabled until then.');
    return null;
  }
}

// ─── connect ────────────────────────────────────────────────────────────────
export async function connectDB() {
  const url = process.env.DATABASE_URL;
  if (!url) {
    console.warn('⚠️  DB: DATABASE_URL not set — all DB features disabled.');
    return;
  }

  const Client = await loadPrismaClient();
  if (!Client) return;

  try {
    const pool   = new pg.Pool({ connectionString: url });
    const adapter = new PrismaPg(pool);
    prisma = new Client({ adapter });

    // Verify the connection with a lightweight ping
    await prisma.$queryRaw`SELECT 1`;
    _ready = true;
    console.error('✅ DB: PostgreSQL connected via Prisma 7 + pg adapter');
  } catch (err) {
    console.error('❌ DB: Connection failed —', err.message);
    prisma  = null;
    _ready  = false;
  }
}

// ─── disconnect ─────────────────────────────────────────────────────────────
export async function disconnectDB() {
  if (prisma) {
    try {
      await prisma.$disconnect();
      console.error('✅ DB: Disconnected cleanly');
    } catch (_) { /* silent */ }
    prisma = null;
    _ready = false;
  }
}

// ─── helpers ────────────────────────────────────────────────────────────────
export function isDBReady() { return _ready && prisma !== null; }

export { prisma };

// ─── default export convenience ─────────────────────────────────────────────
export default { prisma, connectDB, disconnectDB, isDBReady };
