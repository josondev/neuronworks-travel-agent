/**
 * src/db/CacheService.js
 *
 * DB-backed API response cache with per-tool TTL strategy.
 * Cache key = sha256(toolName + sorted JSON args) — deterministic regardless
 * of argument property order.
 *
 * TTL strategy (minutes):
 *   currency          →  15  (rates move fast)
 *   search_flights    →  30
 *   search_hotels     →  60
 *   get_weather       → 120
 *   search_places     → 1440 (places don't change daily)
 *   build_trip_data   →  30  (composite — use shortest member TTL)
 *   calculate_budget  → 720  (static formula, but bust daily is fine)
 *   default           →  60
 */

import crypto from 'crypto';
import { prisma, isDBReady } from './prisma.js';

// ─── TTL map (minutes) ───────────────────────────────────────────────────────
const TTL_MINUTES = {
  get_exchange_rate:      15,
  search_flights:         30,
  build_trip_data:        30,
  search_hotels:          60,
  calculate_trip_budget:  720,
  get_weather_forecast:   120,
  search_places:          1440,
};

function getTTLms(toolName) {
  const minutes = TTL_MINUTES[toolName] ?? 60;
  return minutes * 60 * 1000;
}

// ─── cache key ───────────────────────────────────────────────────────────────
function makeCacheKey(toolName, args) {
  // Sort keys for deterministic hashing regardless of argument insertion order
  const sorted = JSON.stringify(args, Object.keys(args ?? {}).sort());
  const payload = `${toolName}::${sorted}`;
  return crypto.createHash('sha256').update(payload).digest('hex');
}

// ─── internal guard ──────────────────────────────────────────────────────────
function db() {
  if (!isDBReady() || !prisma) return null;
  return prisma;
}

// ─── public API ──────────────────────────────────────────────────────────────

/**
 * Retrieve a cached result. Returns parsed object/array, or null on miss/expired.
 */
export async function getCached(toolName, args) {
  const client = db();
  if (!client) return null;

  const key = makeCacheKey(toolName, args);
  try {
    const row = await client.searchCache.findUnique({ where: { cacheKey: key } });
    if (!row) return null;

    if (new Date() > new Date(row.expiresAt)) {
      // Expired — delete lazily
      client.searchCache.delete({ where: { cacheKey: key } }).catch(() => {});
      return null;
    }

    return JSON.parse(row.resultJson);
  } catch (err) {
    console.error('CacheService getCached error:', err.message);
    return null;
  }
}

/**
 * Store a result in the cache. Upserts so repeated calls don't create duplicates.
 */
export async function setCached(toolName, args, result) {
  const client = db();
  if (!client) return;

  const key      = makeCacheKey(toolName, args);
  const ttlMs    = getTTLms(toolName);
  const expiresAt = new Date(Date.now() + ttlMs);

  try {
    await client.searchCache.upsert({
      where:  { cacheKey: key },
      update: { resultJson: JSON.stringify(result), expiresAt },
      create: {
        cacheKey:   key,
        toolName,
        resultJson: JSON.stringify(result),
        expiresAt,
      },
    });
  } catch (err) {
    console.error('CacheService setCached error:', err.message);
  }
}

/**
 * Wrap any async function with cache-aside logic.
 * If a cached result exists it is returned immediately; otherwise fn() is
 * called, its result is stored, and then returned.
 *
 * @param {string}   toolName
 * @param {object}   args
 * @param {Function} fn       — async () => result
 */
export async function withCache(toolName, args, fn) {
  const hit = await getCached(toolName, args);
  if (hit !== null) {
    console.error(`📦 CACHE HIT: ${toolName}`);
    return hit;
  }
  const result = await fn();
  // Store in background — don't await so tool response is not delayed
  setCached(toolName, args, result).catch(() => {});
  return result;
}

/**
 * Remove all expired entries from the cache table.
 * Safe to call on a periodic interval (e.g. hourly).
 */
export async function pruneExpiredCache() {
  const client = db();
  if (!client) return 0;
  try {
    const { count } = await client.searchCache.deleteMany({
      where: { expiresAt: { lt: new Date() } },
    });
    if (count > 0) console.error(`🧹 Cache pruned ${count} expired entries`);
    return count;
  } catch (err) {
    console.error('CacheService pruneExpiredCache error:', err.message);
    return 0;
  }
}
