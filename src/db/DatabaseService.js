/**
 * src/db/DatabaseService.js
 *
 * CRUD helpers for every table.  All methods are safe to call even when the
 * database is not connected — they return null/[] silently so the MCP server
 * keeps working without a DB in development.
 */

import { prisma, isDBReady } from './prisma.js';

// ─── internal guard ─────────────────────────────────────────────────────────
function db() {
  // prisma is exported as a let-binding; we re-read it each call so the
  // singleton is available after connectDB() resolves asynchronously.
  if (!isDBReady() || !prisma) return null;
  return prisma;
}

// ═══════════════════════════════════════════════════════════════════════════
// USERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Find or create a User row for the given SSE sessionId.
 * Returns the User record, or null when DB is unavailable.
 */
export async function upsertUser(sessionKey) {
  const client = db();
  if (!client) return null;
  try {
    return await client.user.upsert({
      where:  { sessionKey },
      update: {},
      create: { sessionKey },
    });
  } catch (err) {
    console.error('DB upsertUser error:', err.message);
    return null;
  }
}

/**
 * Find an existing User by sessionKey. Returns null when not found or DB down.
 */
export async function findUser(sessionKey) {
  const client = db();
  if (!client) return null;
  try {
    return await client.user.findUnique({ where: { sessionKey } });
  } catch (err) {
    console.error('DB findUser error:', err.message);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRIP SEARCHES (search history)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Log one tool call to trip_searches.
 * @param {string} userId
 * @param {string} toolName
 * @param {object} params   — raw arguments object
 * @param {any}    result   — tool result (will be JSON-serialised)
 */
export async function logSearch(userId, toolName, params, result) {
  const client = db();
  if (!client || !userId) return null;
  try {
    return await client.tripSearch.create({
      data: {
        userId,
        toolName,
        paramsJson: JSON.stringify(params),
        resultJson: JSON.stringify(result),
      },
    });
  } catch (err) {
    console.error('DB logSearch error:', err.message);
    return null;
  }
}

/**
 * Fetch the last `limit` searches for a user (default 20).
 */
export async function getSearchHistory(userId, limit = 20) {
  const client = db();
  if (!client || !userId) return [];
  try {
    const rows = await client.tripSearch.findMany({
      where:   { userId },
      orderBy: { createdAt: 'desc' },
      take:    limit,
      select: {
        id:        true,
        toolName:  true,
        paramsJson: true,
        resultJson: true,
        createdAt: true,
      },
    });
    return rows.map(r => ({
      id:        r.id,
      toolName:  r.toolName,
      params:    JSON.parse(r.paramsJson),
      result:    JSON.parse(r.resultJson),
      createdAt: r.createdAt,
    }));
  } catch (err) {
    console.error('DB getSearchHistory error:', err.message);
    return [];
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// SAVED ITINERARIES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Save a named itinerary for a user.
 */
export async function saveItinerary(userId, { title, destination, startDate, endDate, notes, itinerary }) {
  const client = db();
  if (!client || !userId) return null;
  try {
    return await client.savedItinerary.create({
      data: {
        userId,
        title,
        destination,
        startDate:     startDate  || null,
        endDate:       endDate    || null,
        notes:         notes      || null,
        itineraryJson: JSON.stringify(itinerary),
      },
    });
  } catch (err) {
    console.error('DB saveItinerary error:', err.message);
    return null;
  }
}

/**
 * Fetch all saved itineraries for a user, newest first.
 */
export async function getSavedItineraries(userId) {
  const client = db();
  if (!client || !userId) return [];
  try {
    const rows = await client.savedItinerary.findMany({
      where:   { userId },
      orderBy: { createdAt: 'desc' },
    });
    return rows.map(r => ({
      id:          r.id,
      title:       r.title,
      destination: r.destination,
      startDate:   r.startDate,
      endDate:     r.endDate,
      notes:       r.notes,
      itinerary:   JSON.parse(r.itineraryJson),
      createdAt:   r.createdAt,
    }));
  } catch (err) {
    console.error('DB getSavedItineraries error:', err.message);
    return [];
  }
}

/**
 * Delete a saved itinerary by id (only if it belongs to the user).
 */
export async function deleteItinerary(userId, itineraryId) {
  const client = db();
  if (!client || !userId) return false;
  try {
    await client.savedItinerary.deleteMany({
      where: { id: itineraryId, userId },
    });
    return true;
  } catch (err) {
    console.error('DB deleteItinerary error:', err.message);
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// BUDGET ESTIMATES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Persist a budget estimate result.
 */
export async function saveBudgetEstimate(userId, { destination, durationDays, travelers, budgetLevel, totalUsd, breakdown }) {
  const client = db();
  if (!client || !userId) return null;
  try {
    return await client.budgetEstimate.create({
      data: {
        userId,
        destination,
        durationDays: Number(durationDays),
        travelers:    Number(travelers),
        budgetLevel,
        totalUsd:     Number(totalUsd),
        breakdownJson: JSON.stringify(breakdown),
      },
    });
  } catch (err) {
    console.error('DB saveBudgetEstimate error:', err.message);
    return null;
  }
}
