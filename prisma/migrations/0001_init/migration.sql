-- Migration: 0001_init
-- Generated from prisma/schema.prisma via:
--   npx prisma migrate diff --from-empty --to-schema prisma/schema.prisma --script
--
-- Run against a live DB with:
--   npx prisma migrate deploy        (production)
--   npx prisma migrate dev           (development, creates shadow DB)

-- CreateSchema
CREATE SCHEMA IF NOT EXISTS "public";

-- CreateTable
CREATE TABLE "User" (
    "id"         TEXT        NOT NULL,
    "sessionKey" TEXT        NOT NULL,
    "createdAt"  TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "search_cache" (
    "id"         TEXT        NOT NULL,
    "cacheKey"   TEXT        NOT NULL,
    "toolName"   TEXT        NOT NULL,
    "resultJson" TEXT        NOT NULL,
    "expiresAt"  TIMESTAMP(3) NOT NULL,
    "createdAt"  TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "search_cache_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "trip_searches" (
    "id"         TEXT        NOT NULL,
    "userId"     TEXT        NOT NULL,
    "toolName"   TEXT        NOT NULL,
    "paramsJson" TEXT        NOT NULL,
    "resultJson" TEXT        NOT NULL,
    "createdAt"  TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "trip_searches_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "saved_itineraries" (
    "id"            TEXT        NOT NULL,
    "userId"        TEXT        NOT NULL,
    "title"         TEXT        NOT NULL,
    "destination"   TEXT        NOT NULL,
    "startDate"     TEXT,
    "endDate"       TEXT,
    "notes"         TEXT,
    "itineraryJson" TEXT        NOT NULL,
    "createdAt"     TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "saved_itineraries_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "budget_estimates" (
    "id"            TEXT             NOT NULL,
    "userId"        TEXT             NOT NULL,
    "destination"   TEXT             NOT NULL,
    "durationDays"  INTEGER          NOT NULL,
    "travelers"     INTEGER          NOT NULL,
    "budgetLevel"   TEXT             NOT NULL,
    "totalUsd"      DOUBLE PRECISION NOT NULL,
    "breakdownJson" TEXT             NOT NULL,
    "createdAt"     TIMESTAMP(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "budget_estimates_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_sessionKey_key"          ON "User"("sessionKey");
CREATE UNIQUE INDEX "search_cache_cacheKey_key"    ON "search_cache"("cacheKey");
CREATE INDEX        "search_cache_cacheKey_idx"    ON "search_cache"("cacheKey");
CREATE INDEX        "search_cache_expiresAt_idx"   ON "search_cache"("expiresAt");
CREATE INDEX        "trip_searches_userId_createdAt_idx" ON "trip_searches"("userId", "createdAt");
CREATE INDEX        "trip_searches_toolName_idx"   ON "trip_searches"("toolName");
CREATE INDEX        "saved_itineraries_userId_idx" ON "saved_itineraries"("userId");
CREATE INDEX        "budget_estimates_userId_idx"  ON "budget_estimates"("userId");

-- AddForeignKey
ALTER TABLE "trip_searches"
    ADD CONSTRAINT "trip_searches_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "saved_itineraries"
    ADD CONSTRAINT "saved_itineraries_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "budget_estimates"
    ADD CONSTRAINT "budget_estimates_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
