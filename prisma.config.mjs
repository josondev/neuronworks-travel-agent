// Prisma 7 config file — replaces url in schema.prisma datasource block.
// DATABASE_URL is read from .env via dotenv, loaded explicitly in src/db/prisma.js.

import { defineConfig } from 'prisma/config';

export default defineConfig({
  schema: 'prisma/schema.prisma',
  migrations: {
    path: 'prisma/migrations',
  },
  datasource: {
    url: process.env.DATABASE_URL,
  },
});
