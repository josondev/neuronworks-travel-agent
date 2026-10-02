import { defineConfig } from 'prisma/config';
import { config } from 'dotenv';

// Load .env so DATABASE_URL is available to prisma studio / migrate / generate
config();

export default defineConfig({
  schema: 'prisma/schema.prisma',
  migrations: {
    path: 'prisma/migrations',
  },
  datasource: {
    url: process.env.DATABASE_URL,
  },
});
