# --- Build stage ---
FROM node:24-alpine AS build
WORKDIR /app
# Install deps first (cache layer); package-lock.json is the source of truth
COPY package.json package-lock.json ./
RUN npm ci
# Compile TypeScript → dist/
COPY tsconfig.json tsconfig.build.json ./
COPY src ./src
RUN npm run build
# tsc doesn't copy non-TS assets — schema.sql is read at runtime by dist/db/migrate.js
COPY src/db/schema.sql dist/db/schema.sql

# --- Runtime stage ---
FROM node:24-alpine AS runtime
ENV NODE_ENV=production
WORKDIR /app
# Production deps only
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
# Compiled output
COPY --from=build /app/dist ./dist
# Data provisioning scripts + config/data dir (provisioned at runtime, not built)
COPY scripts ./scripts
RUN mkdir -p config/data
EXPOSE 8000
# .env is injected via docker-compose environment; dotenv/config tolerates absence
CMD ["node", "dist/server/index.js"]
