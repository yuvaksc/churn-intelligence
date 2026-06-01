# docker/frontend.Dockerfile
#
# MULTI-STAGE BUILD — a new concept:
#
# Problem: building a Next.js app requires node_modules (~300MB of dev tools).
# Running it only needs the compiled output. Why ship build tools in production?
#
# Solution: two FROM statements = two stages.
#   Stage 1 "builder": full node, installs everything, compiles the app.
#   Stage 2 "runner" : minimal node, copies ONLY the compiled output from stage 1.
#
# Final image only contains what stage 2 put in — build tools are discarded.
# Result: production image ~3x smaller than a single-stage build.

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM node:20-slim AS builder

WORKDIR /app

# Install dependencies first (cached if package.json unchanged)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

# Copy source code
COPY frontend/ ./

# NEXT_PUBLIC_* variables are embedded at BUILD time, not runtime.
# The browser will call this URL directly, so it must be reachable
# from the user's machine — not from inside the Docker network.
# Override this build arg in docker-compose if your API is elsewhere.
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE

# Build the production app
RUN npm run build

# ── Stage 2: Run ─────────────────────────────────────────────────────────────
FROM node:20-slim AS runner

WORKDIR /app

ENV NODE_ENV=production

# Copy only the built output and runtime from the builder stage.
# NOTHING from the host machine — only what builder produced.
COPY --from=builder /app/public        ./public
COPY --from=builder /app/.next/standalone  ./
COPY --from=builder /app/.next/static  ./.next/static

EXPOSE 3000

# next start uses a minimal standalone server (requires output: "standalone" in next.config)
CMD ["node", "server.js"]
