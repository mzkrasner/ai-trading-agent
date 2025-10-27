# Nocturne: AI Trading Agent on Hyperliquid

This project implements an AI-powered trading agent that leverages LLM models to analyze real-time market data, social sentiment, and technical indicators to make informed trading decisions on the Hyperliquid decentralized exchange. The agent runs in a continuous loop, monitoring specified cryptocurrency assets at configurable intervals, executing trades with sophisticated risk management including dynamic TP/SL adjustments and intelligent order cleanup.

## Table of Contents

- [Disclaimer](#disclaimer)
- [Architecture](#architecture)
- [Nocturne Live Agents](#nocturne-live-agents)
- [Structure](#structure)
- [Env Configuration](#env-configuration)
- [Usage](#usage)
- [Tool Calling](#tool-calling)
- [Deployment to EigenCloud](#deployment-to-eigencloud)

## Disclaimer

There is no guarantee of any returns. This code has not been audited. Please use at your own risk.

## Architecture

See the full [Architecture Documentation](docs/ARCHITECTURE.md) for subsystems, data flow, and design principles.

![Architecture Diagram](docs/architecture.png)

## Nocturne Live Agents 

- GPT-5 Pro: [Portfolio Dashboard](https://hypurrscan.io/address/0xa049db4b3dfcb25c3092891010a629d987d26113) | [Live Logs](https://35.190.43.182/logs/0xC0BE8E55f469c1a04c0F6d04356828C5793d8a9D) (Seeded with $200)
- DeepSeek R1: [Portfolio Dashboard](https://hypurrscan.io/address/0xa663c80d86fd7c045d9927bb6344d7a5827d31db) | [Live Logs](https://35.190.43.182/logs/0x4da68B78ef40D12f378b8498120f2F5A910Af1aD) (Seeded with $100) -- PAUSED
- Grok 4: [Portfolio Dashboard](https://hypurrscan.io/address/0x3c71f3cf324d0133558c81d42543115ef1a2be79) | [Live Logs](https://35.190.43.182/logs/0xe6a9f97f99847215ea5813812508e9354a22A2e0) (Seeded with $100) -- PAUSED

## Structure
- `src/main.py`: Entry point, handles main trading loop with order reconciliation and cleanup logic.
- `src/agent/decision_maker.py`: LLM logic for trade decisions via OpenRouter (supports tool calling for additional indicators).
- `src/indicators/hyperliquid_indicators.py`: Fetches technical indicators directly from Hyperliquid candle data.
- `src/sentiment/x_sentiment.py`: Analyzes X/Twitter sentiment using Grok API for market context.
- `src/trading/hyperliquid_api.py`: Executes trades on Hyperliquid with retry logic and order management.
- `src/config_loader.py`: Centralized config loaded from `.env`.

## Key Features
- **Multi-source analysis**: Technical indicators (5m + 4h timeframes), X/Twitter sentiment, and market metrics
- **Dynamic risk management**: LLM can update TP/SL on existing positions without closing them
- **Intelligent order cleanup**: Automatically detects and removes orphaned/duplicate orders
- **Accurate fill detection**: Validates order fills from exchange response (not timing-dependent)
- **Comprehensive logging**: Detailed decision rationale, trade diary, and performance tracking

## Env Configuration
Populate `.env` (use `.env.example` as reference):

**Required:**
- `HYPERLIQUID_PRIVATE_KEY` (or LIGHTER_PRIVATE_KEY)
- `OPENROUTER_API_KEY`
- `LLM_MODEL` (e.g., "deepseek/deepseek-chat-v3.1", "x-ai/grok-4")
- `GROK_API_KEY` (for X/Twitter sentiment analysis)
- `ASSETS` (space or comma-separated, e.g., "BTC ETH SOL")
- `INTERVAL` (e.g., "5m", "1h")

**Optional:**
- `TAAPI_API_KEY` (for additional indicators via tool calling, fallback when Hyperliquid data unavailable)
- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_REFERER`, `OPENROUTER_APP_TITLE` (for OpenRouter metadata)
- `API_HOST` (default: `0.0.0.0`), `API_PORT` (default: `3000`)

### Obtaining API Keys
- **HYPERLIQUID_PRIVATE_KEY**: Generate an Ethereum-compatible private key for Hyperliquid. Use tools like MetaMask or `eth_account` library. For security, never share this key.
- **OPENROUTER_API_KEY**: Create an account at [OpenRouter.ai](https://openrouter.ai/), then generate an API key in your account settings.
- **GROK_API_KEY**: Sign up at [x.ai](https://x.ai/) and generate an API key for Grok access.
- **TAAPI_API_KEY** (optional): Sign up at [TAAPI.io](https://taapi.io/) and generate an API key from your dashboard.

## Usage
Run: `poetry run python src/main.py --assets BTC ETH --interval 1h`

### Local API Endpoints
When the agent runs, it also serves a minimal API:
- `GET /diary?limit=200` — returns recent JSONL diary entries as JSON.
- `GET /logs?path=llm_requests.log&limit=2000` — tails the specified log file.

Configure bind host/port via env:
- `API_HOST` (default `0.0.0.0`)
- `API_PORT` or `APP_PORT` (default `3000`)

Docker:
```bash
docker build --platform linux/amd64 -t trading-agent .
docker run --rm -p 3000:3000 --env-file .env trading-agent
# Now: curl http://localhost:3000/diary
```

## Data Sources

### Technical Indicators
The agent primarily uses **Hyperliquid native candle data** to calculate indicators (EMA, MACD, RSI, ATR, etc.) for both 5-minute intraday and 4-hour longer-term analysis. This provides:
- Zero API rate limits
- Lower latency
- Consistent data source with exchange

The LLM can also dynamically fetch **TAAPI indicators** via tool calls for additional analysis or fallback. See [TAAPI Indicators](https://taapi.io/indicators/) for 200+ available indicators.

### Market Sentiment
The agent uses **Grok API** to analyze recent X/Twitter posts about each asset, providing:
- Post themes and sentiment words
- Volume metrics (spike detection, post frequency)
- Sentiment velocity (15min, 1hr shifts)
- Whale activity and institutional mentions
- Price expectations from community

This sentiment data is provided as **observational context** - the LLM interprets and weighs it alongside technical analysis.

## Deployment to EigenCloud

EigenCloud (via EigenX CLI) allows deploying this trading agent in a Trusted Execution Environment (TEE) with secure key management.

### Prerequisites
- Allowlisted Ethereum account (Sepolia for testnet). Request onboarding at [EigenCloud Onboarding](https://onboarding.eigencloud.xyz).
- Docker installed.
- Sepolia ETH for deployments.

### Installation
#### macOS/Linux
```bash
curl -fsSL https://eigenx-scripts.s3.us-east-1.amazonaws.com/install-eigenx.sh | bash
```

#### Windows
```bash
curl -fsSL https://eigenx-scripts.s3.us-east-1.amazonaws.com/install-eigenx.ps1 | powershell -
```

### Initial Setup
```bash
docker login
eigenx auth login  # Or eigenx auth generate --store (if you don't have a eth account, keep this account separate from your trading account)
```

### Deploy the Agent
From the project directory:
```bash
cp .env.example .env
# Edit .env: set ASSETS, INTERVAL, API keys
eigenx app deploy
```

### Monitoring
```bash
eigenx app info --watch
eigenx app logs --watch
```

### Updates
Edit code or .env, then:
```bash
eigenx app upgrade <app-name>
```

For full CLI reference, see the [EigenX Documentation](https://github.com/Layr-Labs/eigenx-cli).
