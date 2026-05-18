# Halal Portfolio AI

**Sharia-compliant stock portfolio builder powered by RAG**

> 🚧 **Actively under development** — core engine and RAG pipeline are being built. Stars and feedback welcome.

---

## The Problem

Muslim investors face a unique challenge: most popular ETFs — SCHD, VYMI, IVW — hold hundreds of stocks, many of which are haram (non-permissible under Sharia law). Banks, alcohol companies, weapons manufacturers, and casinos sit alongside perfectly halal businesses inside the same fund. You cannot simply buy an ETF off the shelf.

Current alternatives are unsatisfying:
- **Halal ETFs** exist but have limited selection, higher fees, and opaque screening methodologies.
- **Manual screening** is time-consuming, error-prone, and hard to rebalance quarterly.
- **Robo-advisors** don't account for purification obligations on mixed-income stocks.

There is no tool that automates the full lifecycle — screen → build → purify → rebalance — while letting you ask plain-English questions about your portfolio.

**Halal Portfolio AI** fills that gap.

---

## What It Does

| Feature | Description |
|---|---|
| **Portfolio Builder** | Scrapes ETF holdings (SCHD, VYMI, IVW), runs Sharia screening per stock, groups halal stocks by sector, and allocates money using a sector-weighted formula |
| **Purification Engine** | Calculates the exact USD amount to donate to charity each quarter from dividends and capital gains on mixed-income stocks |
| **Quarterly Redistribution** | Detects weight drift and compliance changes, computes delta fractions, and outputs buy/sell instructions with purification adjustments |
| **RAG-Powered Chat** | Ask natural language questions about your portfolio and Islamic finance — grounded answers sourced from your holdings data and a curated fiqh knowledge base |
| **React Dashboard** | Visual allocation breakdown by sector, purification tracker, redistribution history, and interactive Q&A *(in progress)* |

---

## Architecture

![Architecture](docs/architecture.png)

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **API** | FastAPI + Uvicorn |
| **Market Data** | yfinance |
| **Scraping** | BeautifulSoup4 + requests |
| **Database** | PostgreSQL + SQLAlchemy |
| **Vector Store** | ChromaDB |
| **RAG Framework** | LangChain |
| **LLM** | Anthropic Claude API (claude-sonnet-4-6) |
| **Frontend** | React + Recharts |
| **Containerisation** | Docker + Docker Compose |

---

## Project Structure

```
halal-portfolio-ai/
├── src/
│   ├── data/
│   │   ├── etf_scraper.py      # Scrape SCHD, VYMI, IVW holdings pages
│   │   ├── stock_data.py       # Prices, dividends, sectors via yfinance
│   │   └── sharia_screener.py  # Halal/haram/mixed classification (AAOIFI)
│   ├── portfolio/
│   │   ├── builder.py          # Sector-weighted allocation (Method 1)
│   │   ├── purification.py     # Quarterly charity calculator (pluggable)
│   │   └── redistribution.py  # Delta fractions, buy/sell instructions
│   ├── rag/
│   │   ├── embeddings.py       # Document chunking and embedding pipeline
│   │   ├── vector_store.py     # ChromaDB setup and retrieval
│   │   └── chain.py            # LangChain RAG chain + Claude
│   ├── api/
│   │   └── main.py             # FastAPI endpoints
│   └── utils/
│       ├── sectors.py          # ETF ↔ Yahoo Finance sector name mapping
│       └── config.py           # Centralised configuration
├── evaluation/
│   ├── rag_eval.py             # Retrieval accuracy, relevance, faithfulness
│   └── test_queries.json       # 15 benchmark queries with expected answers
├── tests/                      # pytest unit tests
├── docs/                       # Architecture diagrams
├── frontend/                   # React dashboard (in progress)
├── requirements.txt
├── docker-compose.yml
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (for PostgreSQL)
- An [Anthropic API key](https://console.anthropic.com/)

### Install

```bash
git clone https://github.com/your-username/halal-portfolio-ai.git
cd halal-portfolio-ai

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run

```bash
# Start PostgreSQL
docker compose up db -d

# Start the API server
uvicorn src.api.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Run RAG evaluation
python -m evaluation.rag_eval --queries evaluation/test_queries.json
```

API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs) once the server is running.

---

## Key Formulas

### Money Allocation (Method 1)

Given `N` target companies and `available_cash`:

```
companies_in_sector = (sector_weight / 100) × N

stock_units         = sector_weight / companies_in_sector
sector_avg_price    = mean(open_price for all stocks in sector)
stock_cost          = sector_avg_price × stock_units

ratio               = stock_cost / Σ(all stock_costs)
money_allocated     = available_cash × ratio
stock_fraction      = money_allocated / open_price
```

If an entire sector contains no halal stocks, its weight is redistributed equally across the remaining investable sectors before the formula runs.

### Purification

**Dividend purification** (quarterly):
```
dividend_purification = (dividend_rate / 4) × old_fraction
                        × purification_pct × 0.7
```
The `0.7` factor adjusts for ~30% withholding tax drag (configurable per jurisdiction).

**Growth purification** (on redistribution, only when `delta < 0` and price appreciated):
```
growth_purification = (new_price - old_price) × |delta_fraction|
                      × purification_pct
```

### Redistribution

```
delta_fraction  = new_fraction - old_fraction
adjusted_amount = (new_open_price - old_open_price) × delta_fraction
```

`delta_fraction > 0` → buy more shares  
`delta_fraction < 0` → sell shares (and check growth purification)

---

## Sharia Screening Methodology

Stocks are classified using the **AAOIFI standard** — three sequential screens:

1. **Qualitative screen** — exclude companies whose primary business involves alcohol, tobacco, weapons, adult content, pork, conventional banking, or gambling.

2. **Financial ratios screen** — exclude companies where:
   - Interest-bearing debt / total assets **> 33%**
   - Non-permissible income / total revenue **> 5%**
   - Accounts receivable / total assets **> 45%**

3. **Purification** — stocks that pass the qualitative screen but have some non-permissible revenue (< 5%) are classified as **mixed**. The investor holds them but donates the proportional income share to charity each quarter.

The screener is pluggable: a Zoya API integration is planned to replace the manual override list with automated, scholar-reviewed verdicts.

---

## ETFs Tracked

| Ticker | Name | Strategy | Default Holdings |
|---|---|---|---|
| **SCHD** | Schwab US Dividend Equity ETF | US dividend income | 30 halal picks |
| **VYMI** | Vanguard International High Dividend Yield ETF | International dividend income | 30 halal picks |
| **IVW** | iShares S&P 500 Growth ETF | US large-cap growth | 20 halal picks |

---

## Evaluation

The `evaluation/` directory contains a benchmark suite for the RAG pipeline:

**Retrieval accuracy (Recall@K):** Given a test query, does at least one of the top-K retrieved chunks contain the expected answer snippet?

**Answer relevance:** An LLM judge (Claude Haiku) rates how well the generated answer addresses the question on a 1–5 scale.

**Faithfulness:** The judge identifies claims in the answer that are not grounded in the retrieved context. Faithfulness = supported claims / total claims.

Run the full benchmark:
```bash
python -m evaluation.rag_eval --queries evaluation/test_queries.json \
                               --output evaluation/results.json
```

The `test_queries.json` file contains 15 curated queries across four categories: portfolio questions, purification questions, Islamic finance knowledge, and ETF explanations.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built for Muslim investors who want modern portfolio management without compromising their principles.*
