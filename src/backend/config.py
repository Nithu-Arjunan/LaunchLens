"""Central configuration for LaunchLens.

All tunable constants live here so the rest of the code never hardcodes a
value inline.
"""

# --- Short-term memory / summarization -------------------------------------
# How many messages we let pile up before summarizing older turns.
SUMMARIZE_AFTER_N_MESSAGES = 20
# How many of the most recent messages to keep verbatim after a summary.
KEEP_RECENT_N_MESSAGES = 4

# --- LLM -------------------------------------------------------------------
# Default OpenAI model. Read at call time as the fallback for the

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# --- External API endpoints ------------------------------------------------
OXYLABS_ENDPOINT = "https://realtime.oxylabs.io/v1/queries"
SERPAPI_ENDPOINT = "https://serpapi.com/search"

# --- Request defaults ------------------------------------------------------
HTTP_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RESULTS = 10
DEFAULT_TRENDS_TIMEFRAME = "today 12-m"

# --- CLI defaults ----------------------------------------------------------
DEFAULT_THREAD_ID = "default-thread"
DEFAULT_CHECKPOINT_DB = "checkpoints.db"

# --- Regions ---------------------------------------------------------------
DEFAULT_REGION = "United States"

REGION_ALIASES = {
    "us": "United States",
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "america": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "emirates": "United Arab Emirates",
    "dubai": "United Arab Emirates",
    "abu dhabi": "United Arab Emirates",
}

SERPAPI_GEO_BY_REGION = {
    "United States": "US",
    "United Kingdom": "GB",
    "India": "IN",
    "United Arab Emirates": "AE",
    "Canada": "CA",
    "Australia": "AU",
}

SERPAPI_GL_BY_REGION = {
    "United States": "us",
    "United Kingdom": "gb",
    "India": "in",
    "United Arab Emirates": "ae",
    "Canada": "ca",
    "Australia": "au",
}

SERPAPI_HL_BY_REGION = {
    "United States": "en",
    "United Kingdom": "en",
    "India": "en",
    "United Arab Emirates": "en",
    "Canada": "en",
    "Australia": "en",
}

AMAZON_DOMAIN_BY_REGION = {
    "United States": "com",
    "United Kingdom": "co.uk",
    "India": "in",
    "United Arab Emirates": "ae",
    "Canada": "ca",
    "Australia": "com.au",
}

# --- Memory routing --------------------------------------------------------
# Keywords that deterministically route a question to the memory node.
MEMORY_KEYWORDS = (
    "earlier",
    "previous",
    "previously",
    "remember",
    "memory",
    "history",
    "conversation",
    "discussed",
    "asked",
    "before",
    "so far",
    "summarize our",
    "what did i",
    "what have i",
    "what was my",
    "what products",
    "how many products",
)
