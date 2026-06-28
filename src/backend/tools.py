"""
LaunchLens — tools for the agent node.

Same core functions from api.py as the fan-out nodes use — these wrappers
just add the docstring + schema an LLM needs to decide WHEN and HOW to
call them. No duplicated API logic: a change to how fetch_amazon_search
hits Oxylabs is picked up by both the fan-out path and this tool path.
"""

from langchain_core.tools import tool

from config import DEFAULT_MAX_RESULTS, DEFAULT_REGION, DEFAULT_TRENDS_TIMEFRAME
from api import (
    fetch_google_trends,
    fetch_amazon_search,
    fetch_amazon_product,
    fetch_google_news,
)


@tool
def trends_tool(
    query: str,
    timeframe: str = DEFAULT_TRENDS_TIMEFRAME,
    region: str = DEFAULT_REGION,
) -> dict:
    """Look up Google Trends interest-over-time for a search term. Use this
    when you need fresh or differently-scoped demand data than what's
    already in the fan-out results — e.g. a narrower query, a different
    timeframe, or a related term the founder mentioned."""
    return fetch_google_trends(query, timeframe, region)


@tool
def amazon_tool(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    region: str = DEFAULT_REGION,
) -> dict:
    """Search Amazon listings for a product query. Use this for follow-up
    competitor or pricing lookups the fan-out result didn't cover — e.g.
    a specific competitor name, a narrower product category, or digging
    into a listing the fan-out surfaced."""
    return fetch_amazon_search(query, max_results, region)


@tool
def amazon_product_tool(
    asin: str,
    region: str = DEFAULT_REGION,
    domain: str | None = None,
) -> dict:
    """Fetch detailed Amazon product information for a specific ASIN found
    in Amazon search results. Use this to inspect competitor details such as
    brand, availability, bullet points, images, and product positioning."""
    return fetch_amazon_product(asin, region=region, domain=domain)


@tool
def news_tool(query: str, region: str = DEFAULT_REGION) -> dict:
    """Search Google News for a query. Use this for follow-up market
    context the fan-out result didn't cover — e.g. recent news about a
    specific competitor or category trend."""
    return fetch_google_news(query, region=region)


ALL_TOOLS = [trends_tool, amazon_tool, amazon_product_tool, news_tool]
