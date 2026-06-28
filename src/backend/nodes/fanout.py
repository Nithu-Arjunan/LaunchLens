"""Fan-out nodes — each calls one external data source in parallel."""

import logging

from api import (
    fetch_amazon_product,
    fetch_amazon_search,
    fetch_google_news,
    fetch_google_trends,
)
from state import AgentState

from . import _helpers

logger = logging.getLogger(__name__)


def fetch_trends_node(state: AgentState) -> dict:
    """Fan-out branch: demand signal via Google Trends."""
    query = _helpers._search_query(state)
    region = _helpers._target_region(state)
    logger.info("fanout.fetch_trends: start region=%s search_query=%r", region, query)
    result = fetch_google_trends(query, region=region)
    logger.info("fanout.fetch_trends: done source=%s", result.get("source"))
    return {"trends_result": result}


def fetch_amazon_node(state: AgentState) -> dict:
    """Fan-out branch: pricing/competitor signal via Amazon search."""
    query = _helpers._search_query(state)
    region = _helpers._target_region(state)
    logger.info("fanout.fetch_amazon: start region=%s search_query=%r", region, query)
    result = fetch_amazon_search(query, region=region)
    listings = result.get("listings", [])
    logger.info(
        "fanout.fetch_amazon: done source=%s region=%s domain=%s listings=%s error=%s preview_titles=%s",
        result.get("source"),
        result.get("region"),
        result.get("domain"),
        len(listings),
        bool(result.get("error")),
        _helpers._preview_titles(listings),
    )
    if result.get("error"):
        logger.warning("fanout.fetch_amazon: degraded error=%s", result.get("error"))
    elif "listings" in result and not listings:
        logger.warning(
            "fanout.fetch_amazon: no listings returned region=%s domain=%s search_query=%r",
            result.get("region"),
            result.get("domain"),
            query,
        )
    return {"amazon_result": result}


def fetch_amazon_products_node(state: AgentState) -> dict:
    """Fan-out branch: enrich discovered Amazon competitors via product detail pages."""
    query = _helpers._search_query(state)
    region = _helpers._target_region(state)
    amazon_result = state.get("amazon_result") or {}
    domain = amazon_result.get("domain")
    asins = _helpers._top_asins_from_amazon_result(amazon_result)
    logger.info(
        "fanout.fetch_amazon_products: start region=%s domain=%s search_query=%r asins=%s",
        region,
        domain,
        query,
        asins,
    )

    products = [
        fetch_amazon_product(asin, region=region, domain=domain)
        for asin in asins
    ]
    errors = [item.get("error") for item in products if item.get("error")]
    result = {
        "source": "amazon_products",
        "query": query,
        "region": region,
        "domain": domain,
        "stub": False,
        "products": products,
    }
    if not asins:
        result["error"] = "No ASINs available from amazon_search results."
    elif errors:
        result["error"] = "; ".join(errors)

    logger.info(
        "fanout.fetch_amazon_products: done products=%s errors=%s preview_titles=%s",
        len(products),
        len(errors),
        _helpers._preview_titles([item.get("product") or {} for item in products]),
    )
    if result.get("error"):
        logger.warning("fanout.fetch_amazon_products: degraded error=%s", result.get("error"))
    return {"amazon_products_result": result}


def fetch_news_node(state: AgentState) -> dict:
    """Fan-out branch: market context via Google News."""
    query = _helpers._search_query(state)
    region = _helpers._target_region(state)
    logger.info("fanout.fetch_news: start region=%s search_query=%r", region, query)
    result = fetch_google_news(query, region=region)
    logger.info(
        "fanout.fetch_news: done source=%s articles=%s",
        result.get("source"),
        len(result.get("articles", [])),
    )
    return {"news_result": result}
