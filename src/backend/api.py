"""
LaunchLens — core data-source functions.

One plain function per external data source. No LangGraph/LangChain
awareness here on purpose — these get called two ways:
  1. Directly, inside fan-out nodes (deterministic path)
  2. Wrapped with @tool, bound to the agent (LLM-chosen path)

fetch_amazon_* go through Oxylabs; fetch_google_trends and
fetch_google_news go through SerpApi.

Each Oxylabs call has a SCOPED try/except that turns a network failure
into a degraded result (empty listings + an `error` field) so the graph
can continue instead of crashing. SerpApi calls deliberately let
exceptions propagate for now — broader hardening (retries, backoff) is
still deferred.
"""

import logging
import os
from typing import Literal

import requests
from pydantic import BaseModel, Field

from config import (
    AMAZON_DOMAIN_BY_REGION,
    DEFAULT_MAX_RESULTS,
    DEFAULT_REGION,
    DEFAULT_TRENDS_TIMEFRAME,
    HTTP_TIMEOUT_SECONDS,
    OXYLABS_ENDPOINT,
    REGION_ALIASES,
    SERPAPI_ENDPOINT,
    SERPAPI_GEO_BY_REGION,
    SERPAPI_GL_BY_REGION,
    SERPAPI_HL_BY_REGION,
)

logger = logging.getLogger(__name__)


class AmazonListing(BaseModel):
    title: str | None = None
    price: str | float | int | None = None
    currency: str | None = None
    rating: float | str | None = None
    rating_count: int | str | None = None
    url: str | None = None
    asin: str | None = None


class AmazonSearchResult(BaseModel):
    source: Literal["amazon_search"] = "amazon_search"
    query: str
    max_results: int
    region: str
    domain: str
    stub: bool = False
    listings: list[AmazonListing] = Field(default_factory=list)
    error: str | None = None


class AmazonProduct(BaseModel):
    title: str | None = None
    asin: str | None = None
    brand: str | None = None
    price: str | float | int | None = None
    currency: str | None = None
    rating: float | str | None = None
    rating_count: int | str | None = None
    availability: str | None = None
    description: str | None = None
    bullet_points: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    url: str | None = None


class AmazonProductResult(BaseModel):
    source: Literal["amazon_product"] = "amazon_product"
    query: str
    region: str
    domain: str
    stub: bool = False
    product: AmazonProduct | None = None
    error: str | None = None


def _normalized_region(region: str | None) -> str:
    cleaned = (region or "").strip()
    if not cleaned:
        return DEFAULT_REGION
    return REGION_ALIASES.get(cleaned.casefold(), cleaned)


def _amazon_domain(region: str) -> str:
    """Resolve the Amazon marketplace domain for a region, allowing an
    explicit OXYLABS_AMAZON_DOMAIN override."""
    return os.environ.get("OXYLABS_AMAZON_DOMAIN") or AMAZON_DOMAIN_BY_REGION.get(region, "com")


def _dump_amazon_result(result: AmazonSearchResult) -> dict:
    return result.model_dump(exclude_none=True)


def _dump_amazon_product_result(result: AmazonProductResult) -> dict:
    return result.model_dump(exclude_none=True)


def _extract_oxylabs_product(data: dict) -> dict:
    first_result = (data.get("results") or [{}])[0]
    content = first_result.get("content") or {}
    if not isinstance(content, dict):
        return {}
    for key in ("product", "result", "details"):
        value = content.get(key)
        if isinstance(value, dict):
            return value
    return content


def _list_from_value(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _string_from_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, list):
        text_items = [str(item).strip() for item in value if item is not None and str(item).strip()]
        return "\n".join(text_items) if text_items else None
    return str(value)


def _oxylabs_job_error(data: dict) -> str | None:
    job = data.get("job") or {}
    status = job.get("status")
    if status and status not in {"done", "successful"}:
        source = job.get("source")
        category_id = None
        for item in job.get("context") or []:
            if item.get("key") == "category_id":
                category_id = item.get("value")
                break
        return (
            f"Oxylabs job status={status}"
            f" source={source}"
            f" category_id={category_id}"
        )
    return None


def fetch_google_trends(
    query: str,
    timeframe: str = DEFAULT_TRENDS_TIMEFRAME,
    region: str = DEFAULT_REGION,
) -> dict:
    """Real SerpApi Google Trends call.

    SerpApi auth is a single api_key query param. `engine` tells SerpApi
    which Google product to scrape (same pattern as Oxylabs' `source`).

    Real response shape: interest_over_time.timeline_data is a list of
    {date, values: [{query, value, extracted_value}]} points. We collapse
    that into a simple average interest score (0-100) rather than passing
    the raw timeseries — good enough signal for routing/verdict, and far
    less to stuff into the agent's prompt.
    """
    api_key = os.environ["SERPAPI_API_KEY"]
    region = _normalized_region(region)

    params = {
        "engine": "google_trends",
        "q": query,
        "date": timeframe,
        "api_key": api_key,
    }
    geo = SERPAPI_GEO_BY_REGION.get(region)
    if geo:
        params["geo"] = geo

    response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()

    timeline_data = data.get("interest_over_time", {}).get("timeline_data", [])
    values = []
    for point in timeline_data:
        for v in point.get("values", []):
            extracted = v.get("extracted_value", v.get("value"))
            if isinstance(extracted, (int, float)):
                values.append(extracted)

    average_interest = round(sum(values) / len(values), 1) if values else None

    return {
        "source": "google_trends",
        "query": query,
        "timeframe": timeframe,
        "region": region,
        "stub": False,
        "average_interest": average_interest,
        "data_points": len(timeline_data),
    }


def fetch_amazon_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    region: str = DEFAULT_REGION,
) -> dict:
    """Real Oxylabs Amazon Search call.

    Oxylabs auth is HTTP Basic Auth (username/password), not a single API
    key. One endpoint handles every source — `source: "amazon_search"` in
    the JSON body is what tells Oxylabs which scraper to run.
    `parse: true` makes it return structured JSON instead of raw HTML.

    The scoped try/except turns a network failure into a degraded result
    so the graph can continue. (A known local issue is Oxylabs' Hetzner
    IP ranges being blocked on some dev machines — confirmed working from
    Postman and GCP Cloud Shell, so it is environment-specific.)
    """
    username = os.environ["OXYLABS_USERNAME"]
    password = os.environ["OXYLABS_PASSWORD"]
    region = _normalized_region(region)
    domain = _amazon_domain(region)

    payload = {
        "source": "amazon_search",
        "domain": domain,
        "query": query,
        "parse": True,
    }

    try:
        response = requests.post(
            OXYLABS_ENDPOINT,
            auth=(username, password),
            json=payload,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        # Real response shape: results -> [0] -> content -> results -> organic
        organic = data["results"][0]["content"]["results"]["organic"]

        listings = [
            AmazonListing(
                title=item.get("title"),
                price=item.get("price"),
                currency=item.get("currency"),
                rating=item.get("rating"),
                rating_count=item.get("rating_count") or item.get("reviews_count"),
                url=item.get("url"),
                asin=item.get("asin"),
            )
            for item in organic[:max_results]
        ]
    except requests.exceptions.RequestException as e:
        logger.warning(
            "fetch_amazon_search: Oxylabs call failed (%s); returning degraded result. error=%s",
            type(e).__name__,
            e,
        )
        return _dump_amazon_result(
            AmazonSearchResult(
                query=query,
                max_results=max_results,
                region=region,
                domain=domain,
                error=f"{type(e).__name__}: {e}",
            )
        )

    return _dump_amazon_result(
        AmazonSearchResult(
            query=query,
            max_results=max_results,
            region=region,
            domain=domain,
            listings=listings,
        )
    )


def fetch_amazon_product(
    asin: str,
    region: str = DEFAULT_REGION,
    domain: str | None = None,
) -> dict:
    """Real Oxylabs Amazon Product call for a discovered competitor ASIN."""
    username = os.environ["OXYLABS_USERNAME"]
    password = os.environ["OXYLABS_PASSWORD"]
    region = _normalized_region(region)
    domain = domain or _amazon_domain(region)

    payload = {
        "source": "amazon_product",
        "domain": domain,
        "query": asin,
        "parse": True,
    }

    try:
        response = requests.post(
            OXYLABS_ENDPOINT,
            auth=(username, password),
            json=payload,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        job_error = _oxylabs_job_error(data)
        if job_error:
            return _dump_amazon_product_result(
                AmazonProductResult(
                    query=asin,
                    region=region,
                    domain=domain,
                    error=job_error,
                )
            )

        raw_product = _extract_oxylabs_product(data)
        product = AmazonProduct(
            title=_string_from_value(raw_product.get("title")),
            asin=_string_from_value(raw_product.get("asin")) or asin,
            brand=_string_from_value(raw_product.get("brand")),
            price=raw_product.get("price"),
            currency=_string_from_value(raw_product.get("currency")),
            rating=raw_product.get("rating"),
            rating_count=raw_product.get("rating_count") or raw_product.get("reviews_count"),
            availability=_string_from_value(raw_product.get("availability") or raw_product.get("stock")),
            description=_string_from_value(raw_product.get("description")),
            bullet_points=_list_from_value(
                raw_product.get("bullet_points") or raw_product.get("features")
            ),
            images=_list_from_value(raw_product.get("images")),
            categories=_list_from_value(raw_product.get("categories")),
            url=_string_from_value(raw_product.get("url")),
        )
    except requests.exceptions.RequestException as e:
        logger.warning(
            "fetch_amazon_product: Oxylabs call failed (%s); returning degraded result. error=%s",
            type(e).__name__,
            e,
        )
        return _dump_amazon_product_result(
            AmazonProductResult(
                query=asin,
                region=region,
                domain=domain,
                error=f"{type(e).__name__}: {e}",
            )
        )

    return _dump_amazon_product_result(
        AmazonProductResult(
            query=asin,
            region=region,
            domain=domain,
            product=product,
        )
    )


def fetch_google_news(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    region: str = DEFAULT_REGION,
) -> dict:
    """Real SerpApi Google News call (engine=google_news).

    Real response shape: news_results is a list of
    {title, link, source: {name}, date, snippet}.
    """
    api_key = os.environ["SERPAPI_API_KEY"]
    region = _normalized_region(region)

    params = {
        "engine": "google_news",
        "q": query,
        "api_key": api_key,
    }
    gl = SERPAPI_GL_BY_REGION.get(region)
    hl = SERPAPI_HL_BY_REGION.get(region)
    if gl:
        params["gl"] = gl
    if hl:
        params["hl"] = hl

    response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()

    news_results = data.get("news_results", [])

    articles = [
        {
            "title": item.get("title"),
            "link": item.get("link"),
            "source": (item.get("source") or {}).get("name"),
            "date": item.get("date"),
            "snippet": item.get("snippet"),
        }
        for item in news_results[:max_results]
    ]

    return {
        "source": "google_news",
        "query": query,
        "max_results": max_results,
        "region": region,
        "stub": False,
        "articles": articles,
    }
