"""
All State variables required in the langgraph nodes 
"""

from typing import Annotated, Literal, NotRequired, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

RouteType = Literal["demand", "pricing", "full_report", "memory"]

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # summary of older turns
    summary: NotRequired[str]

    #Stores the type of user query and extract data from query
    route: NotRequired[RouteType]
    route_reason: NotRequired[str]
    target_region: NotRequired[str]
    target_region_reason: NotRequired[str]
    search_query: NotRequired[str]
    search_query_reason: NotRequired[str]

    # Fan-out node results
    trends_result: NotRequired[dict]
    amazon_result: NotRequired[dict]
    amazon_products_result: NotRequired[dict]
    news_result: NotRequired[dict]

    verdict: NotRequired[dict]




