"""Safe natural-language search: strict filter schema, LLM + rules parsers, parameterized DuckDB executor."""

from dfv.search.executor import execute_property_search
from dfv.search.llm import parse_query_nl
from dfv.search.rules import parse_query_rules
from dfv.search.schema import PropertyFilters, SearchGuardrailError, summarize_filters

__all__ = [
    "PropertyFilters",
    "SearchGuardrailError",
    "execute_property_search",
    "parse_query_nl",
    "parse_query_rules",
    "summarize_filters",
]
