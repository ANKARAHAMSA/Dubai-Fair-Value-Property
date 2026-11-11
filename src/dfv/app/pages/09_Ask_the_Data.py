"""Page 9: Ask the Data — Safe Natural-Language Property Search (LLM + Rule Fallback)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ask the Data — DubaiFairValue", layout="wide")

from dfv.app.components.data_loader import (  # noqa: E402
    area_sqm_to_display,
    load_transactions_view,
    ppsqm_to_display,
    render_sidebar_controls,
    unit_suffix,
)
from dfv.search.executor import execute_property_search  # noqa: E402
from dfv.search.llm import parse_query_nl  # noqa: E402
from dfv.search.schema import SearchGuardrailError  # noqa: E402

mode, data_dir, _, unit = render_sidebar_controls()
tx_df = load_transactions_view(mode, str(data_dir))
known_areas = sorted(tx_df["area_name"].unique().tolist())
u_lbl = unit_suffix(unit)

st.title("Ask the Data (Natural-Language Search)")
st.markdown(
    "Search verified transactions using natural English. Requests map strictly to validated structured "
    "filters and execute exclusively via parameterized DuckDB queries (no LLM SQL generation)."
)

examples = [
    "2-bed near a metro under 1.5M with good yield",
    "Ready 1-bed apartment in Dubai Marina below 1.8M",
    "Off-plan 4 bedroom villa in Arabian Ranches under 6M",
]
if "nl_search_box" not in st.session_state:
    st.session_state["nl_search_box"] = examples[0]

example_triggered = False
ex_cols = st.columns(len(examples))
for idx, ex in enumerate(examples):
    if ex_cols[idx].button(ex, key=f"ex_btn_{idx}"):
        st.session_state["nl_search_box"] = ex
        example_triggered = True

query_text = st.text_input(
    "Natural-Language Query",
    max_chars=400,
    key="nl_search_box",
)
run_clicked = st.button("Search Dataset", type="primary") or example_triggered

if run_clicked or "last_nl_search" not in st.session_state:
    try:
        filters, used_llm, parser_note = parse_query_nl(query_text, known_areas=known_areas)
        features_path = data_dir / "features.parquet"
        yield_path = data_dir / "analysis" / "yield_risk.json"
        if not yield_path.exists():
            yield_path = data_dir / "yield_risk.json"
        out = execute_property_search(
            filters, features_path=features_path, yield_json_path=yield_path
        )
        st.session_state["last_nl_search"] = {
            "ok": True,
            "used_llm": used_llm,
            "parser_note": parser_note,
            **out,
        }
    except SearchGuardrailError as exc:
        st.session_state["last_nl_search"] = {"ok": False, "error": str(exc)}
    except ValueError as exc:
        st.session_state["last_nl_search"] = {"ok": False, "error": str(exc)}
    except Exception:
        st.session_state["last_nl_search"] = {
            "ok": False,
            "error": "Search failed safely while parsing filters. Try rephrasing your query.",
        }

res = st.session_state["last_nl_search"]
if not res["ok"]:
    st.error(res["error"])
    st.stop()

badge = (
    "LLM Structured Extraction" if res["used_llm"] else "Deterministic Rule Fallback (No API Key)"
)
st.success(f"**{res['explanation']}**  |  Parser: `{badge}` (`{res['parser_note']}`)")

with st.expander("Inspect Validated JSON Filters"):
    st.json(res["filters"])

s1, s2, s3 = st.columns(3)
s1.metric("Total Matching Transactions", f"{res['total_matches']:,}")
med_p = res["summary"].get("median_price_aed")
med_psq = res["summary"].get("median_ppsqm")
disp_med_psq = ppsqm_to_display(med_psq, unit) if med_psq is not None else None
s2.metric("Matched Median Price", f"AED {med_p:,.0f}" if med_p is not None else "n/a")
s3.metric(
    f"Matched Median AED/{u_lbl}", f"AED {disp_med_psq:,.0f}" if disp_med_psq is not None else "n/a"
)

rows = res["results"]
if not rows:
    st.warning(
        "No matching listings found under those exact bounds. Try loosening price or distance constraints."
    )
else:
    res_df = pd.DataFrame(rows)
    if "area_sqm" in res_df.columns:
        res_df[f"area_{u_lbl}"] = res_df["area_sqm"].map(
            lambda v: round(area_sqm_to_display(v, unit), 1)
        )
    if "price_per_sqm" in res_df.columns:
        res_df[f"price_per_{u_lbl}_aed"] = res_df["price_per_sqm"].map(
            lambda v: round(ppsqm_to_display(v, unit), 0)
        )
    if unit == "sqft":
        res_df = res_df.drop(
            columns=[c for c in ("area_sqm", "price_per_sqm") if c in res_df.columns]
        )
    elif "price_per_sqm" in res_df.columns and "price_per_sqm_aed" in res_df.columns:
        res_df = res_df.drop(columns=["price_per_sqm"])
    st.dataframe(res_df, width="stretch", hide_index=True)
