"""Streamlit demo: base vs fine-tuned, side by side.

Two tabs, and the default is deliberately the fast one.

**Replay** browses the saved dev predictions. Every one of the 1,034 questions is already
answered by all three systems in `predictions.jsonl`, so comparisons are instant and --
more importantly -- they are the *same* predictions the headline number was computed from.
Nothing shown here is regenerated on the fly and therefore nothing can quietly disagree
with the metric.

**Live** generates from a typed question. On CPU a 1.5B model takes tens of seconds per
query, and generating base and tuned means doing it twice. That is expected and is stated
in the UI rather than hidden behind a spinner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from text2sql_lora.config import Paths
from text2sql_lora.data.schema import parse_tables_json, render_schema

st.set_page_config(page_title="Text-to-SQL: base vs LoRA", page_icon="::", layout="wide")

PATHS = Paths()
SYSTEMS = ["base_zero_shot", "base_few_shot", "lora_zero_shot"]
SYSTEM_LABELS = {
    "base_zero_shot": "Base (zero-shot)",
    "base_few_shot": "Base (3-shot)",
    "lora_zero_shot": "LoRA fine-tuned",
}


@st.cache_data
def load_metrics(path: str) -> dict | None:
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        return pd.DataFrame()
    return pd.DataFrame(
        json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()
    )


@st.cache_data
def load_schemas(path: str) -> dict:
    file = Path(path)
    return parse_tables_json(file) if file.exists() else {}


metrics = load_metrics(str(PATHS.metrics_json()))
predictions = load_predictions(str(PATHS.predictions_jsonl()))
schemas = load_schemas(str(PATHS.tables_json()))

st.title("Text-to-SQL by LoRA fine-tuning")
st.caption(
    "Qwen2.5-1.5B-Instruct, LoRA adapters on a frozen 4-bit base, evaluated by execution "
    "accuracy on Spider's held-out dev set (databases unseen during training)."
)

if metrics is None:
    st.warning(
        "**No `artifacts/metrics.json` yet.** The training run has not been ingested, so "
        "there are no results to show. Run `notebooks/kaggle_lora_sft.ipynb` on a GPU and "
        "unzip its output into `artifacts/` — see `docs/kaggle_run.md`.",
        icon="!",
    )

# --- headline ----------------------------------------------------------------

if metrics and "headline" in metrics:
    headline = metrics["headline"]
    base = headline["base_zero_shot_execution_accuracy"]
    tuned = headline["lora_zero_shot_execution_accuracy"]

    columns = st.columns(4)
    columns[0].metric("Base, zero-shot", f"{base:.1%}")
    columns[1].metric(
        "Base, 3-shot", f"{headline.get('base_few_shot_execution_accuracy', float('nan')):.1%}"
    )
    columns[2].metric("LoRA fine-tuned", f"{tuned:.1%}", delta=f"{tuned - base:+.1%}")
    columns[3].metric(
        "Examples", f"{metrics['results']['lora_zero_shot']['n']:,}", help="Spider dev"
    )

tab_results, tab_replay, tab_live = st.tabs(
    ["Results", "Replay (instant)", "Live generation (slow)"]
)

# --- results -----------------------------------------------------------------

with tab_results:
    if not metrics:
        st.info("Nothing to show until the training run has been ingested.")
    else:
        st.subheader("Execution accuracy on Spider dev")
        table = pd.DataFrame(metrics["results"]).T
        table.index = [SYSTEM_LABELS.get(name, name) for name in table.index]
        st.dataframe(
            table[["n", "execution_accuracy", "exact_match", "invalid_sql_rate"]].style.format(
                {
                    "execution_accuracy": "{:.4f}",
                    "exact_match": "{:.4f}",
                    "invalid_sql_rate": "{:.4f}",
                }
            ),
            use_container_width=True,
        )
        st.bar_chart(table["execution_accuracy"])

        st.caption(
            "The 3-shot baseline is here on purpose: two data points invite \"your "
            "baseline was a strawman\". Three answer it."
        )

        if metrics.get("by_hardness"):
            st.subheader("By query complexity")
            st.caption(
                "Bucketing is our own component-count heuristic (joins / nesting / "
                "aggregates), **not** Yale's official Spider hardness definition."
            )
            hardness = pd.DataFrame(metrics["by_hardness"]).pivot(
                index="hardness", columns="system", values="execution_accuracy"
            )
            st.dataframe(hardness.style.format("{:.4f}"), use_container_width=True)
            st.bar_chart(hardness)

        history = metrics.get("training", {}).get("log_history", [])
        train_points = [(h["step"], h["loss"]) for h in history if "loss" in h]
        if train_points:
            st.subheader("Training curve")
            curve = pd.DataFrame(train_points, columns=["step", "train loss"]).set_index("step")
            eval_points = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
            if eval_points:
                evaluation = pd.DataFrame(
                    eval_points, columns=["step", "val loss"]
                ).set_index("step")
                curve = curve.join(evaluation, how="outer")
            st.line_chart(curve)
            st.caption(
                "The validation split is carved out of the training set **by database**, "
                "so it never shares a schema with training."
            )

        with st.expander("Run manifest — everything needed to reproduce or challenge this"):
            st.json(metrics.get("manifest", {}))

# --- replay ------------------------------------------------------------------

with tab_replay:
    if predictions.empty:
        st.info("No `artifacts/predictions.jsonl` yet.")
    else:
        st.caption(
            f"All {len(predictions):,} Spider dev questions, already answered by every "
            "system. These are the exact predictions the headline number was computed "
            "from — not regenerated."
        )

        left, right = st.columns([1, 1])
        with left:
            only_fixed = st.checkbox("Only where fine-tuning fixed a base failure", value=True)
        with right:
            database = st.selectbox(
                "Database", ["(all)", *sorted(predictions["db_id"].unique())]
            )

        view = predictions
        if only_fixed:
            view = view[
                view["lora_zero_shot"].apply(lambda d: d["execution_correct"])
                & ~view["base_zero_shot"].apply(lambda d: d["execution_correct"])
            ]
        if database != "(all)":
            view = view[view["db_id"] == database]

        if view.empty:
            st.info("No questions match that filter.")
        else:
            st.write(f"**{len(view):,}** matching questions")
            choice = st.selectbox(
                "Question",
                view.index,
                format_func=lambda i: f"[{view.loc[i, 'db_id']}] {view.loc[i, 'question']}",
            )
            row = view.loc[choice]

            st.markdown(f"**Question:** {row['question']}")
            st.markdown(f"`{row['db_id']}` · complexity **{row['hardness']}**")

            columns = st.columns(3)
            for column, system in zip(columns, SYSTEMS, strict=False):
                payload = row.get(system)
                if not isinstance(payload, dict):
                    continue
                mark = "CORRECT" if payload["execution_correct"] else "WRONG"
                column.markdown(f"**{SYSTEM_LABELS[system]}** — {mark}")
                column.code(payload["prediction"] or "(empty)", language="sql")
                if payload.get("error"):
                    column.caption(f"error: {payload['error']}")

            st.markdown("**Gold**")
            st.code(row["gold"], language="sql")

# --- live --------------------------------------------------------------------

with tab_live:
    st.warning(
        "Generation runs on CPU. A 1.5B model takes **tens of seconds** per query, and "
        "this generates twice (adapter off, then on). That is expected — the GPU was used "
        "for training, not for serving.",
        icon="!",
    )

    if not schemas:
        st.info("Spider not downloaded — run `python scripts/download_data.py`.")
    else:
        db_id = st.selectbox("Database", sorted(schemas))
        schema_text = render_schema(schemas[db_id])
        with st.expander("Schema the model will see"):
            st.code(schema_text, language="sql")

        question = st.text_input("Question", "How many singers are there?")

        if st.button("Generate with both models", type="primary"):
            from text2sql_lora.api.service import Text2SqlService

            service = Text2SqlService(paths=PATHS)
            with st.spinner("Loading model and generating (this is the slow part)..."):
                base_out, tuned_out = service.compare(question=question, db_id=db_id)

            left, right = st.columns(2)
            left.markdown("**Base model**")
            left.code(base_out.sql or "(empty)", language="sql")
            right.markdown("**LoRA fine-tuned**")
            right.code(tuned_out.sql or "(empty)", language="sql")

            st.caption(
                f"{base_out.generation_seconds + tuned_out.generation_seconds:.1f}s for "
                "both. One set of base weights is loaded; the adapter is switched off and "
                "on around it."
            )

            for label, outcome in (("Base", base_out), ("Fine-tuned", tuned_out)):
                result = service.execute(db_id, outcome.sql)
                if result.ok:
                    st.success(f"{label}: executed, {len(result.rows)} rows")
                    if result.rows:
                        st.dataframe(pd.DataFrame(result.rows).head(20))
                else:
                    st.error(f"{label}: {result.error}")
