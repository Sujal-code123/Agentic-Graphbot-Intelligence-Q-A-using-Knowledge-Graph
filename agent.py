import os
import pandas as pd
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import gradio as gr
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
import matplotlib

matplotlib.use("Agg")
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", None)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
#MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-r1-0528:free")
#MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:free")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


from langchain_openai import ChatOpenAI

def make_llm() -> BaseChatModel:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        base_url=OPENAI_API_BASE,
        temperature=0,
        max_retries=2,
    )

llm = make_llm()
cypher_prompt = ChatPromptTemplate.from_template("""
You are a Cypher expert. Use ONLY the schema below to write valid Cypher queries. Output ONLY the Cypher query in triple backticks. Answer based on comparison.

{schema}

User question: {question}
""")

explanation_prompt = ChatPromptTemplate.from_template("""
You are a business analyst.
Given the user question and the Neo4j query result, produce a clean, structured answer in **Markdown**.
Do NOT include the Cypher query or DB syntax.
Give a short descriptive chart title if a chart is relevant.

Your answer must have two parts:

1. Summary – bullet points
2. Table – a Markdown table with the returned rows

Question: {question}
Neo4j Result: {cypher_result}

Output format:
Summary:
- Point 1
- Point 2

Table:
| Column1 | Column2 |
|---------|---------|
| ...     | ...     |
""")

generate_cypher_chain = cypher_prompt | llm | StrOutputParser()
explanation_chain = explanation_prompt | llm | StrOutputParser()

def get_dynamic_graph_schema():
    session_args = {}
    if NEO4J_DATABASE:
        session_args["database"] = NEO4J_DATABASE
    with driver.session(**session_args) as session:
        try:
            node_results = session.run(
                "CALL db.schema.nodeTypeProperties() "
                "YIELD nodeType, propertyName "
                "RETURN nodeType, collect(DISTINCT propertyName) AS properties"
            ).data()
        except Exception:
            node_results = session.run(
                "CALL apoc.meta.nodeTypeProperties() "
                "YIELD node, property "
                "RETURN node AS nodeType, collect(DISTINCT property) AS properties"
            ).data()
        rel_results = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN DISTINCT type(r) AS relType, labels(a)[0] AS sourceNodeType, labels(b)[0] AS targetNodeType"
        ).data()
    node_schema = {row["nodeType"]: row["properties"] for row in node_results}
    relationship_schema = [
        {"relType": rel["relType"], "sourceNodeType": rel["sourceNodeType"], "targetNodeType": rel["targetNodeType"]}
        for rel in rel_results
    ]
    return {"nodes": node_schema, "relationships": relationship_schema}

def format_graph_schema(graph_json):
    lines = ["Graph Schema:", "Node Labels and Their Properties:"]
    for label, props in graph_json["nodes"].items():
        prop_list = ", ".join(sorted({str(p) for p in (props or [])}))
        lines.append(f"- {label} ({prop_list})")
    lines.append("\nValid Relationships:")
    for rel in graph_json["relationships"]:
        lines.append(f"- ({rel['sourceNodeType']})-[:{rel['relType']}]->({rel['targetNodeType']})")
    lines.append("Only use these relationships and properties. Do NOT make up new ones.")
    return "\n".join(lines)

def run_cypher(query: str):
    session_args = {}
    if NEO4J_DATABASE:
        session_args["database"] = NEO4J_DATABASE
    with driver.session(**session_args) as session:
        result = session.run(query)
        rows = []
        from datetime import date
        for record in result:
            row = record.data()
            for k, v in list(row.items()):
                try:
                    if hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day"):
                        row[k] = date(v.year, v.month, v.day)
                    elif hasattr(v, "days") or hasattr(v, "months") or hasattr(v, "seconds"):
                        total_days = getattr(v, "days", 0) + getattr(v, "months", 0) * 30 + getattr(v, "seconds", 0) / 86400
                        row[k] = round(total_days, 2)
                    else:
                        row[k] = v
                except Exception:
                    row[k] = v
            rows.append(row)
        return rows

def plot_data_as_chart(data, chart_choice="auto"):
    if not data:
        return None
    df = pd.DataFrame(data)
    if df.empty:
        return None

    df.columns = [str(col).split('.')[-1] for col in df.columns]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    if df.shape[0] == 1 and df.shape[1] > 1:
        row = df.iloc[0]
        df = pd.DataFrame({"Category": row.index.astype(str), "Value": row.values})

    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
    else:
        return None

    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    forced = (chart_choice or "auto").lower()

    if df[y_col].notna().sum() == 0:
        counts = df[x_col].astype(str).value_counts().reset_index()
        counts.columns = [x_col, "Count"]
        df = counts
        y_col = "Count"

    df = df.dropna(subset=[y_col])
    if df.empty:
        return None

    if forced in ("bar", "line", "pie"):
        chart_type = forced
    else:
        if pd.api.types.is_datetime64_any_dtype(df[x_col]) or pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df[x_col], errors="coerce")):
            chart_type = "line"
        elif pd.api.types.is_numeric_dtype(df[y_col]) and (pd.api.types.is_object_dtype(df[x_col]) or df[x_col].nunique() < 20):
            chart_type = "bar"
        else:
            chart_type = "pie" if df[x_col].nunique() <= 6 else "bar"

    def try_bar(dfx, xc, yc):
        figb, axb = plt.subplots(figsize=(8, 5))
        dfx.plot(kind="bar", x=xc, y=yc, ax=axb, legend=False)
        axb.set_xlabel(xc); axb.set_ylabel(yc); axb.set_title(f"{yc} by {xc}")
        axb.tick_params(axis="x", rotation=45); plt.tight_layout()
        return figb

    def try_line(dfx, xc, yc):
        figl, axl = plt.subplots(figsize=(8, 5))
        try:
            maybe_dt = pd.to_datetime(dfx[xc], errors="coerce")
            if maybe_dt.notna().sum() > 0:
                dfx = dfx.copy(); dfx[xc] = maybe_dt; dfx = dfx.sort_values(xc)
        except Exception:
            pass
        dfx.plot(kind="line", x=xc, y=yc, ax=axl, marker="o", legend=False)
        axl.set_xlabel(xc); axl.set_ylabel(yc); axl.set_title(f"{yc} over {xc}")
        plt.tight_layout()
        return figl

    def try_pie(dfx, xc, yc):
        figp, axp = plt.subplots(figsize=(6, 6))
        series = dfx.set_index(xc)[yc]
        series.plot(kind="pie", autopct="%1.1f%%", ax=axp)
        axp.set_ylabel(""); axp.set_title(f"{yc} distribution")
        plt.tight_layout()
        return figp

    fig = None
    try:
        if chart_type == "bar":
            fig = try_bar(df, x_col, y_col)
        elif chart_type == "line":
            if not pd.api.types.is_numeric_dtype(df[y_col]):
                counts = df[x_col].astype(str).value_counts().reset_index()
                counts.columns = [x_col, "Count"]
                fig = try_line(counts, x_col, "Count")
            else:
                fig = try_line(df, x_col, y_col)
        elif chart_type == "pie":
            if df[x_col].nunique() > 40:
                topn = df.set_index(x_col)[y_col].nlargest(10).reset_index()
                fig = try_bar(topn, x_col, y_col)
            else:
                fig = try_pie(df, x_col, y_col)
    except Exception:
        fig = None

    if fig is None:
        for fallback in ("bar", "line", "pie"):
            try:
                if fallback == "bar":
                    fig = try_bar(df, x_col, y_col)
                elif fallback == "line" and pd.api.types.is_numeric_dtype(df[y_col]):
                    fig = try_line(df, x_col, y_col)
                elif fallback == "pie" and df[x_col].nunique() <= 100:
                    fig = try_pie(df, x_col, y_col)
            except Exception:
                fig = None
            if fig is not None:
                break

    return fig

def langchain_pipeline(user_question, chart_choice):
    GRAPH_SCHEMA = get_dynamic_graph_schema()
    schema_text = format_graph_schema(GRAPH_SCHEMA)

    raw_query = generate_cypher_chain.invoke({
    "schema": schema_text,
    "question": user_question
}).strip()
    if raw_query.startswith("```"):
        lines = [l for l in raw_query.splitlines() if not l.strip().startswith("```")]
        cypher_query = "\n".join(lines).strip()
    else:
        cypher_query = raw_query

    result = run_cypher(cypher_query)
    answer = explanation_chain.invoke({
    "question": user_question,
    "cypher_result": result
})
    if not isinstance(answer, str):
        answer = str(answer)

    fig = None
    if chart_choice and chart_choice.lower() != "none":
        fig = plot_data_as_chart(result, chart_choice=chart_choice.lower())

    return answer, fig

def handle_submit(question, chart_pref):
    answer, fig = langchain_pipeline(question, chart_pref)
    answer = str(answer)
    if fig is not None and not hasattr(fig, "savefig"):
        fig = None
    return answer, fig

with gr.Blocks() as demo:
    gr.Markdown("#Intelligent Q&A using knowledge graph")
    question = gr.Textbox(label="Ask a question about the graph", lines=2)
    chart_pref = gr.Radio(["Auto", "Bar", "Line", "Pie", "None"], label="Chart preference", value="Auto")
    submit = gr.Button("Submit")
    output_text = gr.Markdown(label="Answer")
    output_plot = gr.Plot(label="Chart", visible=True)
    submit.click(fn=handle_submit, inputs=[question, chart_pref], outputs=[output_text, output_plot])

if __name__ == "__main__":
    demo.launch()











'''import os
import pandas as pd
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import gradio as gr
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
import matplotlib

matplotlib.use("Agg")

# Load env
load_dotenv()

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", None)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-r1-0528:free")

from langchain_openai import ChatOpenAI

def make_llm() -> BaseChatModel:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        base_url=OPENAI_API_BASE,
        temperature=0,
        max_retries=2,
    )

llm = make_llm()

cypher_prompt = ChatPromptTemplate.from_template("""
You are a Cypher expert. Use ONLY the schema below to write valid Cypher queries. Output ONLY the Cypher query in triple backticks. Answer based on comparison.

{schema}

User question: {question}
""")

explanation_prompt = ChatPromptTemplate.from_template("""
You are a business analyst.
Given the user question and the Neo4j query result, produce a clean, structured answer in **Markdown**.
Do NOT include the Cypher query or DB syntax.
Give a short descriptive chart title if a chart is relevant.

Your answer must have two parts:

1. Summary – bullet points
2. Table – a Markdown table with the returned rows

Question: {question}
Neo4j Result: {cypher_result}

Output format:
Summary:
- Point 1
- Point 2

Table:
| Column1 | Column2 |
|---------|---------|
| ...     | ...     |
""")

generate_cypher_chain = LLMChain(llm=llm, prompt=cypher_prompt, output_parser=StrOutputParser())
explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt, output_parser=StrOutputParser())

def get_dynamic_graph_schema():
    session_args = {}
    if NEO4J_DATABASE:
        session_args["database"] = NEO4J_DATABASE
    with driver.session(**session_args) as session:
        try:
            node_results = session.run(
                "CALL db.schema.nodeTypeProperties() "
                "YIELD nodeType, propertyName "
                "RETURN nodeType, collect(DISTINCT propertyName) AS properties"
            ).data()
        except Exception:
            node_results = session.run(
                "CALL apoc.meta.nodeTypeProperties() "
                "YIELD node, property "
                "RETURN node AS nodeType, collect(DISTINCT property) AS properties"
            ).data()
        rel_results = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN DISTINCT type(r) AS relType, labels(a)[0] AS sourceNodeType, labels(b)[0] AS targetNodeType"
        ).data()
    node_schema = {row["nodeType"]: row["properties"] for row in node_results}
    relationship_schema = [
        {"relType": rel["relType"], "sourceNodeType": rel["sourceNodeType"], "targetNodeType": rel["targetNodeType"]}
        for rel in rel_results
    ]
    return {"nodes": node_schema, "relationships": relationship_schema}

def format_graph_schema(graph_json):
    lines = ["Graph Schema:", "Node Labels and Their Properties:"]
    for label, props in graph_json["nodes"].items():
        prop_list = ", ".join(sorted({str(p) for p in (props or [])}))
        lines.append(f"- {label} ({prop_list})")
    lines.append("\nValid Relationships:")
    for rel in graph_json["relationships"]:
        lines.append(f"- ({rel['sourceNodeType']})-[:{rel['relType']}]->({rel['targetNodeType']})")
    lines.append("Only use these relationships and properties. Do NOT make up new ones.")
    return "\n".join(lines)

# Run Cypher
def run_cypher(query: str):
    session_args = {}
    if NEO4J_DATABASE:
        session_args["database"] = NEO4J_DATABASE
    with driver.session(**session_args) as session:
        result = session.run(query)
        rows = []
        from datetime import date
        for record in result:
            row = record.data()
            for k, v in list(row.items()):
                try:
                    if hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day"):
                        row[k] = date(v.year, v.month, v.day)
                    elif hasattr(v, "days") or hasattr(v, "months") or hasattr(v, "seconds"):
                        total_days = getattr(v, "days", 0) + getattr(v, "months", 0) * 30 + getattr(v, "seconds", 0) / 86400
                        row[k] = round(total_days, 2)
                    else:
                        row[k] = v
                except Exception:
                    row[k] = v
            rows.append(row)
        return rows

# Plotting
def plot_data_as_chart(data, chart_choice="auto"):
    if not data:
        return None
    df = pd.DataFrame(data)
    if df.empty:
        return None

    df.columns = [str(col).split('.')[-1] for col in df.columns]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    if df.shape[0] == 1 and df.shape[1] > 1:
        row = df.iloc[0]
        df = pd.DataFrame({"Category": row.index.astype(str), "Value": row.values})

    if df.shape[1] >= 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
    else:
        return None

    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    forced = (chart_choice or "auto").lower()

    if df[y_col].notna().sum() == 0:
        counts = df[x_col].astype(str).value_counts().reset_index()
        counts.columns = [x_col, "Count"]
        df = counts
        y_col = "Count"

    df = df.dropna(subset=[y_col])
    if df.empty:
        return None

    if forced in ("bar", "line", "pie"):
        chart_type = forced
    else:
        if pd.api.types.is_datetime64_any_dtype(df[x_col]) or pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df[x_col], errors="coerce")):
            chart_type = "line"
        elif pd.api.types.is_numeric_dtype(df[y_col]) and (pd.api.types.is_object_dtype(df[x_col]) or df[x_col].nunique() < 20):
            chart_type = "bar"
        else:
            chart_type = "pie" if df[x_col].nunique() <= 6 else "bar"

    def try_bar(dfx, xc, yc):
        figb, axb = plt.subplots(figsize=(8, 5))
        dfx.plot(kind="bar", x=xc, y=yc, ax=axb, legend=False)
        axb.set_xlabel(xc); axb.set_ylabel(yc); axb.set_title(f"{yc} by {xc}")
        axb.tick_params(axis="x", rotation=45); plt.tight_layout()
        return figb

    def try_line(dfx, xc, yc):
        figl, axl = plt.subplots(figsize=(8, 5))
        try:
            maybe_dt = pd.to_datetime(dfx[xc], errors="coerce")
            if maybe_dt.notna().sum() > 0:
                dfx = dfx.copy(); dfx[xc] = maybe_dt; dfx = dfx.sort_values(xc)
        except Exception:
            pass
        dfx.plot(kind="line", x=xc, y=yc, ax=axl, marker="o", legend=False)
        axl.set_xlabel(xc); axl.set_ylabel(yc); axl.set_title(f"{yc} over {xc}")
        plt.tight_layout()
        return figl

    def try_pie(dfx, xc, yc):
        figp, axp = plt.subplots(figsize=(6, 6))
        series = dfx.set_index(xc)[yc]
        series.plot(kind="pie", autopct="%1.1f%%", ax=axp)
        axp.set_ylabel(""); axp.set_title(f"{yc} distribution")
        plt.tight_layout()
        return figp

    fig = None
    try:
        if chart_type == "bar":
            fig = try_bar(df, x_col, y_col)
        elif chart_type == "line":
            if not pd.api.types.is_numeric_dtype(df[y_col]):
                counts = df[x_col].astype(str).value_counts().reset_index()
                counts.columns = [x_col, "Count"]
                fig = try_line(counts, x_col, "Count")
            else:
                fig = try_line(df, x_col, y_col)
        elif chart_type == "pie":
            if df[x_col].nunique() > 40:
                topn = df.set_index(x_col)[y_col].nlargest(10).reset_index()
                fig = try_bar(topn, x_col, y_col)
            else:
                fig = try_pie(df, x_col, y_col)
    except Exception:
        fig = None

    if fig is None:
        for fallback in ("bar", "line", "pie"):
            try:
                if fallback == "bar":
                    fig = try_bar(df, x_col, y_col)
                elif fallback == "line" and pd.api.types.is_numeric_dtype(df[y_col]):
                    fig = try_line(df, x_col, y_col)
                elif fallback == "pie" and df[x_col].nunique() <= 100:
                    fig = try_pie(df, x_col, y_col)
            except Exception:
                fig = None
            if fig is not None:
                break

    return fig

def langchain_pipeline(user_question, chart_choice):
    GRAPH_SCHEMA = get_dynamic_graph_schema()
    schema_text = format_graph_schema(GRAPH_SCHEMA)

    raw_query = generate_cypher_chain.run(schema=schema_text, question=user_question).strip()
    if raw_query.startswith("```"):
        lines = [l for l in raw_query.splitlines() if not l.strip().startswith("```")]
        cypher_query = "\n".join(lines).strip()
    else:
        cypher_query = raw_query

    result = run_cypher(cypher_query)
    answer = explanation_chain.run(question=user_question, cypher_result=result)
    if not isinstance(answer, str):
        answer = str(answer)

    fig = None
    if chart_choice and chart_choice.lower() != "none":
        fig = plot_data_as_chart(result, chart_choice=chart_choice.lower())

    return answer, fig

# Gradio
def handle_submit(question, chart_pref):
    answer, fig = langchain_pipeline(question, chart_pref)
    answer = str(answer)
    if fig is not None and not hasattr(fig, "savefig"):
        fig = None
    return answer, fig

with gr.Blocks() as demo:
    gr.Markdown("#Markk")
    question = gr.Textbox(label="Ask a question about the graph", lines=2)
    chart_pref = gr.Radio(["Auto", "Bar", "Line", "Pie", "None"], label="Chart preference", value="Auto")
    submit = gr.Button("Submit")
    output_text = gr.Markdown(label="Answer")
    output_plot = gr.Plot(label="Chart", visible=True)
    submit.click(fn=handle_submit, inputs=[question, chart_pref], outputs=[output_text, output_plot])

if __name__ == "__main__":
    demo.launch()'''
