import os
import pandas as pd
import matplotlib.pyplot as plt

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "healthcare_dataset.csv"
)

PDF_PATH = os.path.join(
    BASE_DIR,
    "hospital_policy_rag_practice.pdf"
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-5.6",
    use_responses_api=True
)
@tool
def analyze_data(
    column: str,
    operation: str,
    return_column: str | None = None
) -> dict:
    """
    Analyze a column in the healthcare dataset.

    Supported operations:
    mean, median, min, max, sum, count.

    For min or max, return_column can return another
    column from the same row.
    """

    if column not in df.columns:
        return {
            "error": f"Column '{column}' does not exist."
        }

    if (
        return_column is not None
        and return_column not in df.columns
    ):
        return {
            "error":
            f"Column '{return_column}' does not exist."
        }

    if operation == "mean":

        return {
            "operation": "mean",
            "column": column,
            "value": float(df[column].mean())
        }

    elif operation == "median":

        return {
            "operation": "median",
            "column": column,
            "value": float(df[column].median())
        }

    elif operation == "sum":

        return {
            "operation": "sum",
            "column": column,
            "value": float(df[column].sum())
        }

    elif operation == "count":

        return {
            "operation": "count",
            "column": column,
            "value": int(df[column].count())
        }

    elif operation == "max":

        index = df[column].idxmax()

        result = {
            "operation": "max",
            "column": column,
            "value": float(
                df.loc[index, column]
            )
        }

        if return_column:
            result[return_column] = str(
                df.loc[index, return_column]
            )

        return result

    elif operation == "min":

        index = df[column].idxmin()

        result = {
            "operation": "min",
            "column": column,
            "value": float(
                df.loc[index, column]
            )
        }

        if return_column:
            result[return_column] = str(
                df.loc[index, return_column]
            )

        return result

    return {
        "error":
        f"Operation '{operation}' is not supported."
    }
@tool
def group_analysis(
    group_column: str,
    value_column: str,
    operation: str
) -> dict:
    """
    Group the healthcare dataset by a categorical
    column and calculate mean, sum, count, min, or max.

    Examples:
    - average billing by medical condition
    - total revenue by medical condition
    - patient count by admission type
    - average age by gender
    """

    if group_column not in df.columns:
        return {
            "error":
            f"Column '{group_column}' does not exist."
        }

    if value_column not in df.columns:
        return {
            "error":
            f"Column '{value_column}' does not exist."
        }

    grouped = df.groupby(
        group_column
    )[value_column]

    if operation == "mean":
        result = grouped.mean()

    elif operation == "sum":
        result = grouped.sum()

    elif operation == "count":
        result = grouped.count()

    elif operation == "min":
        result = grouped.min()

    elif operation == "max":
        result = grouped.max()

    else:
        return {
            "error":
            f"Operation '{operation}' is not supported."
        }

    result = result.sort_values(
        ascending=False
    )

    return {
        "operation": operation,
        "group_column": group_column,
        "value_column": value_column,

        "highest_category":
            str(result.index[0]),

        "highest_value":
            float(result.iloc[0]),

        "data": {
            str(key): float(value)
            for key, value in result.items()
        }
    }
@tool
def create_chart(
    chart_type: str,
    x_column: str,
    y_column: str | None = None,
    aggregation: str = "mean",
    sort: bool = True
) -> dict:
    """
    Create a chart from the healthcare dataset.

    Supported chart types:
    bar, histogram, scatter.

    For bar charts:
    - average/mean -> aggregation='mean'
    - total/revenue -> aggregation='sum'
    - count/how many -> aggregation='count'

    Use sort=True for highest, ranking, or sorted results.
    """

    if x_column not in df.columns:
        return {
            "error": f"Column '{x_column}' does not exist."
        }

    if y_column is not None and y_column not in df.columns:
        return {
            "error": f"Column '{y_column}' does not exist."
        }

    plt.figure(figsize=(9, 5))

    # -------------------------
    # BAR CHART
    # -------------------------

    if chart_type == "bar":

        if y_column is None:
            plt.close()

            return {
                "error":
                "y_column is required for a bar chart."
            }

        grouped = df.groupby(
            x_column
        )[y_column]

        if aggregation == "mean":

            chart_data = grouped.mean()

            metric_name = (
                f"Average {y_column}"
            )

        elif aggregation == "sum":

            chart_data = grouped.sum()

            metric_name = (
                f"Total {y_column}"
            )

        elif aggregation == "count":

            chart_data = grouped.count()

            metric_name = "Count"

        elif aggregation == "min":

            chart_data = grouped.min()

            metric_name = (
                f"Minimum {y_column}"
            )

        elif aggregation == "max":

            chart_data = grouped.max()

            metric_name = (
                f"Maximum {y_column}"
            )

        else:

            plt.close()

            return {
                "error":
                f"Unsupported aggregation: {aggregation}"
            }

        if sort:

            chart_data = chart_data.sort_values(
                ascending=False
            )

        chart_data.plot(
            kind="bar"
        )

        plt.xlabel(x_column)

        plt.ylabel(metric_name)

        plt.title(
            f"{metric_name} by {x_column}"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        # Save instead of plt.show()

        chart_path = os.path.join(
            BASE_DIR,
            "latest_chart.png"
        )

        plt.savefig(
            chart_path,
            bbox_inches="tight"
        )

        plt.close()

        return {
            "status":
                "Chart created successfully",

            "chart_path":
                chart_path,

            "chart_type":
                "bar",

            "aggregation":
                aggregation,

            "metric":
                metric_name,

            "highest_category":
                str(chart_data.index[0]),

            "highest_value":
                float(chart_data.iloc[0]),

            "data": {
                str(key): float(value)
                for key, value
                in chart_data.items()
            }
        }

    # -------------------------
    # HISTOGRAM
    # -------------------------

    elif chart_type == "histogram":

        plt.hist(
            df[x_column].dropna(),
            bins=10
        )

        plt.xlabel(x_column)
        plt.ylabel("Frequency")

        plt.title(
            f"Distribution of {x_column}"
        )

        plt.tight_layout()

        chart_path = os.path.join(
            BASE_DIR,
            "latest_chart.png"
        )

        plt.savefig(
            chart_path,
            bbox_inches="tight"
        )

        plt.close()

        return {
            "status":
                "Histogram created successfully",

            "chart_path":
                chart_path
        }

    # -------------------------
    # SCATTER
    # -------------------------

    elif chart_type == "scatter":

        if y_column is None:

            plt.close()

            return {
                "error":
                    "y_column is required."
            }

        plt.scatter(
            df[x_column],
            df[y_column]
        )

        plt.xlabel(x_column)
        plt.ylabel(y_column)

        plt.title(
            f"{y_column} vs {x_column}"
        )

        plt.tight_layout()

        chart_path = os.path.join(
            BASE_DIR,
            "latest_chart.png"
        )

        plt.savefig(
            chart_path,
            bbox_inches="tight"
        )

        plt.close()

        return {
            "status":
                "Scatter plot created successfully",

            "chart_path":
                chart_path
        }

    plt.close()

    return {
        "error":
            f"Chart type '{chart_type}' is not supported."
    }
# --------------------------------------------------
# RAG - LOAD HOSPITAL POLICY
# --------------------------------------------------

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(
    documents
)
# --------------------------------------------------
# EMBEDDINGS + VECTOR STORE
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="hospital_policy"
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)
# --------------------------------------------------
# HOSPITAL POLICY RAG TOOL
# --------------------------------------------------

@tool
def search_hospital_policy(
    query: str
) -> dict:
    """
    Search the hospital policy document.

    Use this tool for questions about:
    hospital policies, billing policies,
    emergency admissions, discharge,
    readmissions, medication safety,
    patient satisfaction, procedures,
    guidelines, and data privacy.
    """

    retrieved_docs = retriever.invoke(
        query
    )

    results = []

    for doc in retrieved_docs:

        results.append({
            "content":
                doc.page_content,

            "page":
                doc.metadata.get(
                    "page",
                    "Unknown"
                )
        })

    return {
        "query": query,
        "results": results
    }
# --------------------------------------------------
# REGISTER TOOLS
# --------------------------------------------------

tools = [
    analyze_data,
    group_analysis,
    create_chart,
    search_hospital_policy
]

llm_with_tools = llm.bind_tools(tools)
# --------------------------------------------------
# AGENT NODE
# --------------------------------------------------

def agent_node(state: MessagesState):

    dataset_context = f"""
You are a healthcare data analyst and hospital policy assistant.

A healthcare dataset is already loaded.

Available dataset columns:
{df.columns.tolist()}

Number of rows:
{len(df)}

You have tools for:
- numerical analysis
- grouped analysis
- data visualization
- hospital policy retrieval

IMPORTANT ANALYSIS RULES:

1. Interpret analytical terms carefully:
   - "average" or "mean" = mean
   - "total" or "revenue" = sum
   - "count", "number of", or "how many" = count
   - "highest" = descending ranking

2. When the user asks for revenue, use SUM of Billing Amount.

3. When the user asks for average billing,
   use MEAN of Billing Amount.

4. When creating a chart and providing a conclusion,
   use the SAME aggregation for both.

5. Never describe a mean as total revenue.

6. Never describe a sum as average billing.

7. Never invent numerical results.
   Use tool results.

8. For questions about hospital policies,
   procedures, guidelines, or recommendations,
   use the hospital policy retrieval tool.

9. Never invent hospital policy information.
   Base policy answers only on retrieved document content.

10. Some questions may require multiple tools.
    Use all necessary tools before answering.

11. If a matplotlib chart has been created,
    do NOT create an ASCII or text-based chart.

12. Keep the final response concise and professional.

13. Do not mention internal tool names,
    function calls, LangChain, or LangGraph
    in the final answer.
"""

    messages = [
        SystemMessage(
            content=dataset_context
        )
    ] + state["messages"]

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }
# --------------------------------------------------
# BUILD LANGGRAPH
# --------------------------------------------------

builder = StateGraph(
    MessagesState
)

builder.add_node(
    "agent",
    agent_node
)

builder.add_node(
    "tools",
    ToolNode(tools)
)

builder.add_edge(
    START,
    "agent"
)

builder.add_conditional_edges(
    "agent",
    tools_condition
)

builder.add_edge(
    "tools",
    "agent"
)

graph = builder.compile()
# --------------------------------------------------
# USER-FACING FUNCTION
# --------------------------------------------------

def ask_agent(question: str) -> str:

    result = graph.invoke({
        "messages": [
            HumanMessage(
                content=question
            )
        ]
    })

    final_message = result[
        "messages"
    ][-1]

    content = final_message.content

    # Responses API may return content blocks
    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":
                    text_parts.append(
                        block.get("text", "")
                    )

        return "\n".join(
            text_parts
        ).strip()

    return str(content)