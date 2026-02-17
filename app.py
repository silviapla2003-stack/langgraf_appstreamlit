import os
import streamlit as st
from typing import Annotated, TypedDict, List, Any

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults


# -----------------------------
# Reducer (sin depender de imports internos de LangGraph)
# -----------------------------
def add_messages(existing: List[Any] | None, new: List[Any] | None) -> List[Any]:
    return (existing or []) + (new or [])


# -----------------------------
# Helpers
# -----------------------------
def msg_to_text(msg) -> str:
    """Normaliza la salida de Gemini (a veces devuelve lista de parts)."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            else:
                texts.append(str(part))
        return "\n".join(texts).strip()
    return str(content)


def get_secret_or_env(key: str, default: str = "") -> str:
    """Lee secrets si existen, si no, env. Nunca crashea si no hay secrets."""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="News Writer Agent (LangGraph)", page_icon="📰", layout="wide")
st.title("📰 News Writer Agent — LangGraph")
st.caption("Genera un artículo usando Gemini + búsqueda web (Tavily) dentro de un grafo LangGraph.")

with st.sidebar:
    st.header("🔐 Claves API (obligatorias)")
    google_key_input = st.text_input("Google API Key", type="password")
    tavily_key_input = st.text_input("Tavily API Key", type="password")
    st.divider()
    show_steps = st.toggle("Mostrar pasos intermedios", value=False)

# Prioridad: input usuario > secrets/env
google_key = google_key_input or get_secret_or_env("GOOGLE_API_KEY", "")
tavily_key = tavily_key_input or get_secret_or_env("TAVILY_API_KEY", "")


# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]


# -----------------------------
# Prompts
# -----------------------------
outliner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un editor senior. Con el tema del usuario y los resultados de búsqueda, crea un OUTLINE detallado "
            "con secciones y bullets. NO preguntes nada al usuario. NO metas disclaimers."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un periodista tech. Escribe un artículo final en ESPAÑOL con entre 750 y 900 palabras.\n"
            "Formato EXACTO:\n"
            "TITLE: <título>\n"
            "BODY: <cuerpo>\n\n"
            "Requisitos:\n"
            "- No preguntes al usuario.\n"
            "- No digas 'no puedo' ni disclaimers.\n"
            "- Integra datos/ideas de los resultados de búsqueda.\n"
            "- Tono divulgativo para negocio.\n"
            "- Si hay incertidumbre, formula como 'según análisis y reportes recientes' en vez de negarte."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# -----------------------------
# Nodes
# -----------------------------
def tavily_node(state: AgentState):
    user_text = msg_to_text(state["messages"][0]).strip()

    search_tool = TavilySearchResults(max_results=6)
    results = search_tool.invoke({"query": user_text})

    return {
        "messages": [
            HumanMessage(content=f"TEMA_USUARIO:\n{user_text}\n\nRESULTADOS_TAVILY:\n{results}")
        ]
    }


def outliner_node(state: AgentState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    chain = outliner_prompt | llm
    res = chain.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=msg_to_text(res))]}


def writer_node(state: AgentState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    chain = writer_prompt | llm
    res = chain.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=msg_to_text(res))]}


def build_graph(google_api_key: str, tavily_api_key: str):
    if not google_api_key or not tavily_api_key:
        raise ValueError("Debes introducir Google API Key y Tavily API Key en la barra lateral.")

    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    g = StateGraph(AgentState)
    g.add_node("tavily", tavily_node)
    g.add_node("outliner", outliner_node)
    g.add_node("writer", writer_node)

    g.set_entry_point("tavily")
    g.add_edge("tavily", "outliner")
    g.add_edge("outliner", "writer")
    g.add_edge("writer", END)

    return g.compile()


# -----------------------------
# App
# -----------------------------
prompt = st.text_area(
    "¿Qué artículo quieres generar?",
    value="",          # <-- VACÍO como pediste
    height=150,
    placeholder="Ej: 'Tendencias de IA en retail para 2026: riesgos, oportunidades, casos de uso, 800 palabras, tono business.'"
)

run = st.button("🚀 Generar artículo", type="primary")

# Validaciones sin crashear
if run:
    if not google_key or not tavily_key:
        st.error("Faltan las API keys. Mételas en la barra lateral (Google y Tavily).")
        st.stop()

    if not prompt.strip():
        st.error("Escribe el tema/instrucciones del artículo antes de generar.")
        st.stop()

    try:
        graph = build_graph(google_key, tavily_key)

        final_text = ""
        with st.spinner("Generando…"):
            for event in graph.stream({"messages": [HumanMessage(content=prompt)]}, stream_mode="values"):
                msg = event["messages"][-1]
                text = msg_to_text(msg)

                if show_steps:
                    st.write(text)

                final_text = text

        st.success("Listo ✅")
        st.subheader("📝 Resultado")
        st.markdown(final_text)

        st.download_button(
            "⬇️ Descargar como .txt",
            data=final_text.encode("utf-8"),
            file_name="articulo_generado.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error("Se ha producido un error.")
        st.exception(e)
