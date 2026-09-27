"""
agent.py — LangGraph StateGraph Agent

ARCHITECTURE (ReAct pattern simplified):
  
  User Query
      ↓
  [classify_intent] — LLM reads query, decides: "rag" or "calculator"
      ↓
  [execute_tool]    — Calls the chosen tool (rag_tool or calculator_tool)
      ↓
  [generate_answer] — LLM reads tool output + prompt, writes final answer
      ↓
  Final Answer

LANGGRAPH CONCEPTS USED HERE:
  - StateGraph    : the graph itself, holds all nodes and edges
  - TypedDict     : defines the state schema (what data flows between nodes)
  - Annotated     : with operator.add, means "append to list" not "replace"
  - Nodes         : functions that receive state, return partial state update
  - Edges         : connections between nodes (can be conditional)
  - END           : special terminal node, graph stops here
"""

import operator
from typing import TypedDict, List, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from app.tools import TOOL_REGISTRY, get_tool_descriptions_for_prompt
from app.prompts import get_prompt


# ─────────────────────────────────────────────
# LLM (local Ollama llama3.2)
# ─────────────────────────────────────────────
llm = ChatOllama(model="llama3.2", temperature=0.1)
# temperature=0.1 → near-deterministic, good for factual Q&A
# For creative tasks you'd set 0.7-0.9


# ═══════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    """
    The state object that flows through every node in the graph.
    
    Think of it as a shared whiteboard — each node reads from it
    and writes back to it. LangGraph merges the partial updates.
    
    Annotated[List, operator.add] means:
      If current messages = ["hello"] and a node returns ["world"],
      the new state = ["hello", "world"]  ← APPENDED, not replaced.
      
      Without Annotated: new state = ["world"]  ← REPLACED (wrong)
      
      This is called a "reducer" — it controls how state fields merge.
    """
    query:          str                              # original user question
    intent:         Optional[str]                    # "rag" or "calculator"
    tool_input:     Optional[str]                    # what to pass to the tool
    tool_output:    Optional[str]                    # what the tool returned
    final_answer:   Optional[str]                    # LLM's final response
    prompt_version: Optional[str]                    # which prompt was used
    error:          Optional[str]                    # error message if any
    messages:       Annotated[List, operator.add]    # full conversation history


# ═══════════════════════════════════════════════════════════════
# NODE 1 — CLASSIFY INTENT
# ═══════════════════════════════════════════════════════════════

def classify_intent(state: RAGState) -> dict:
    """
    Node 1: Ask the LLM to classify the query intent.
    
    The LLM reads the query and decides:
      - "rag"        → needs document search (banking info)
      - "calculator" → needs math computation
    
    WHY CLASSIFY FIRST?
      Without classification, you'd always run RAG, even for "what is 2+2".
      That wastes time and retrieves irrelevant context.
      Routing to the right tool = faster, more accurate answers.
    
    RETURNS: partial state update (only the keys we changed)
    """
    query = state["query"]

    classification_prompt = f"""You are a query classifier for a banking assistant.

Classify the following query into EXACTLY ONE category:
- "rag" : if the query asks about banking products, policies, rates, KYC, loans, payments, or any factual banking information
- "calculator" : if the query requires mathematical computation such as EMI, interest, percentages, or arithmetic

Query: {query}

Respond with ONLY the category name and the exact input to pass to the tool, in this format:
INTENT: <rag or calculator>
TOOL_INPUT: <the query to search OR the math expression to compute>

Examples:
Query: "What is the interest rate for personal loans?"
INTENT: rag
TOOL_INPUT: personal loan interest rate

Query: "Calculate EMI for 5 lakh at 12% for 36 months"
INTENT: calculator
TOOL_INPUT: 500000 * (0.01 * (1.01**36)) / ((1.01**36) - 1)

Query: "What are KYC document requirements?"
INTENT: rag
TOOL_INPUT: KYC documents required"""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    response_text = response.content.strip()

    # Parse the structured response
    intent = "rag"          # default fallback
    tool_input = query      # default: use original query

    for line in response_text.split("\n"):
        if line.startswith("INTENT:"):
            intent = line.replace("INTENT:", "").strip().lower()
            if intent not in ("rag", "calculator"):
                intent = "rag"
        elif line.startswith("TOOL_INPUT:"):
            tool_input = line.replace("TOOL_INPUT:", "").strip()

    print(f"  [classify_intent] intent={intent}, tool_input={tool_input[:60]}...")

    return {
        "intent": intent,
        "tool_input": tool_input,
        "messages": [f"INTENT CLASSIFIED: {intent}"]
    }


# ═══════════════════════════════════════════════════════════════
# NODE 2 — EXECUTE TOOL
# ═══════════════════════════════════════════════════════════════

def execute_tool(state: RAGState) -> dict:
    """
    Node 2: Call the appropriate tool based on classified intent.
    
    Looks up the tool function from TOOL_REGISTRY by name.
    Calls it with tool_input from the previous node.
    Stores the result in tool_output for the next node.
    
    WHY USE A REGISTRY LOOKUP?
      Decoupled — adding a new tool = add one entry to TOOL_REGISTRY.
      No if/elif chains here. Scales cleanly.
    """
    intent     = state.get("intent", "rag")
    tool_input = state.get("tool_input", state["query"])

    # Map intent → tool name
    tool_name_map = {
        "rag":        "rag_tool",
        "calculator": "calculator_tool",
    }

    tool_name = tool_name_map.get(intent, "rag_tool")
    tool_fn   = TOOL_REGISTRY[tool_name]

    print(f"  [execute_tool] calling {tool_name}...")

    try:
        result = tool_fn(tool_input)
        print(f"  [execute_tool] got result ({len(result)} chars)")
        return {
            "tool_output": result,
            "messages": [f"TOOL CALLED: {tool_name}"]
        }
    except Exception as e:
        error_msg = f"Tool {tool_name} failed: {str(e)}"
        print(f"  [execute_tool] ERROR: {error_msg}")
        return {
            "tool_output": "Tool execution failed.",
            "error": error_msg,
            "messages": [f"TOOL ERROR: {error_msg}"]
        }


# ═══════════════════════════════════════════════════════════════
# NODE 3 — GENERATE ANSWER
# ═══════════════════════════════════════════════════════════════

def generate_answer(state: RAGState) -> dict:
    """
    Node 3: Use the LLM to generate the final answer.
    
    Reads the active prompt from prompts.yaml (version-controlled).
    Passes the tool_output as context to the LLM.
    LLM generates a final answer following the prompt instructions.
    
    THIS IS WHERE PROMPT VERSIONING MATTERS:
      If v2 prompt produces poor answers, switch prompts.yaml → active_version: v1
      This node automatically uses the new version on next call.
      No code change needed.
    """
    prompt_config = get_prompt()   # loads active version from prompts.yaml
    query         = state["query"]
    context       = state.get("tool_output", "No context available.")

    # Build prompt using the active version's template
    system_msg = prompt_config["system"]
    human_msg  = prompt_config["human"].format(
        context=context,
        question=query
    )

    print(f"  [generate_answer] using prompt version: {prompt_config['version']}")

    try:
        response = llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=human_msg)
        ])

        answer = response.content.strip()

        return {
            "final_answer":   answer,
            "prompt_version": prompt_config["version"],
            "messages":       [f"ANSWER GENERATED (prompt {prompt_config['version']})"]
        }

    except Exception as e:
        error_msg = f"LLM generation failed: {str(e)}"
        return {
            "final_answer": "Sorry, I could not generate an answer.",
            "error":        error_msg,
            "messages":     [f"GENERATION ERROR: {error_msg}"]
        }


# ═══════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTION
# ═══════════════════════════════════════════════════════════════

def route_after_classify(state: RAGState) -> str:
    """
    Conditional edge: decides which node to go to after classify_intent.
    
    In our simple graph, we always go to execute_tool.
    But this pattern is powerful — you could route to:
      - "execute_tool"   : normal flow
      - "clarify"        : ask user to rephrase
      - "reject"         : refuse inappropriate queries
      - END              : answer directly without a tool
    
    Returns: the name of the NEXT NODE to visit
    """
    # Could add: if state["intent"] == "greeting": return END
    return "execute_tool"


# ═══════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ═══════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Assemble the StateGraph:
    
    classify_intent ──(conditional)──► execute_tool ──► generate_answer ──► END
    
    LANGGRAPH CONCEPTS:
      add_node(name, function)  : register a node
      add_edge(from, to)        : unconditional connection
      add_conditional_edges(    : LLM/logic decides next node
          from_node,
          routing_function,     : returns next node name
          {name: name}          : maps return value → node name
      )
      set_entry_point(name)     : which node runs first
    """
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("execute_tool",    execute_tool)
    graph.add_node("generate_answer", generate_answer)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional edge after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "execute_tool": "execute_tool",   # route name → node name
        }
    )

    # Unconditional edges
    graph.add_edge("execute_tool",    "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════
# COMPILED GRAPH (singleton — built once, reused for all requests)
# ═══════════════════════════════════════════════════════════════

agent_graph = build_graph()


# ═══════════════════════════════════════════════════════════════
# PUBLIC FUNCTION — called by main.py
# ═══════════════════════════════════════════════════════════════

def run_agent(query: str) -> dict:
    """
    Entry point for the agent. Called by POST /query in main.py.
    
    Initializes state, runs the graph, returns the result.
    
    Args:
        query: The user's question
    
    Returns:
        Dict with answer, intent, prompt_version, messages, error
    """
    # Initial state — only query is known, everything else starts as None/empty
    initial_state: RAGState = {
        "query":          query,
        "intent":         None,
        "tool_input":     None,
        "tool_output":    None,
        "final_answer":   None,
        "prompt_version": None,
        "error":          None,
        "messages":       []        # operator.add reducer appends to this
    }

    print(f"\n>>> AGENT: processing query: '{query}'")

    # Run the graph — LangGraph executes nodes in order
    final_state = agent_graph.invoke(initial_state)

    print(f">>> AGENT: done. Answer length: {len(final_state.get('final_answer', ''))}")

    return {
        "answer":         final_state.get("final_answer", "No answer generated."),
        "intent":         final_state.get("intent"),
        "prompt_version": final_state.get("prompt_version"),
        "messages":       final_state.get("messages", []),
        "error":          final_state.get("error")
    }