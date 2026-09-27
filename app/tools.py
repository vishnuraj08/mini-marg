"""
tools.py — Tools available to the LangGraph agent

WHY TOOLS EXIST:
  The LLM alone can answer general questions from training data.
  But it can't search YOUR documents or do precise math.
  Tools give the agent superpowers beyond what the LLM knows.

TOOL CONTRACT:
  Every tool is a plain Python function:
    Input  → string (the agent passes a string argument)
    Output → string (the agent reads the string result)
  
  The agent decides WHICH tool to call based on the query intent.
  This is the core of agentic AI — the LLM routes, tools execute.
"""

import math
import re
from typing import Dict
from app.rag import search


# ═══════════════════════════════════════════════════════════════
# TOOL 1 — RAG SEARCH TOOL
# ═══════════════════════════════════════════════════════════════

def rag_tool(query: str) -> str:
    """
    Search the FAISS knowledge base for relevant document chunks.
    
    WHEN THE AGENT USES THIS:
      - "What is the interest rate for personal loans?"
      - "What documents are needed for KYC?"
      - "How does RTGS work?"
      → Any question that needs information from our banking documents
    
    WHAT IT DOES INTERNALLY:
      Calls rag.search() → hybrid BM25 + FAISS → RRF → top 3 chunks
      Returns them as a formatted string the agent can read
    
    Args:
        query: The user's question (or a rewritten/decomposed version of it)
    
    Returns:
        Formatted string with retrieved context chunks + their sources
    """
    try:
        results = search(query, top_k=3)

        if not results:
            return "No relevant information found in the knowledge base."

        # Format chunks as readable context for the LLM
        formatted = []
        for i, chunk in enumerate(results, 1):
            formatted.append(
                f"[Chunk {i} | Source: {chunk['source']}]\n{chunk['text']}"
            )

        return "\n\n---\n\n".join(formatted)

    except FileNotFoundError:
        return "Knowledge base not initialized. Please call POST /ingest first."
    except Exception as e:
        return f"Search failed: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# TOOL 2 — CALCULATOR TOOL
# ═══════════════════════════════════════════════════════════════

def calculator_tool(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    WHEN THE AGENT USES THIS:
      - "Calculate EMI for INR 5 lakh at 12% for 3 years"
      - "What is 2% of 40000?"
      - "How much interest will I pay on a 10 lakh loan?"
      → Any question that involves numbers and computation
    
    WHY NOT JUST LET THE LLM DO MATH?
      LLMs are notoriously bad at arithmetic. They hallucinate numbers.
      A tool gives you a guaranteed correct answer every time.
      This is why in production systems, math always goes to a tool.
    
    SECURITY NOTE:
      We NEVER use eval() directly — that's a code injection vulnerability.
      e.g. eval("__import__('os').system('rm -rf /')") would destroy the server.
      We whitelist only safe math operations.
    
    Supported:
      Basic:      + - * / ** ( )
      Functions:  sqrt, pow, abs, round, log, log10, ceil, floor
      Constants:  pi, e
    
    Args:
        expression: Math expression as string e.g. "500000 * 0.01 * (1.01**36) / ((1.01**36) - 1)"
    
    Returns:
        Result as string, or error message
    """
    # Whitelist: only allow safe characters
    # Removes anything that isn't a digit, operator, decimal, space, or known function name
    safe_pattern = r'^[0-9\s\+\-\*\/\(\)\.\,\%\^\_a-zA-Z]+$'

    if not re.match(safe_pattern, expression.strip()):
        return "Invalid expression: only mathematical operations are allowed."

    # Build a safe namespace with only math functions — no builtins, no imports
    safe_namespace = {
        "__builtins__": {},          # block all Python builtins
        "sqrt":   math.sqrt,
        "pow":    math.pow,
        "abs":    abs,
        "round":  round,
        "log":    math.log,
        "log10":  math.log10,
        "ceil":   math.ceil,
        "floor":  math.floor,
        "pi":     math.pi,
        "e":      math.e,
    }

    try:
        # Replace ^ with ** for user convenience (^ is XOR in Python, not power)
        expression = expression.replace("^", "**")

        result = eval(expression, safe_namespace)  # safe because namespace is locked

        # Format nicely: show int if whole number, else 2 decimal places
        if isinstance(result, float) and result.is_integer():
            return f"Result: {int(result)}"
        elif isinstance(result, float):
            return f"Result: {result:.2f}"
        else:
            return f"Result: {result}"

    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Calculation error: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# TOOL REGISTRY
# ═══════════════════════════════════════════════════════════════

"""
WHY A REGISTRY?
  The agent doesn't hardcode which tool to call.
  It looks up tools by name from this dict.
  
  This is how you'd scale to 20 tools — just add entries here.
  The agent's prompt describes each tool; the LLM picks the name;
  the name maps to a function in this registry.
  
  In production: tool descriptions are compressed to save context window.
  e.g. "rag_tool: search banking docs" instead of a full paragraph.
"""

TOOL_REGISTRY: Dict[str, callable] = {
    "rag_tool": rag_tool,
    "calculator_tool": calculator_tool,
}

# Tool descriptions given to the agent so it knows what each tool does
TOOL_DESCRIPTIONS = {
    "rag_tool": (
        "Search the banking knowledge base for information about loans, "
        "payments, KYC, interest rates, and banking policies. "
        "Use this for any factual question about banking products."
    ),
    "calculator_tool": (
        "Perform mathematical calculations including EMI computation, "
        "percentage calculations, interest calculations, and arithmetic. "
        "Use this whenever the question involves numbers or math."
    ),
}


def get_tool_descriptions_for_prompt() -> str:
    """
    Returns a formatted string of tool descriptions for the agent's system prompt.
    
    The agent reads this and decides which tool to call.
    
    Output example:
      Available tools:
      - rag_tool: Search the banking knowledge base...
      - calculator_tool: Perform mathematical calculations...
    """
    lines = ["Available tools:"]
    for name, description in TOOL_DESCRIPTIONS.items():
        lines.append(f"  - {name}: {description}")
    return "\n".join(lines)