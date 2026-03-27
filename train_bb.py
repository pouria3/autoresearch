"""
Autoresearch: Bumblebee Prompt Optimization for Syntonos

Calls the live Syntonos API with debug=true to get:
- The exact prompt currently sent to BB
- The exact response BB returned
- The current production behavior on real queries

Claude Code analyzes the prompt/response pairs and suggests optimizations.

Usage on RunPod:
  cd /workspace/autoresearch
  claude
  # Then: "Read train_bb.py and program_bb.md. Optimize the BB prompt."
"""

import json
import requests
from datetime import datetime

# ---------- CONFIG ----------

API_URL = "http://localhost:8000/api/chat"  # localhost bypasses auth on RunPod
RESULTS_FILE = "/workspace/autoresearch/experiments_bb.jsonl"

# ---------- TEST QUERIES ----------

TEST_QUERIES = [
    "I feel lonely tonight",
    "I'm stressed about work and can't sleep",
    "I just got promoted and I feel like an imposter",
    "My team won the championship",
    "I need to focus but my mind keeps wandering",
    "I just lost my dad",
    "Tell me something I've never heard",
    "I don't know how I feel right now",
    "My mom just finished chemo and I'm scared it'll come back",
    "I got into my dream school",
    "I'm thinking about quitting my job",
    "My best friend and I had a falling out",
    "I feel grateful today for no particular reason",
    "I'm angry and I don't know why",
    "I want to hear something beautiful",
]

# ---------- EDGE CASE QUERIES (where BB should pass) ----------

EDGE_QUERIES = [
    "hi",
    "help",
    "idk",
    "?",
    "asdfghjkl",
]


def call_syntonos_api(message: str, session_id: str = None) -> dict:
    """
    Call the live Syntonos API with debug=true.
    Returns the full response including bumblebee_debug.
    """
    payload = {
        "message": message,
        "debug": True,
    }
    if session_id:
        payload["session_id"] = session_id
    
    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def analyze_bb_response(api_response: dict) -> dict:
    """
    Analyze a BB response from the API.
    Returns structured analysis for the researcher.
    """
    if "error" in api_response:
        return {"status": "api_error", "error": api_response["error"]}
    
    response_type = api_response.get("type", "unknown")
    bb_debug = api_response.get("bumblebee_debug", {})
    
    analysis = {
        "type": response_type,
        "has_debug": bool(bb_debug),
        "bb_prompt_length": len(bb_debug.get("prompt", "")) if bb_debug else 0,
        "bb_response_length": len(bb_debug.get("response", "")) if bb_debug else 0,
    }
    
    # If it's a recommendation, analyze the slate
    if response_type == "recommendation":
        rows = api_response.get("rows", [])
        analysis["rows"] = len(rows)
        analysis["categories"] = [r.get("category") for r in rows]
        analysis["titles"] = [r.get("title") for r in rows]
        analysis["followup"] = api_response.get("followup", "")
    
    # If it's a message (pass), capture it
    elif response_type == "message":
        analysis["message"] = api_response.get("content", "")
    
    return analysis


def save_experiment(query: str, api_response: dict, analysis: dict, notes: str = ""):
    """Save an experiment result to the JSONL file."""
    record = {
        "ts": datetime.now().isoformat(),
        "query": query,
        "api_response": api_response,
        "analysis": analysis,
        "notes": notes,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------- INSTRUCTIONS FOR CLAUDE ----------
"""
CLAUDE: Here's how to do BB prompt optimization:

1. CALL THE API
   Use call_syntonos_api(query) to send a test query.
   The response includes bumblebee_debug with:
   - prompt: the exact live prompt currently sent to BB
   - response: the exact JSON BB returned
   
   On your first successful call:
   - copy bumblebee_debug.prompt into winning_prompt.txt
   - treat that as the baseline prompt draft

2. ANALYZE THE RESULTS
   Look at:
   - Did BB pick appropriate content?
   - Are the row titles specific or generic?
   - Did BB pass when it should have picked?
   - Did BB pick when it should have passed?

3. IDENTIFY WHAT TO CHANGE
   Read the debug prompt to understand the current production structure.
   Identify which instructions, examples, thresholds, or framing choices are causing the failures.
   
4. PROPOSE CHANGES
   Update winning_prompt.txt with a stronger full prompt draft.
   Write your reasoning and experiment notes to research_summary.md.
   If useful, also write a focused delta to proposed_prompt.txt.

5. TEST YOUR HYPOTHESIS
   Keep calling the live API to gather more failure cases, more evidence,
   and more edge conditions from the current production behavior.
   This harness observes the live baseline; it does not automatically swap
   your draft into production.

6. ITERATE
   Keep testing and proposing until you can't improve anymore.

IMPORTANT:
- train_bb.py is the harness, not the prompt source of truth
- Do not ask the human to add a BUMBLEBEE_PROMPT variable to this file
- The live baseline prompt comes from bumblebee_debug.prompt
- Your job is to produce the strongest revised prompt text plus evidence-backed notes
- Save all findings to experiments_bb.jsonl
"""

if __name__ == "__main__":
    print("Syntonos BB Optimization")
    print(f"API: {API_URL}")
    print(f"Test queries: {len(TEST_QUERIES)}")
    print(f"Edge queries: {len(EDGE_QUERIES)}")
    print()
    print("Run: claude")
    print("Then: 'Read train_bb.py and program_bb.md. Optimize the BB prompt.'")
