"""
Autoresearch: Bumblebee Prompt Optimization for Syntonos

Calls the production API at syntonos.ai/api/chat with debug=true to get:
- The exact prompt sent to BB
- The exact response BB returned
- All candidates considered

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

API_URL = "https://syntonos.ai/api/chat"
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
    Call the production Syntonos API with debug=true.
    Returns the full response including bumblebee_debug.
    
    NOTE: This requires authentication. You may need to:
    1. Get a valid auth token from syntonos.ai
    2. Or use a test/dev endpoint that bypasses auth
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
   - prompt: the exact prompt sent to BB
   - response: the exact JSON BB returned

2. ANALYZE THE RESULTS
   Look at:
   - Did BB pick appropriate content?
   - Are the row titles specific or generic?
   - Did BB pass when it should have picked?
   - Did BB pick when it should have passed?

3. IDENTIFY THE PROMPT SECTION TO CHANGE
   The BB prompt is in backend/brain.py on the production server.
   Read the debug prompt to understand the current structure.
   
4. PROPOSE CHANGES
   Write your proposed prompt modification to proposed_prompt.txt
   Explain WHY you think it will improve results.

5. TEST YOUR HYPOTHESIS
   After the prompt is updated on the server, run the same queries
   and compare before/after.

6. ITERATE
   Keep testing and proposing until you can't improve anymore.

IMPORTANT:
- You cannot directly modify the production prompt
- Your job is to ANALYZE and PROPOSE changes
- The human will review and deploy your suggestions
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
