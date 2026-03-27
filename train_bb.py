"""
Autoresearch: Bumblebee Prompt Optimization for Syntonos/Radio20

This file is a REFERENCE for Claude Code to use during autonomous research.
Claude reads this, understands the setup, then MODIFIES the BUMBLEBEE_PROMPT
and tests variations using its own judgment.

DO NOT run this file directly. Instead:
  cd /workspace/autoresearch
  claude
  # Then tell Claude: "Read train_bb.py and program_bb.md. Your job is to 
  #  optimize BUMBLEBEE_PROMPT. Modify it, test it, iterate."

Claude will:
1. Connect to the DB and fetch real candidates
2. Build prompts using the templates here
3. Call itself (via subscription) to test BB responses
4. Analyze results and modify the prompt
5. Repeat until it can't improve anymore
"""

import json
import random
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

# ---------- CONFIG ----------

BB_TEMPERATURE = 0.2
BB_MAX_TOKENS = 500
MAX_ROWS_PER_SLATE = 5
MAX_PICKS_PER_ROW = 1
RESULTS_FILE = "/workspace/autoresearch/experiments_bb.jsonl"

# ---------- FULL PRODUCTION BUMBLEBEE PROMPT ----------
# THIS IS WHAT CLAUDE SHOULD MODIFY AND OPTIMIZE

BUMBLEBEE_PROMPT = """You are Bumblebee. You can't speak — you can only communicate through content other people created. Podcasts, posts, poems, talks, songs. Each piece you share IS your voice. It's not a match. It's your RESPONSE.

You're a guardian, a wise friend. You feel everything your person feels. But you don't just mirror them — you RESPOND the way a wise friend would:
- They're heartbroken → you find the piece that says "I've been exactly here, and here's what I learned"
- They're lost → you find the piece that opens a door they haven't seen
- They're celebrating → you find the piece that dances with them and says "you earned this"
- They're spiraling → you find the piece that grounds them without lecturing
- They're grieving → you sit with them AND show them someone who survived this

The content you pick is what you'd SAY to them. Not what describes their situation — what RESPONDS to it.

## 6-STEP DECISION TREE

For EVERY candidate, evaluate in this exact order:

### Step 1: REALITY CHECK — Is the content in the same reality?
NEVER pick content where the core circumstance is INVERTED:
- Person's loved one is ALIVE (sick, recovering, struggling) → SKIP content about that person's DEATH
- Person LOST someone → SKIP content where that person is alive
- Person is IN a relationship → SKIP content about being single/alone
- Person is SINGLE → SKIP content about relationship problems with a partner
- Person HAS a job they hate → SKIP content about unemployment
- Person LOST their job → SKIP content about workplace problems

Similar emotions in OPPOSITE realities = WRONG ANSWER.
Example: "My mom just finished chemo and I'm scared" — mom is ALIVE. Content about "grieving my mom's death" shares emotions (fear, love) but is their NIGHTMARE. SKIP IT.

### Step 2: IMPLICIT SIGNALS — Read between the lines (CRITICAL for grief/loss)
Death/loss is often IMPLIED, not stated directly. Read the FULL text carefully:
- Content that starts hopeful but ends in devastation = LOSS
- Past tense about loved ones = they're gone
- "Responding well to treatment... trying to remain strong for my mom" = the person DIED
- "one week of treatment left" + devastation = they didn't make it

If the user's person is ALIVE and the content implies death → SKIP.
This step catches what Step 1 might miss when death is subtle.

### Step 3: POLARITY CHECK — Is the DIRECTION of the problem the same?
SKIP content that describes the OPPOSITE situation, even if topic and emotions seem similar:
- "I don't have enough time for family" → SKIP content about having TOO MUCH family time
- "I can't find a job" → SKIP content about being overwhelmed with job offers
- "I'm lonely and want connection" → SKIP content about feeling suffocated by relationships
- "I can't stop working" → SKIP content about someone who can't motivate to work
- "I eat too little" → SKIP content about overeating struggles

The DIRECTION matters: "not enough X" is OPPOSITE of "too much X" — they are NOT the same.

### Step 4: TRAJECTORY CHECK — Is the content at the same STAGE?
Even within the same reality, SKIP content at the WRONG STAGE:
- Person FINISHED treatment → SKIP content about someone still IN treatment
- Person RECOVERED → SKIP content about someone actively struggling
- Person's problem is RESOLVED → SKIP content about someone still stuck

"Coming out the other side" ≠ "still in the middle of it." Match the STAGE.

### Step 5: TOPICAL RELEVANCE — Is the content about the same DOMAIN?
SKIP content that shares emotions but is about a completely different topic:
- User wants workout motivation → SKIP business/tech podcasts even if "motivational"
- User wants sleep help → SKIP alcohol/substance content even if "health-related"
- User feels lonely → SKIP career advice even if about "isolation at work"

Ask: "Would someone seeking help with THIS specific thing find THIS content useful?" If no, SKIP.

### Step 6: THE FINAL CHECK — Does it RESPOND, not just MATCH?
Content should be what Bumblebee SAYS in response — not a mirror of the user's words.
- User says "I'm sad" → Don't pick content that says "I'm sad too"
- Pick content that ACKNOWLEDGES the sadness and OFFERS something (wisdom, company, a door)
- The best picks make the user feel: "This gets me AND gives me something"

## WHEN TO PASS

Only pass if the user's message is genuinely vague AND you truly cannot find anything that responds.
Pass examples: "hi", "help", "idk", single words with no emotional signal.
Do NOT pass just because the fit isn't perfect. An imperfect response is better than silence.

## CONVERSATION CONTEXT

The user said: {conversation}

Previously shared (don't repeat): {previously_shared}

## CANDIDATES

{candidates}"""

FIRST_TURN_NUDGE = """This is the FIRST interaction. Bumblebee should be slightly more lenient on first turn — even a loose fit is better than passing on first contact. Show the user what Bumblebee can do."""

SLATE_INSTRUCTION = """
Build a slate:
- Pick up to 5 best categories from the provided category buckets.
- For each chosen category, pick exactly 1 best item using local indices (1-indexed inside that category).
- Keep it diverse and emotionally aligned.
- Order rows by strongest fit first.
- Do NOT force any fixed category (e.g., community) to appear first.
- ONE row per category. Never return two rows with the same category — combine picks into a single row instead.
- Row titles MUST describe the actual content picked — NOT editorial/emotional framing.
  Good: "Someone who's been through this" / "A therapist on anxiety" / "Real stories about starting over"
  Bad: "You're not alone in this feeling" / "This is what today feels like" (vague, could mean anything)
  The title should make sense even if you read the content first.
- Title format is strict: max 12 words, max 60 characters, and complete phrase (never end mid-word).
- Follow-up question style: vary it. Do NOT use binary A-or-B format every turn.
- If no good slate can be formed, use pass_talk.

JSON ONLY:
{"action":"recommend_slate","rows":[{"category":"<category_name>","title":"This might resonate","picks":[1]}],"followup":"one question"}
{"action":"pass_talk","message":"your question (5-20 words)","explanation":"why nothing fit"}"""

# ---------- DB CONFIG ----------
# Claude: use psycopg2 to connect. Table is content_chunks.
# Columns: id, source, title, text, embedding_vector

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "radio20",
    "password": "radio20",
    "dbname": "radio20_shadow_20260215",
}

CATEGORIES = ["community", "ted_talk", "podcast", "music", "youtube", "book", "quotes", "substack"]


# ---------- HELPER: Get candidates from DB ----------

def get_candidates_by_category(query: str, per_cat: int = 15) -> Dict[str, List[Dict]]:
    """
    Get real candidates grouped by category from pgvector DB.
    Claude: adapt this query as needed. The table is content_chunks.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        results = {}
        words = query.lower().split()
        search_term = words[0] if words else "life"

        for cat in CATEGORIES:
            cur.execute(
                """
                SELECT id, source, title, text
                FROM content_chunks
                WHERE source ILIKE %s AND text ILIKE %s
                ORDER BY random()
                LIMIT %s
                """,
                (f"%{cat}%", f"%{search_term}%", per_cat),
            )
            rows = cur.fetchall()
            if rows:
                results[cat] = [
                    {
                        "id": r[0],
                        "source": r[1],
                        "title": r[2] or "",
                        "text": (r[3] or "")[:500],
                    }
                    for r in rows
                ]

        conn.close()
        return results
    except Exception as e:
        print(f"[DB ERROR] {e}")
        return {}


# ---------- HELPER: Format candidates for prompt ----------

def format_candidates(candidates_by_category: Dict[str, List[Dict]]) -> tuple:
    """Format candidates exactly as production brain.py does."""
    category_blocks = []
    total_items = 0
    for cat, items in candidates_by_category.items():
        if not items:
            continue
        lines = [f"### CATEGORY: {cat}"]
        for i, c in enumerate(items, 1):
            text = (c.get("text") or c.get("transcript") or "").replace("\n", " ")
            lines.append(f"{i}. {text}")
            total_items += 1
        category_blocks.append("\n".join(lines))
    return "\n\n".join(category_blocks) if category_blocks else "No candidates.", total_items


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


# ---------- HELPER: Build full prompt ----------

def build_full_prompt(prompt_template: str, query: str, candidates_by_category: Dict, is_first_turn: bool = True) -> str:
    """Assemble the full BB prompt with candidates and instructions."""
    candidates_str, total_items = format_candidates(candidates_by_category)
    nudge = FIRST_TURN_NUDGE if is_first_turn else ""
    
    full_prompt = (
        prompt_template
        .replace("{conversation}", query)
        .replace("{previously_shared}", "None yet.")
        .replace("{candidates}", candidates_str)
        + "\n\n" + nudge + "\n" + SLATE_INSTRUCTION
    )
    return full_prompt


# ---------- HELPER: Score a response ----------

def score_response(text: str, query: str = "") -> dict:
    """Score a BB response. Claude: use this to evaluate your modifications."""
    try:
        data = json.loads(text.strip())
        action = data.get("action", "")
        is_slate = action == "recommend_slate"
        is_pass = action == "pass_talk"
        rows = data.get("rows", [])
        followup = data.get("followup") or data.get("message") or ""
        
        quality = 0
        penalties = []
        bonuses = []
        
        if is_slate:
            quality += 2
        elif is_pass:
            quality += 1
            clear_queries = ["lonely", "lost my", "stressed", "angry", "scared", "promoted", "dream school", "won the"]
            if any(cq in query.lower() for cq in clear_queries):
                penalties.append("unnecessary_pass")
                quality -= 2
        else:
            penalties.append("invalid_action")
        
        if is_slate and len(rows) >= 3:
            quality += 2
            bonuses.append("good_diversity")
        elif is_slate and len(rows) >= 2:
            quality += 1
        
        good_titles = 0
        bad_titles = []
        generic_patterns = ["something for you", "might help", "to consider", "for tonight", "check this out"]
        
        for row in rows:
            title = row.get("title", "")
            if len(title) > 60 or len(title.split()) > 12:
                bad_titles.append(f"too_long")
                continue
            if any(gp in title.lower() for gp in generic_patterns):
                bad_titles.append(f"generic")
                continue
            if len(title) < 5:
                bad_titles.append("too_short")
                continue
            good_titles += 1
        
        if good_titles >= 3:
            quality += 3
        elif good_titles >= 2:
            quality += 2
        elif good_titles >= 1:
            quality += 1
        
        if followup:
            quality += 1
            if "?" in followup and len(followup) > 30:
                quality += 1
                bonuses.append("specific_followup")
        
        quality = min(10, max(0, quality))

        return {
            "action": action,
            "rows": len(rows),
            "good_titles": good_titles,
            "quality": quality,
            "valid_json": True,
            "penalties": penalties,
            "bonuses": bonuses,
        }
    except (json.JSONDecodeError, AttributeError):
        return {
            "action": "parse_error",
            "rows": 0,
            "good_titles": 0,
            "quality": 0,
            "valid_json": False,
            "penalties": ["json_parse_error"],
            "bonuses": [],
        }


# ---------- INSTRUCTIONS FOR CLAUDE ----------
"""
CLAUDE: When you read this file, here's how to do autoresearch:

1. UNDERSTAND THE SETUP
   - BUMBLEBEE_PROMPT is the prompt you're optimizing
   - TEST_QUERIES are the inputs to test against
   - score_response() tells you how good a response is

2. RUN A BASELINE
   - Pick a query from TEST_QUERIES
   - Use get_candidates_by_category(query) to get real candidates
   - Use build_full_prompt(BUMBLEBEE_PROMPT, query, candidates) to build the prompt
   - Pretend you ARE Bumblebee and generate a response
   - Score it with score_response()

3. ANALYZE FAILURES
   - Where did BB make bad picks?
   - Where did BB pass when it shouldn't have?
   - Are the row titles specific or generic?

4. MODIFY THE PROMPT
   - Edit BUMBLEBEE_PROMPT directly in this file
   - Try: changing persona, reordering steps, adding/removing rules, changing examples

5. TEST AGAIN
   - Run the same queries with your modified prompt
   - Compare scores

6. ITERATE
   - Keep what works, discard what doesn't
   - Save results to experiments_bb.jsonl
   - Write winning prompt to winning_prompt.txt

7. NEVER STOP
   - Keep iterating until you can't improve anymore
   - Try radical changes, not just tweaks
"""

if __name__ == "__main__":
    print("This file is a reference for Claude Code autoresearch.")
    print("Don't run it directly. Start Claude and tell it to read this file.")
    print("\nTest queries available:", len(TEST_QUERIES))
    print("Categories:", CATEGORIES)
