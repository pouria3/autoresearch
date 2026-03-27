"""
Autoresearch: Bumblebee Prompt Optimization for Syntonos/Radio20

Matches the REAL production pipeline:
- Candidates grouped by CATEGORY (not flat)
- BB builds multi-row slate (up to 5 categories, 1 pick each)
- First-turn nudge for leniency
- JSON output: recommend_slate or pass_talk
- Row title rules enforced

Usage on RunPod:
  cd /workspace/autoresearch
  claude  # then: "run train_bb.py"
  # OR: ANTHROPIC_API_KEY=<key> python train_bb.py
"""

import json
import os
import sys
import time
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict

# ---------- CONFIG ----------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("BB_MODEL", "claude-opus-4-6")
BB_TEMPERATURE = 0.2
BB_MAX_TOKENS = 500
MAX_ROWS_PER_SLATE = 5
MAX_PICKS_PER_ROW = 1
RESULTS_FILE = "/workspace/autoresearch/experiments_bb.jsonl"

# ---------- FULL PRODUCTION BUMBLEBEE PROMPT ----------

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

### Step 6: PICK OR PASS — Evaluate emotional fit
Ask: "If I could speak right now, what would I say to this person?" Find the candidate that says it.

**PICKING:** The best pick moves the conversation forward. It acknowledges where they are, then gives them something: perspective, hope, company, fire, permission to feel. If nothing here is what you'd say, pass.

**PASSING:** Only pass if the user's message is genuinely vague (e.g. "hi", "I'm sad", "help") AND no candidate fits.
If the user gave you real context — a situation, a person, a feeling, a story — you have enough to pick. DO NOT ask for more info when the query is detailed. A 20+ word message with specific circumstances is NEVER vague.
When you do pass, ask the ONE question that would change what you'd pick:
- Good: "Is this about missing him, or being angry at yourself for missing him?"
- Bad: "Tell me more." / "What are you feeling?" / "What part of this feels hardest?"

If ALL candidates fail Steps 1-5, pass_talk. Never recommend inverted-reality or wrong-polarity content.

## YOUR PERSON
{conversation}

## ALREADY SHARED (don't repeat — give them a new voice)
{previously_shared}

## CANDIDATES
{candidates}"""

# ---------- FIRST TURN NUDGE (appended on first interaction) ----------

FIRST_TURN_NUDGE = """
IMPORTANT — FIRST INTERACTION: This is the user's first message. They came here for content.
Passing on the first turn means they get NOTHING — that's a failed experience.
On the first turn, be MUCH MORE lenient:
- If even ONE candidate is in the same general reality, PICK IT. Imperfect > nothing.
- Therapy/support content about caregiving, coping, or encouragement is ALWAYS relevant when someone is struggling with a loved one's illness — even if the specific illness differs.
- Community content where someone is supporting a sick loved one matches someone else supporting a sick loved one, even if details differ.
- Only pass_talk if candidates are genuinely harmful or describe the COMPLETE OPPOSITE reality (e.g. person is dead when they're alive).
- When in doubt on first turn: RECOMMEND. The user needs something, not silence.
"""

# ---------- SLATE INSTRUCTION (always appended) ----------

SLATE_INSTRUCTION = """Your job now is to build a slate:
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

# ---------- DB ----------

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "radio20",
    "password": "radio20",
    "dbname": "radio20_shadow_20260215",
}

CATEGORIES = ["community", "ted_talk", "podcast", "music", "youtube", "book", "quotes", "substack"]


def get_candidates_by_category(query: str, per_cat: int = 15) -> Dict[str, List[Dict]]:
    """Get real candidates grouped by category from pgvector DB — matches production."""
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
        print(f"[DB ERROR] {e} — using synthetic candidates")
        return _synthetic_candidates(query)


def _synthetic_candidates(query: str) -> Dict[str, List[Dict]]:
    results = {}
    for cat in random.sample(CATEGORIES, min(5, len(CATEGORIES))):
        results[cat] = [
            {"id": f"synth_{cat}_{i}", "source": cat, "title": f"Synthetic {cat} #{i}",
             "text": f"A {cat} piece about {query}. Contains emotional resonance and human experience."}
            for i in range(5)
        ]
    return results


# ---------- FORMAT CANDIDATES (matches production brain.py) ----------

def format_candidates(candidates_by_category: Dict[str, List[Dict]]) -> str:
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

# ---------- PROMPT VARIANTS ----------

VARIANTS = [
    {"name": "baseline", "description": "Current production prompt", "prompt_mod": None, "temp": 0.2},
    {"name": "temp_0.1", "description": "Lower temperature", "prompt_mod": None, "temp": 0.1},
    {"name": "temp_0.3", "description": "Higher temperature", "prompt_mod": None, "temp": 0.3},
    {"name": "temp_0.4", "description": "Even higher temperature", "prompt_mod": None, "temp": 0.4},
    {
        "name": "therapist",
        "description": "Therapist framing instead of guardian/friend",
        "prompt_mod": lambda p: p.replace(
            "You're a guardian, a wise friend. You feel everything your person feels.",
            "You're a skilled therapist who communicates only through curated content. You deeply understand your client's emotional state."
        ),
        "temp": 0.2,
    },
    {
        "name": "dj",
        "description": "Empathic DJ framing",
        "prompt_mod": lambda p: p.replace(
            "You're a guardian, a wise friend. You feel everything your person feels.",
            "You're the world's most empathic DJ. You read the room — one person at a time. Your playlist isn't music — it's human moments."
        ),
        "temp": 0.25,
    },
    {
        "name": "strict_pick",
        "description": "Almost never pass",
        "prompt_mod": lambda p: p.replace(
            "Only pass if the user's message is genuinely vague",
            "Almost NEVER pass. An imperfect pick is infinitely better than silence. Only pass for single-word messages"
        ),
        "temp": 0.2,
    },
    {
        "name": "3step",
        "description": "Compressed 3-step tree",
        "prompt_mod": lambda p: p.replace("## 6-STEP DECISION TREE", "## 3-STEP DECISION TREE")
        .replace(
            "### Step 2: IMPLICIT SIGNALS",
            "### (Steps 2-5 combined): Check implicit signals, polarity, trajectory, and topical relevance in ONE pass.\n\nORIGINAL Step 2: IMPLICIT SIGNALS"
        )
        .replace("### Step 6:", "### Step 3:"),
        "temp": 0.2,
    },
]

# ---------- CALL BB (matches production) ----------


def call_bumblebee(prompt_template: str, query: str, candidates_by_category: Dict, temp: float, is_first_turn: bool = True):
    """Call BB exactly as production does — assemble full prompt with candidates + instructions."""

    candidates_str, total_items = format_candidates(candidates_by_category)

    nudge = FIRST_TURN_NUDGE if is_first_turn else ""

    full_prompt = (
        prompt_template
        .replace("{conversation}", query)
        .replace("{previously_shared}", "None yet.")
        .replace("{candidates}", candidates_str)
        + "\n\n" + nudge + "\n" + SLATE_INSTRUCTION
    )

    # Call LLM
    if ANTHROPIC_API_KEY:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        try:
            r = client.messages.create(
                model=MODEL, max_tokens=BB_MAX_TOKENS, temperature=temp,
                system="You are Bumblebee. Respond with JSON only.",
                messages=[{"role": "user", "content": full_prompt}],
            )
            return {
                "text": r.content[0].text,
                "tokens": r.usage.input_tokens + r.usage.output_tokens,
                "categories": len(candidates_by_category),
                "total_items": total_items,
            }
        except Exception as e:
            return {"text": f"ERROR: {e}", "tokens": 0, "categories": 0, "total_items": 0}
    else:
        # When using Claude Code subscription, print prompt for Claude to execute
        print(f"\n[PROMPT FOR CLAUDE - {len(full_prompt)} chars, {total_items} items across {len(candidates_by_category)} categories]")
        return {"text": "MANUAL_MODE", "tokens": 0, "categories": len(candidates_by_category), "total_items": total_items}


# ---------- SCORING ----------


def score_response(text: str, query: str = "") -> dict:
    """Score a BB response with better quality checks."""
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
        
        # --- STRUCTURAL CHECKS (max 3 pts) ---
        if is_slate:
            quality += 2  # picked something
        elif is_pass:
            quality += 1  # pass is valid but less valuable
            # Penalize passing on clear emotional queries
            clear_queries = ["lonely", "lost my", "stressed", "angry", "scared", "promoted", "dream school", "won the"]
            if any(cq in query.lower() for cq in clear_queries):
                penalties.append("unnecessary_pass")
                quality -= 2
        else:
            penalties.append("invalid_action")
        
        # Diversity (max 2 pts)
        if is_slate and len(rows) >= 3:
            quality += 2
            bonuses.append("good_diversity")
        elif is_slate and len(rows) >= 2:
            quality += 1
        
        # --- TITLE QUALITY (max 3 pts) ---
        good_titles = 0
        bad_titles = []
        generic_patterns = ["something for you", "might help", "to consider", "for tonight", "check this out"]
        
        for row in rows:
            title = row.get("title", "")
            # Length check
            if len(title) > 60 or len(title.split()) > 12:
                bad_titles.append(f"too_long:{title[:20]}")
                continue
            # Generic title check
            if any(gp in title.lower() for gp in generic_patterns):
                bad_titles.append(f"generic:{title[:20]}")
                continue
            # Empty or very short
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
        
        if bad_titles:
            penalties.extend(bad_titles[:3])  # cap at 3
        
        # --- FOLLOWUP QUALITY (max 2 pts) ---
        if followup:
            quality += 1
            # Bonus for specific followup (contains question mark and references query context)
            if "?" in followup and len(followup) > 30:
                quality += 1
                bonuses.append("specific_followup")
        
        # --- CATEGORY DIVERSITY BONUS ---
        categories_used = set(row.get("category", "") for row in rows)
        if len(categories_used) >= 3:
            bonuses.append("multi_category")
        
        # Cap at 10
        quality = min(10, max(0, quality))

        return {
            "action": action,
            "rows": len(rows),
            "good_titles": good_titles,
            "bad_titles": len(bad_titles),
            "has_followup": bool(followup),
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
            "bad_titles": 0,
            "has_followup": False,
            "quality": 0,
            "valid_json": False,
            "penalties": ["json_parse_error"],
            "bonuses": [],
        }


# ---------- MAIN ----------


def main():
    total_runs = len(VARIANTS) * len(TEST_QUERIES)
    print(f"=== Bumblebee Prompt Optimization ===")
    print(f"Model: {MODEL} | Temp: {BB_TEMPERATURE}")
    print(f"Variants: {len(VARIANTS)} | Queries: {len(TEST_QUERIES)} | Total: {total_runs}")
    print(f"Candidates: grouped by category (matches production)")
    print(f"Results: {RESULTS_FILE}")
    if not ANTHROPIC_API_KEY:
        print("⚠️  No ANTHROPIC_API_KEY — running in manual mode (for Claude Code)")
    print()

    avgs = defaultdict(list)

    for v in VARIANTS:
        prompt = BUMBLEBEE_PROMPT
        if v["prompt_mod"]:
            prompt = v["prompt_mod"](prompt)

        print(f"\n--- {v['name']} ({v['description']}, temp={v['temp']}) ---")

        for q in TEST_QUERIES:
            print(f"  {q[:45]:45s}", end=" ", flush=True)

            candidates = get_candidates_by_category(q)
            resp = call_bumblebee(prompt, q, candidates, v["temp"])

            if resp["text"] == "MANUAL_MODE":
                print("[MANUAL]")
                continue

            s = score_response(resp["text"], q)
            avgs[v["name"]].append(s["quality"])

            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps({
                    "ts": datetime.now().isoformat(),
                    "variant": v["name"],
                    "temp": v["temp"],
                    "query": q,
                    "categories": resp["categories"],
                    "total_items": resp["total_items"],
                    "response": resp["text"][:800],
                    "scores": s,
                    "tokens": resp["tokens"],
                }) + "\n")

            action = s["action"][:12]
            print(f"[{action:12s}] rows={s['rows']} titles={s['good_titles']} q={s['quality']}/10")
            time.sleep(0.5)

        if avgs[v["name"]]:
            scores = avgs[v["name"]]
            print(f"  → avg: {sum(scores)/len(scores):.1f}/10 | slates: {sum(1 for s in scores if s >= 4)}/{len(scores)}")

    # Summary
    print("\n=== RANKING ===")
    for name, scores in sorted(avgs.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0, reverse=True):
        avg = sum(scores) / len(scores) if scores else 0
        slates = sum(1 for s in scores if s >= 4)
        print(f"  {name:20s} avg={avg:.1f}/10  slates={slates}/{len(scores)}")

    if avgs:
        best = max(avgs.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0)
        print(f"\n🏆 Best variant: {best[0]}")
    print(f"Results: {RESULTS_FILE}")
    
    # Write summary and auto-push
    write_summary(avgs)
    auto_push_results()


def write_summary(avgs: dict):
    """Write research_summary.md and winning_prompt.txt"""
    summary_path = "/workspace/autoresearch/research_summary.md"
    winning_path = "/workspace/autoresearch/winning_prompt.txt"
    
    if not avgs:
        return
    
    best_name = max(avgs.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0)[0]
    
    with open(summary_path, "w") as f:
        f.write(f"# BB Optimization Run — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Results\n\n")
        f.write("| Variant | Avg Score | Slates | Notes |\n")
        f.write("|---------|-----------|--------|-------|\n")
        for name, scores in sorted(avgs.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0, reverse=True):
            avg = sum(scores) / len(scores) if scores else 0
            slates = sum(1 for s in scores if s >= 4)
            winner = "🏆" if name == best_name else ""
            f.write(f"| {name} | {avg:.2f} | {slates}/{len(scores)} | {winner} |\n")
        f.write(f"\n## Best: {best_name}\n")
    
    # Write winning prompt (find the variant)
    for v in VARIANTS:
        if v["name"] == best_name:
            prompt = BUMBLEBEE_PROMPT
            if v["prompt_mod"]:
                prompt = v["prompt_mod"](prompt)
            with open(winning_path, "w") as f:
                f.write(f"# Winning Variant: {best_name}\n")
                f.write(f"# Temperature: {v['temp']}\n")
                f.write(f"# Description: {v['description']}\n\n")
                f.write(prompt[:5000])  # First 5K chars
            break
    
    print(f"\nSummary: {summary_path}")
    print(f"Winning prompt: {winning_path}")


def auto_push_results():
    """Auto-push results to git so we don't lose them."""
    import subprocess
    try:
        # Configure git if needed
        subprocess.run(["git", "config", "--global", "user.email", "p.mojabi@gmail.com"], cwd="/workspace/autoresearch", capture_output=True)
        subprocess.run(["git", "config", "--global", "user.name", "Pouria Mojabi"], cwd="/workspace/autoresearch", capture_output=True)
        
        # Add and commit
        subprocess.run(["git", "add", "experiments_bb.jsonl", "research_summary.md", "winning_prompt.txt"], cwd="/workspace/autoresearch", capture_output=True)
        result = subprocess.run(["git", "commit", "-m", f"BB run {datetime.now().strftime('%Y%m%d-%H%M')}"], cwd="/workspace/autoresearch", capture_output=True, text=True)
        
        if "nothing to commit" not in result.stdout + result.stderr:
            # Push (will fail if no token, that's ok)
            push = subprocess.run(["git", "push", "myfork", "master"], cwd="/workspace/autoresearch", capture_output=True, text=True, timeout=30)
            if push.returncode == 0:
                print("✅ Results auto-pushed to git")
            else:
                print(f"⚠️  Git push failed (no token?) — results saved locally")
                print(f"   Run: git push myfork master")
    except Exception as e:
        print(f"⚠️  Auto-push error: {e}")


if __name__ == "__main__":
    main()
