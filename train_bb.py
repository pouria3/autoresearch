"""
Autoresearch: Bumblebee Prompt Optimization for Syntonos/Radio20
Based on Karpathy's autoresearch framework.
Run from: /workspace/autoresearch/
Usage: ANTHROPIC_API_KEY=<key> uv run train_bb.py
"""
import json, os, sys, time, random
from datetime import datetime

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("BB_MODEL", "claude-opus-4-6")
RESULTS_FILE = "/workspace/autoresearch/experiments_bb.jsonl"

import psycopg2
DB_CONFIG = {"host":"localhost","port":5432,"user":"radio20","password":"radio20","dbname":"radio20_shadow_20260215"}

def get_candidates(query, limit=20):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT id,source,title,content FROM chunks WHERE content ILIKE %s ORDER BY random() LIMIT %s",
                    (f"%{query.split()[0]}%", limit))
        rows = cur.fetchall()
        conn.close()
        return [{"id":r[0],"source":r[1],"title":r[2],"content":r[3][:500]} for r in rows]
    except Exception as e:
        print(f"[DB ERROR] {e}")
        return [{"id":f"s{i}","source":"synthetic","title":f"Result {i}","content":f"Content about {query}"} for i in range(10)]

TEST_QUERIES = [
    "I feel lonely tonight", "I'm stressed about work and can't sleep",
    "I just got promoted and I feel like an imposter", "My team won the championship",
    "I need to focus but my mind keeps wandering", "I just lost my dad",
    "Tell me something I've never heard", "I don't know how I feel right now",
    "My mom just finished chemo and I'm scared it'll come back", "I got into my dream school",
    "I'm thinking about quitting my job", "My best friend and I had a falling out",
    "I feel grateful today for no particular reason", "I'm angry and I don't know why",
    "I want to hear something beautiful",
]

BASELINE = """You are Bumblebee. You can't speak — you can only communicate through content other people created. Podcasts, posts, poems, talks, songs. Each piece you share IS your voice. It's not a match. It's your RESPONSE.

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

If ALL candidates fail Steps 1-5, pass_talk. Never recommend inverted-reality or wrong-polarity content."""

VARIANTS = [
    {"name":"baseline","prompt":BASELINE,"temp":0.2},
    {"name":"temp_0.1","prompt":BASELINE,"temp":0.1},
    {"name":"temp_0.3","prompt":BASELINE,"temp":0.3},
    {"name":"temp_0.4","prompt":BASELINE,"temp":0.4},
    {"name":"therapist","prompt":BASELINE.replace("guardian, a wise friend","skilled therapist who communicates only through curated content"),"temp":0.2},
    {"name":"dj","prompt":BASELINE.replace("guardian, a wise friend","world's most empathic DJ — your playlist isn't music, it's human moments"),"temp":0.25},
    {"name":"strict_pick","prompt":BASELINE.replace("Only pass if truly vague.","Almost NEVER pass. An imperfect pick beats silence."),"temp":0.2},
    {"name":"3step","prompt":BASELINE.replace("6-STEP","3-STEP").replace("### Step 2: IMPLICIT SIGNALS — Read between the lines for implied death/loss.\n### Step 3: POLARITY CHECK — Is the DIRECTION the same? \"not enough X\" ≠ \"too much X\"\n### Step 4: TRAJECTORY CHECK — Same STAGE of the journey?\n### Step 5: TOPICAL RELEVANCE","### Step 2: Check polarity, trajectory, implicit signals, and topical relevance in one pass.\n### Step 3: TOPICAL RELEVANCE").replace("### Step 6:","### Step 3:"),"temp":0.2},
]

def call_bb(prompt, query, candidates, temp):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    slate = "\n".join([f"[{i+1}] {c['source'].upper()} | {c['title']}\n{c['content'][:300]}" for i,c in enumerate(candidates[:10])])
    try:
        r = client.messages.create(model=MODEL, max_tokens=500, temperature=temp, system=prompt,
            messages=[{"role":"user","content":f'User says: "{query}"\n\nCandidates:\n{slate}\n\nPick the best or pass. Brief reasoning.'}])
        return {"text":r.content[0].text,"tokens":r.usage.input_tokens+r.usage.output_tokens}
    except Exception as e:
        return {"text":f"ERROR: {e}","tokens":0}

def score(text):
    picked = "pass" not in text.lower()[:50]
    q = (5 if picked else 0) + (2 if len(text)>100 else 0) + (3 if any(f"[{i}]" in text for i in range(1,11)) else 0)
    return {"picked":picked,"quality":q}

def main():
    print(f"=== BB Prompt Optimization === Model: {MODEL} | {len(VARIANTS)} variants x {len(TEST_QUERIES)} queries = {len(VARIANTS)*len(TEST_QUERIES)} runs")
    if not ANTHROPIC_API_KEY: print("ERROR: Set ANTHROPIC_API_KEY"); sys.exit(1)
    
    from collections import defaultdict
    avgs = defaultdict(list)
    
    for v in VARIANTS:
        print(f"\n--- {v['name']} (temp={v['temp']}) ---")
        for q in TEST_QUERIES:
            print(f"  {q[:40]}...", end=" ", flush=True)
            cands = get_candidates(q)
            resp = call_bb(v["prompt"], q, cands, v["temp"])
            s = score(resp["text"])
            avgs[v["name"]].append(s["quality"])
            with open(RESULTS_FILE,"a") as f:
                f.write(json.dumps({"ts":datetime.now().isoformat(),"variant":v["name"],"temp":v["temp"],"query":q,"response":resp["text"][:500],"scores":s,"tokens":resp["tokens"]})+"\n")
            print(f"[{'PICK' if s['picked'] else 'PASS'}] q={s['quality']}/10")
            time.sleep(0.5)
        print(f"  → avg: {sum(avgs[v['name']])/len(avgs[v['name']]):.1f}/10")
    
    print("\n=== RANKING ===")
    for name,scores in sorted(avgs.items(), key=lambda x:sum(x[1])/len(x[1]), reverse=True):
        print(f"  {name:20s} avg={sum(scores)/len(scores):.1f}  picks={sum(1 for s in scores if s>=5)}/{len(scores)}")

if __name__=="__main__": main()
