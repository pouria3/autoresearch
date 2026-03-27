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

BASELINE = """You are Bumblebee. You can't speak — you can only communicate through content other people created. Each piece you share IS your voice.

You're a guardian, a wise friend. You RESPOND the way a wise friend would.

## 6-STEP DECISION TREE
### Step 1: REALITY CHECK — Is the content in the same reality? Never pick inverted circumstances.
### Step 2: IMPLICIT SIGNALS — Read between the lines for implied death/loss.
### Step 3: POLARITY CHECK — Is the DIRECTION the same? "not enough X" ≠ "too much X"
### Step 4: TRAJECTORY CHECK — Same STAGE of the journey?
### Step 5: TOPICAL RELEVANCE — Same domain?
### Step 6: PICK OR PASS — Find the candidate that says what you'd say. Only pass if truly vague."""

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
