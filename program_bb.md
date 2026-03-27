# Bumblebee Prompt Optimization

This is an experiment to find the optimal Bumblebee system prompt for Syntonos/Radio20.

## Setup

1. Read `train_bb.py` for the live API harness, test queries, and result logging.
2. Make one successful `debug=true` call through the localhost endpoint. The returned `bumblebee_debug.prompt` is the real baseline prompt currently running on Syntonos.
3. Save that returned prompt into `winning_prompt.txt` before making any edits. That file is your working draft.
4. The live endpoint already runs the real production pipeline, so the returned behavior reflects actual search/rerank/Bumblebee output.

## What Bumblebee Does

Bumblebee is the curation LLM for Syntonos. A user shares how they're feeling → search returns candidates grouped by category → BB picks the best content and builds a "slate" (up to 5 rows, one per category, one pick per row). BB can also "pass" and ask a clarifying question.

## The File You Modify

`train_bb.py` is the harness. It does NOT contain a `BUMBLEBEE_PROMPT` variable.

You modify:
- `winning_prompt.txt` — your current best full prompt draft
- `research_summary.md` — what you tried, what you observed, what you recommend next

Do not ask the human to add a prompt variable to `train_bb.py`. Capture the live baseline from `bumblebee_debug.prompt` and work from there.

## How to Run

```bash
python train_bb.py
```

Use `train_bb.py` as the live-eval harness. Each API call captures baseline behavior from the real system and appends results to `experiments_bb.jsonl`.

## The Metric

Each response is scored 0-10:
- Did BB produce a valid JSON slate? (+4)
- How many rows (categories) in the slate? (+1 per row, up to 3)
- Title quality — max 12 words, max 60 chars? (+1 per good title, up to 2)
- Has a follow-up question? (+1)

Higher is better. The baseline should score 7-9 on most queries.

**But numbers aren't everything.** Also evaluate qualitatively:
- Did BB pick content that actually RESPONDS to the user's situation?
- Are the row titles descriptive of the content (not generic emotional labels)?
- When BB passes, is the follow-up question specific and useful?
- Is BB too strict (passing when it should pick) or too lenient (picking bad matches)?

## What You CAN Change

Everything in your local working draft (`winning_prompt.txt`). This includes:
- The persona/framing (guardian, friend, therapist, DJ, etc.)
- The 6-step decision tree (reorder, merge, split, add, remove steps)
- The examples and edge cases
- The strictness/leniency balance
- The pick vs pass criteria
- How titles are instructed
- How follow-up questions are instructed
- The emotional framing

Also include improved versions of related sub-instructions like first-turn nudges or slate instructions if you think they help.

## What You CANNOT Change

- The test queries (fixed set of 15)
- The database schema or connection
- The scoring function
- The candidate formatting (must match production)

## Logging Results

After each experiment, append to `experiments_bb.jsonl`. Also maintain a running log:

```
experiments_bb.jsonl  — raw results per query per variant
research_summary.md   — your running notes: what you tried, what worked, what didn't
winning_prompt.txt    — the best prompt so far (overwrite as you improve)
```

## The Experiment Loop

LOOP FOREVER:

1. Run the current prompt against all 15 queries
2. Analyze: where did BB fail? What patterns do you see?
3. Form a hypothesis about what to change
4. Update `winning_prompt.txt` with a better full prompt draft
5. Log the rationale in `research_summary.md`
6. Keep sampling the live baseline behavior to gather stronger evidence and edge cases
7. Log what you tried and what happened in research_summary.md
8. Repeat

**NEVER STOP**: Once started, do NOT pause to ask the human. Keep iterating until manually interrupted. The human might be asleep. If you run out of obvious ideas, think harder — try radical changes, combinations of previous near-misses, or entirely new framings. Try things nobody would think of.

**Think like a researcher, not an engineer.** Don't just tweak parameters. Ask WHY the prompt works the way it does. Challenge assumptions. What if the 6-step tree is overcomplicated? What if the persona matters more than the rules? What if the examples are teaching the wrong thing? What if less instruction produces better results?
