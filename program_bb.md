# Bumblebee Prompt Optimization

This is an experiment to find the optimal Bumblebee system prompt for Syntonos/Radio20.

## Setup

1. Read `train_bb.py` for full context — it contains the production Bumblebee prompt, DB connection, candidate formatting, scoring, and test queries.
2. Read the `BUMBLEBEE_PROMPT` variable — that's your baseline prompt. It's the real production prompt running on syntonos.ai right now.
3. The database `content_chunks` table has ~3.2M rows of real content (TED talks, podcasts, community posts, music, books, youtube, quotes, substack).

## What Bumblebee Does

Bumblebee is the curation LLM for Syntonos. A user shares how they're feeling → search returns candidates grouped by category → BB picks the best content and builds a "slate" (up to 5 rows, one per category, one pick per row). BB can also "pass" and ask a clarifying question.

## The File You Modify

`train_bb.py` contains the `BUMBLEBEE_PROMPT` variable. That's the prompt you iterate on. Everything else (DB connection, scoring, test queries) stays fixed.

## How to Run

```bash
python train_bb.py
```

Each run tests the current prompt against all 15 test queries using real candidates from the database. Results append to `experiments_bb.jsonl`.

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

Everything in the `BUMBLEBEE_PROMPT` string. This includes:
- The persona/framing (guardian, friend, therapist, DJ, etc.)
- The 6-step decision tree (reorder, merge, split, add, remove steps)
- The examples and edge cases
- The strictness/leniency balance
- The pick vs pass criteria
- How titles are instructed
- How follow-up questions are instructed
- The emotional framing

Also change the `FIRST_TURN_NUDGE` and `SLATE_INSTRUCTION` if you think they can be improved.

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
4. Modify the prompt
5. Run again
6. Compare to previous best. If better, keep. If worse, revert.
7. Log what you tried and what happened in research_summary.md
8. Repeat

**NEVER STOP**: Once started, do NOT pause to ask the human. Keep iterating until manually interrupted. The human might be asleep. If you run out of obvious ideas, think harder — try radical changes, combinations of previous near-misses, or entirely new framings. Try things nobody would think of.

**Think like a researcher, not an engineer.** Don't just tweak parameters. Ask WHY the prompt works the way it does. Challenge assumptions. What if the 6-step tree is overcomplicated? What if the persona matters more than the rules? What if the examples are teaching the wrong thing? What if less instruction produces better results?
