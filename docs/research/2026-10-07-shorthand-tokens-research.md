# Can shorthand reduce LLM token counts? — research findings

*Spike, 2026-10-07. Question: does writing prompts or asking for replies in shorthand (dropped vowels, abbreviations, symbols, telegraphic style) cut billed tokens, and at what cost?*

## Bottom line

| Technique | Input side | Output side |
|---|---|---|
| **True shorthand** (Pitman/Teeline-style vowel dropping, phonetic respelling) | **Backfires: +60–80% tokens** | **Backfires: longer *and* less accurate** |
| **Ad-hoc abbreviations** (b/c, w/, u, thru, info) | Neutral to slightly worse (−2% to +13%) | Not worth it |
| **Telegraphic / "caveman"** (drop articles, filler, connectives) | 20–33% fewer *prompt* tokens, but models **answer longer**, so **net cost goes up** and accuracy drops | **Works: 1.4–2.4× cheaper per item** on Claude/GPT, often no accuracy loss on short-answer tasks |
| **Structured symbolic rewrite** (Telegraph English, MetaGlyph) | 40–80% fewer tokens with ~1–3pp accuracy loss on frontier models — but needs an LLM pass to produce it, and Claude Haiku 4.5 scored 0–26% on bare symbol comprehension | n/a |
| **Learned compression** (LLMLingua-2, gist tokens) | 2–5× (LLMLingua-2) / up to 26× (gist, needs fine-tuning) with ~90%+ accuracy retained | n/a |

**Recommendation:** Don't use shorthand on the input. Do use a terse-output instruction on the output, where it actually pays. For input, the only things that work are removing *content* (filler sentences, redundant context, prompt caching) or a proper compressor like LLMLingua-2, not respelling words.

---

## 1. Why true shorthand backfires: tokenizers, not words

Shorthand saves *pen strokes*. LLMs are billed in *BPE tokens*, and BPE vocabularies are built from normal spelling. A common English word is one token; a mangled one splits into several.

Measured on the same four passages (prose, instruction, technical, code) with three tokenizers — OpenAI cl100k, OpenAI o200k, and a Claude-2-era proxy (`Xenova/claude-tokenizer`). Claude 4.7+/Fable models use a newer tokenizer that Anthropic doesn't publish; absolute counts differ (~30% higher) but the *direction* is a property of BPE and held identically across all three vocabularies tested.

Token change vs. original (prose / instruction / technical / code, o200k; Claude-proxy in brackets):

| Style | Prose | Instruction | Technical | Code |
|---|---|---|---|---|
| Caveman (drop filler words) | **−33%** [−32%] | **−20%** [−20%] | **−24%** [−24%] | −7% [−6%] |
| Symbolic (∵ → & ¬ + caveman) | −15% [−17%] | −16% [−14%] | −19% [−15%] | −7% [−6%] |
| Caveman + abbreviations | −13% [−15%] | −20% [−18%] | −20% [−20%] | −7% [−6%] |
| Abbreviations only (b/c, w/, info…) | **+13%** [+11%] | −2% [0%] | +4% [+4%] | 0% [0%] |
| Remove spaces | +28% [+26%] | 0% [+4%] | +11% [+22%] | −10% [−2%] |
| Teeline-ish (drop inner vowels) | **+67%** [+64%] | **+70%** [+78%] | **+78%** [+82%] | +44% [+21%] |
| Drop all non-initial vowels | +59% [+57%] | +68% [+75%] | +76% [+78%] | +32% [+19%] |

Per-word examples (o200k): `because`=1 token, `b/c`=2, `bcs`=2. `information`=1, `info`=1, `infrmtn`=3. `connection`=1, `conn`=1, `cnnctn`=3. `database`=1, `db`=1, `dtbs`=2.

Takeaways:
- **Dropping vowels is the single worst thing you can do** — fewer characters, ~1.7× the tokens.
- Abbreviations only help when the abbreviation is itself a common token (`info`, `db`, `conn`, `pls`, `thru`). Slash-forms (`b/c`, `w/o`) cost *more* than the full word.
- The only input-side win in this table is deleting whole words (caveman). That's not shorthand; it's editing.

Script: [shorthand_token_test.py](shorthand_token_test.py) (throwaway).

## 2. Input side: compressing the prompt

### CAVEWOMAN (Adobe Research, June 2026) — the direct test
Eight models (incl. Claude Haiku 4.5, Claude Sonnet 4.6, GPT-4o, GPT-5.4), five benchmarks, five compression levels, measured on *realized cost* not just prompt tokens.

- Input compression is "**a strict lose-lose**": prompt tokens fall ~15%, but models compensate with longer answers. Net cost: Claude Haiku 4.5 **+3.1%**, Claude Sonnet 4.6 **+5.1%**, GPT-5.4 **+15.4%**; up to **1.8×** on the worst dataset and **2.7×** at deeper compression as accuracy collapses.
- Accuracy under input compression degrades at every level beyond light trimming, and on reasoning-style tasks (GSM8K) it collapses quickly.

### Telegraph English (May 2026)
An LLM rewrites prose into fact-lines with ~40 symbols (`ML→MEDICAL-DIAGNOSTICS: EARLY-DETECTION+27.5% ∧ FALSE-POSITIVE−12% [JOHNSON:2023]`).
- Mean **41.5%** token reduction; key-fact accuracy 99.1% (GPT-4.1), 95.7% (GPT-4o-mini) vs. 100%/99.1% uncompressed.
- Beats LLMLingua-2 at matched ratios, but **needs a model call per chunk** to produce — only pays for context you'll re-read many times (RAG corpora, cached knowledge), not one-shot prompts.
- Failures cluster on dates, unit qualifiers, conditional boundaries ("30 calendar days" → `30D` lost the calendar/business distinction).
- Tested on OpenAI models only.

### MetaGlyph (Jan 2026)
Encodes instructions as math symbols (∈, ⇒, ¬, ∘) with no decoding legend. 62–81% token reduction — but comprehension is wildly model-dependent: GPT-5.2 91% on ∈, **Claude Haiku 4.5 26% on ∈ and 0% on ⇒**. Not usable on Claude without a legend, and the legend eats the savings.

### LLMLingua-2 (Microsoft, 2024)
Trained token classifier drops low-information tokens. **2–5×** compression, >90% of baseline accuracy on GSM8K at 3×; 46–50% context reduction in RAG with ~0.02 BERTScore loss. This is the real "compress the input" tool. Cross-lingual audits (2026) show it degrades more on non-English.

### Gist tokens (NeurIPS 2023)
Up to 26× compression of *repeated* prompts — but requires fine-tuning the model; not applicable to API models. Prompt caching is the API-world equivalent and is what Anthropic actually offers.

## 3. Output side: making the model answer short

This is where the technique works, because output tokens are 4–8× pricier than input and models comply with terse instructions.

### CAVEWOMAN
Output compression cuts realized per-item cost **1.4–2.4× per model (up to 3×)** on GPT-4o, Claude Haiku 4.5, Claude Sonnet 4.6, cheaper on all 15 of their benchmark cells. Claude models were among the *most* robust to output compression (output/input accuracy ratio 2.6–2.7). Caveats: biggest wins on short-answer tasks (yes/no, MCQ); ~52% of correct answers no longer textually entail the model's unconstrained reasoning — you lose the "why", not the "what". GPT-5.4 didn't benefit because hidden reasoning tokens dominate its bill.

### "How well do LLMs compress their own chain-of-thought" (Columbia, 2025)
31 output-style prompts on GPT-4o, Claude 3.5 Sonnet, Llama 3.3. Key result: **accuracy is a function of output length, not of style** — bullets, no-spaces, Chinese, no-grammar all sit on one tradeoff curve.

MMLU-Pro Math, GPT-4o (accuracy / avg tokens): DefaultCoT 80.8% / 585 · BeConcise 80.4% / 415 · BulletPoints 75.0% / 185 · NoProperGrammar 78.8% / 189 · **AbbreviateWords 74.2% / 275** · NoSpaces 78.8% / 248.

"Abbreviate words as much as possible" (the closest thing to shorthand) was the **worst of all strategies**: lower accuracy than bullets *and* more tokens than bullets or no-grammar — because `Slvng fr rdr f Z18` tokenizes badly. Same pattern on GPT-4o-mini (56.6% / 259).

### Caveman (Claude Code skill), JetBrains measurement
Marketed as −65% output tokens. JetBrains ran 86 real SWE tasks in Claude Code: **−8.5% output tokens**, no measurable change in success rate, code quality, or time. Independent runs: 12–23%. Why so small: in agentic coding, output text is 0.6–2.5% of billed tokens; file reads, tool results, and generated code dominate. Ceiling ≈ 0.4–3% of the bill.

## 4. Where it breaks

- **Code, identifiers, numbers, dates, units** — abbreviation destroys exact tokens the model must reproduce; Telegraph English's errors clustered exactly here.
- **Instructions with conditionals / negation** — "do not include X unless Y" survives caveman poorly; MetaGlyph shows Claude doesn't reliably read `¬`/`⇒` bare.
- **Reasoning tasks** — any output compression that shortens the chain of thought below the task's "token complexity" fails hard (sharp threshold, not graceful).
- **Agentic sessions** — the prose you'd compress is a rounding error next to tool output and context.
- **Audit trail** — terse answers are correct but unexplained; fine for yes/no, bad when you need the reasoning.

## 5. What's actually worth doing, ranked

1. **Prompt caching** for any repeated prefix (system prompt, docs, tool schemas) — up to 90% off cached input. Nothing else comes close.
2. **Output-style instruction**: "Answer in ≤N words / bullet points / final answer only" for short-answer or classification tasks. 1.4–2.4× cheaper per CAVEWOMAN; keep CoT for reasoning tasks.
3. **Edit the input** — delete redundant context, examples, and boilerplate sentences. Cutting whole sentences is the only input compression that's free. Measure with `/v1/messages/count_tokens` against the exact model ID (Claude 4.7+/Fable tokenizers count ~30% higher than earlier ones; don't reuse old counts).
4. **LLMLingua-2** on long retrieved context (RAG, logs, transcripts) if volume justifies the extra model pass. 2–3× with modest loss.
5. **Don't**: drop vowels, phonetic respelling, slash-abbreviations, remove spaces, bare symbolic operators on Claude. All measured as neutral-to-much-worse.

## Sources

- CAVEWOMAN: How LLMs Behave Under Linguistic Input and Output Compression — https://arxiv.org/abs/2606.24083
- How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach — https://arxiv.org/abs/2503.01141
- Telegraph English: Semantic Prompt Compression via Structured Symbolic Rewriting — https://arxiv.org/abs/2605.04426
- Semantic Compression of LLM Instructions via Symbolic Metalanguages (MetaGlyph) — https://arxiv.org/abs/2601.07354
- Learning to Compress Prompts with Gist Tokens — https://arxiv.org/abs/2304.08467
- LLMLingua-2 overview — https://bdtechtalks.com/2024/04/01/llmlingua-2-prompt-compression/
- Lost in Compression: cross-lingual audit of extractive compressors — https://arxiv.org/abs/2608.26175
- JetBrains caveman measurement (InfoWorld) — https://www.infoworld.com/article/4193775/talk-like-a-caveman-prompts-save-tokens-but-far-less-than-promised.html
- Caveman cost ceiling analysis — https://www.implicator.ai/caveman-claude-code-skill-cuts-output-20-your-bill-barely-notices-2/
- Anthropic token counting docs (tokenizer change note, count_tokens endpoint) — https://platform.claude.com/docs/en/build-with-claude/token-counting
- Claude vs GPT tokenizer efficiency — https://getcoai.com/news/claude-models-up-to-30-pricier-than-gpt-due-to-hidden-token-costs

---

# Part 2 — Competitive landscape (2026-10-07)

## Input-side compression: crowded, with a dominant OSS leader

| Product | What it does | Traction / model | Output side? |
|---|---|---|---|
| **Headroom** (Headroom Labs, ex-Netflix) | Compresses tool output (JSON SmartCrusher 60–95%), code (AST-aware, ~20%), prose (own HF model), images, history. Library / proxy / `headroom wrap <agent>` / MCP / LangChain·LiteLLM·Vercel adapters. Reversible via cache. Integrated into Portkey gateway. | **74.5k★**, Apache-2.0, commercial team tier (SSO, dashboards, VPC) | Yes, but thin: static "verbosity steering" string + effort routing |
| **RTK** (Rust Token Killer) | CLI proxy filtering dev-command output 60–90% for Claude Code / Cursor / Copilot / Gemini CLI | **51k★**, OSS | No |
| **Compresr** (YC W26, 4 EPFL PhDs) | Intent-conditioned compression API + OSS Go "Context Gateway"; `expand()` handle lets model fetch originals; lazy tool loading. LiteLLM guardrail. | ~$0.50/M tokens compressed (unofficial) | No |
| **The Token Company** (YC W26) | Fast non-LLM classifier drops redundant prompt tokens; claims +2.7pp accuracy, −20% tokens, 100k tok <100ms | Drop-in API, pricing unpublished | No |
| **LLMLingua / -2** (Microsoft) | Research-grade token classifier, 2–20×; LangChain/LlamaIndex integration | OSS, MIT | No |
| **Gateways** (Kong AI Prompt Compressor, LiteLLM Compresr guardrail, Edgee, OmniRoute, Portkey+Headroom) | Compression as a checkbox feature inside routing/observability | Bundled | No |
| **Claude Code plugins** (context-mode 98% claim, Token Savior 77%, lean-ctx, llmtrim, snip, Tokenade, TOON) | Keep tool output out of the window; symbol-level reads | Hundreds to low-thousands ★ each | No |
| **Provider-native** (Anthropic context editing / compaction, Claude Code micro/auto/reactive compact, prompt caching 90% off) | Free, built in; sets the floor any third party must beat | — | No |

## Output-side: nearly empty
- **Caveman** skill — a system-prompt string; measured −8.5% output tokens on real coding tasks (JetBrains), ≈0.4–3% of bill.
- **lean-ctx issue #1125 "Response Shaping"** — proposed proxy-layer response rewriting; not shipped.
- **NeuralTrust TrustGate** — gateway policy for max output length (blunt cap, no quality loop).
- **Headroom verbosity steering** — one static sentence appended to system prompt.
- **Research** (CAVEWOMAN, token-complexity paper, YapBench): output compression is the lever that actually cuts realized cost 1.4–2.4×; *adaptive* budgets (short for easy items, full CoT for hard) are explicitly identified as the under-exploited gap; most verbose models produce 10–20× more than needed on simple tasks.

## Where shorthand itself sits
No product uses true shorthand, and the research explains why: it inflates BPE tokens 60–80%. Symbolic rewrite (Telegraph English, MetaGlyph) exists as papers only, needs an LLM pass, and Claude reads bare symbols poorly. It is not a viable wedge.

## Positioning options

**A. Output-side cost optimizer (recommended if standalone).** Middleware that learns a per-route output budget and format, injects the constraint, shadow-evaluates accuracy against the unconstrained baseline, and auto-tunes. Targets API product workloads (classification, extraction, support bots, summarization) where output is 4–8× the input price and a large share of spend — not coding agents, where output is ~2% of the bill. Nobody owns this; incumbents treat it as a static string. Honest ceiling: 1.4–2.4× on short-answer routes, ~0 on long-form generation.

**B. Build it inside Headroom.** If the local `headroom` checkout is a fork you contribute to, ship the adaptive output-budget module there. Zero distribution problem (74k★, Portkey/LiteLLM integrated), and it's the one channel they haven't built. Positioning becomes "Headroom's output channel".

**C. Claude Code plugin.** Don't. Crowded (caveman, context-mode, RTK, Token Savior, lean-ctx), and the measured ceiling is ~3% of the bill.

**D. Shorthand / symbolic input compressor.** Don't. Measured net-negative.

Sources (Part 2): https://github.com/chopratejas/headroom · https://github.com/pleasedodisturb/awesome-llm-token-optimization · https://www.ycombinator.com/launches/Pb3-the-token-company-intelligent-compression-for-llm-context-bloat · https://www.startuphub.ai/ai-news/claudes-corner/2026/claudes-corner-compresr-yc-w2026 · https://docs.litellm.ai/docs/proxy/guardrails/compresr · https://portkey.ai/docs/aigw/integrations/guardrails/headroom.md · https://developer.konghq.com/ai-gateway/policies/ai-prompt-compressor/ · https://github.com/yvgude/lean-ctx/issues/1125 · https://neuraltrust.ai/blog/output-length-control · https://blog.jetbrains.com/ai/2026/07/speak-to-ai-agents-like-cavemen-tosave-tokens/ · https://computingforgeeks.com/reduce-claude-code-token-usage-tools/ · https://milvus.io/blog/claude-code-context-management-tools.md

---

# Part 3 — Positioning (chosen: Option A, standalone output-side optimizer)

**Statement.** The output-token control plane: drop-in middleware that learns, per route, the shortest output that still passes the customer's quality bar, enforces it through the provider's own knobs (`effort`, `verbosity`, `max_tokens`, output format, cache-safe terse system messages), and proves it with continuous shadow evaluation.

**Why this and not the knobs themselves.** Claude's `output_config.effort` (low→max), `max_tokens`, task budgets and mid-conversation system messages, and OpenAI's `verbosity` / `reasoning_effort`, are static per-request choices made blind. Research (SelfBudgeter 2025, TALE-EP, token-complexity 2025, YapBench 2026) shows adaptive per-request budgets dominate any fixed style and that "be concise" prompts sit far from the optimal frontier. The product is the policy + measurement loop, so it strengthens as providers add knobs rather than being absorbed.

**Economics.** Output is 5× input price on every current Claude model (Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 4.5 $1/$5, Fable 5.1 $10/$50); industry median 4×, up to 8×. Output share of spend: ~⅓ on a typical support-ticket mix (3,150 in / 400 out), 30–50% on chat/classification, ~2% on coding agents.

**ICP.** API product teams on Claude/OpenAI with high-volume short-answer routes (classification, extraction, support chat, summarization), ≥$10k/mo spend. Not coding agents.

**Vs. incumbents.** Headroom / Compresr / The Token Company / LLMLingua compress input; we optimize output. Complementary — distribution via LiteLLM guardrail / Portkey / Headroom plugin is an option.

**Ceiling, stated per route.** 1.4–2.4× on short-answer routes (CAVEWOMAN); ~0 on long-form generation. The product must report "no savings available" on routes where that's true.

Sources (Part 3): https://openrouter.ai/state-of-ai · https://www.silicondata.com/blog/llm-cost-per-token · https://blog.promptlayer.com/gpt-5-api-features/ · https://arxiv.org/abs/2505.11274 (SelfBudgeter) · https://arxiv.org/abs/2601.00624 (YapBench) · https://neuraltrust.ai/blog/output-length-control · Claude pricing/effort from the Anthropic pricing docs (2026-09)
