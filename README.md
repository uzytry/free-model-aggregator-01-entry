# Free-model aggregator — free-model-aggregator-01 entry

One server (`main.ts`, Node 22, standard library only) that serves the OpenAI
Chat Completions API and the Anthropic Messages API over ten simulated
free-tier providers, in JSON and SSE. It reads the brief (providers, ratings,
workload) out of `prompt.txt`, probes every provider's own list route in
parallel (falling back to the ratings table for any failed probe), registers
that catalog once, and routes each request to the best free model that can
serve it — greedy by `rating_milli`, requirements-checked, JSON-mode never to
Anthropic-wire providers.

Failover: one same-provider retry on a transient refusal; 400/401 never
retried; a refusal whose message says the budget is spent is permanent — the
provider is dropped from all later candidate lists. All upstream calls are
non-streamed; the server re-emits JSON or SSE per the client's request, with
exact echo preservation (system/user/tools/max_tokens/stops/image/JSON-mode)
and provider usage passed through verbatim. It polls `/work` and exits 0 when
the env says the workload is done.

## Practice results (kit's real env server + real scorer)

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Upstream calls |
|---|---|---|---|---|---|---|---|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |

99 is the theoretical ceiling: quality 100 would need every request at its best
rating, but 17 requests want openrouter's 1475 model and its budget is only
10–12 calls. All fallbacks landed on the next-best 1470 — optimal routing.
Two runs of trial 0 produce byte-identical exports (deterministic: no wall
clock, no randomness).

## Run it

```
node main.ts
```

with `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` set by the trial runner.
`node main.ts --list` prints the probed catalog as JSON.

## License

MIT — see `LICENSE`.
