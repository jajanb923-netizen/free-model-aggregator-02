# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/jajanb923-netizen/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/jajanb923-netizen/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and streaming) and answers every prompt with a real completion from a live free OpenRouter model, dialled through the grader's gateway. It ranks the fourteen free models by the house's quality rating, serves each prompt with the best model that meets its requirements (JSON mode, tools, vision), retries a refused model twice before failing over to the next, and reports every refusal it recovered from in the `x-makefun-upstream-failures` header. `node main.ts --list` prints the catalog of the fourteen free models it can serve.

## Run it

```sh
node main.ts
```

Inside a trial the harness sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `MAKEFUN_TRIAL_SEED`, `OPENAI_BASE_URL` and the trial's dummy keys, then runs `node main.ts`. The server picks a free port, registers it with the env, serves the two wires, and exits 0 when the env says the work is done. Its completions are real calls to OpenRouter's free models: an entry needs an OpenRouter API key (from an account with at least 10 credits purchased) pushed to the competition before submitting, or every model call is refused.

---

<sub>Entered on [make.fun](https://make.fun/?ref=vttfhgk4&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
