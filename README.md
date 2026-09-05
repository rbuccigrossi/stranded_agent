# STRANDed Agent

STRANDed is a small, single-user agent harness built on the
[AWS Strands Agents SDK](https://github.com/strands-agents/harness-sdk). The name is
the point: it is *stranded* on purpose, kept as thin as an agent can be while still
being genuinely useful, so the whole system stays easy to read in one sitting.

It speaks the OpenAI **Chat Completions** API, so plain OpenAI and an
OpenAI-compatible gateway such as APISIX are the same code path with a different
entry in `config.json` -- including a gateway that gives each model its own route.

Strands supplies the parts every agent needs — the agent loop, tool schemas, model
providers, streaming, skills, approval interventions, and turn limits. STRANDed adds
only what is specific to this project: a shell tool, planning and goal tools, public
web tools, a terminal interface, and a local browser interface.

STRANDed is designed as an individual developer sandbox (as opposed to a multi-user
production agent platform). It is intended to run inside a container or comparable
isolated development environment. The harness prioritizes transparency,
inspectability, and rapid development of skills and tools. Those skills and tools can
then be exported, reviewed, tested, and deployed into more hardened execution
environments with stronger authentication, policy enforcement, resource limits,
auditing, and isolation.

Security boundary: STRANDed assumes that the surrounding container or operating-system
environment provides the primary isolation boundary. Its tool approval is a developer
safeguard (not a substitute for sandboxing, authorization, or resource isolation). Do
not expose the local web server to untrusted networks or run it with sensitive host
access.

## Layout

- `stranded.py` — the terminal agent, and the harness the web UI calls into
- `stranded_web.py` — the local browser interface
- `stranded_tools.py` — planning, goal, search, and web extraction tools
- `config.json` — the models on offer, each with its endpoint and reasoning levels
  (yours, untracked; `config.json.example` is the committed starting point)
- `skills/` — project-local skills, one skill per subdirectory
- `skills/_builtin/` — framework-provided skill instructions
- `tools/` — project-local helper scripts added to the command path

## Install and run

```text
python -m venv .venv
.venv/Scripts/activate          # or: source .venv/bin/activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` (or `OPEN_API_KEY`) and run:

```text
python stranded.py
python stranded_web.py
```

Both run straight from a fresh clone. To change the models on offer, take your own
copy of the catalogue first — `config.json` is ignored by git, so your edits survive
a `git pull`:

```text
cp config.json.example config.json
```

Run the test suite from the project root:

```text
python -m unittest discover -s tests -v
```

The tests use a scripted Strands model provider, so they exercise the real agent loop,
tool executor, and approval path without a network call or an API key. They read
`config.json.example` rather than your `config.json`, so editing your own catalogue
cannot break them — one test does check that your `config.json` is loadable, when
you have one.

## Configuration

Models live in `config.json`, and **each model names its own endpoint**. Both
interfaces read the file, so the terminal and the browser always offer the same
choices, and adding a model never touches the code.

`config.json` is yours and is not in version control.
[`config.json.example`](config.json.example) is the committed copy, and STRANDed
reads it when you have no `config.json` — so a fresh clone runs without setup, and
your own catalogue is never clobbered by a pull. `STRANDED_CONFIG` overrides both,
and is an error if it points at a file that is not there.

```json
{
  "default_model": "GPT 5.6 Luna",
  "models": [
    {"name": "GPT 5.6 Luna", "id": "gpt-5.6-luna", "base_url": null,
     "reasoning": ["none", "low", "medium", "high", "xhigh"],
     "default_reasoning": "none"}
  ]
}
```

`python stranded.py --list` prints what is on offer, and names the file it read.
Omit `base_url` (or leave it `null`) to use the OpenAI client's default. A model
with an empty `reasoning` list
is sent no `reasoning_effort` at all, which is what models that reject the argument
need. Keys beginning with `_` are ignored, so `_note` is a place for comments.

| Flag | Environment variable | Values |
| --- | --- | --- |
| `--model` | `STRANDED_MODEL` | a model name from `config.json` |
| `--reasoning` | `STRANDED_REASONING` | a level that model allows |
| `--approval` | `STRANDED_APPROVAL` | `ask`, `all`, `deny` |
| `--list` | | print the catalogue and exit |
| | `STRANDED_CONFIG` | path to a different config.json |
| | `OPENAI_API_KEY` | bearer token; behind a gateway, the gateway's own credential |
| | `STRANDED_MAX_STEPS` | turns allowed in one invocation (default 200) |
| | `STRANDED_MAX_GOAL_ITERATIONS` | unprompted goal continuations (default 5) |

`-c` continues the most recent session in this directory and `-s` picks one from a
list.

### Gateways with one route per model

Endpoints belong to models rather than the other way round, because a gateway such
as APISIX commonly exposes a separate route for each model. Choosing the model
chooses the endpoint, and there is no second thing to pick:

```json
{
  "default_model": "Luna via APISIX",
  "models": [
    {"name": "Luna via APISIX", "id": "gpt-5.6-luna",
     "base_url": "https://gateway.example/llm/luna/v1",
     "reasoning": ["none", "low", "medium", "high", "xhigh"],
     "default_reasoning": "medium"},
    {"name": "Sol via APISIX", "id": "gpt-5.6-sol",
     "base_url": "https://gateway.example/llm/sol/v1",
     "reasoning": ["none", "low", "medium", "high", "xhigh"],
     "default_reasoning": "high"}
  ]
}
```

Models from different gateways, or a mix of gateway and direct models, can sit side
by side in the same list. Set `OPENAI_API_KEY` to the gateway's token. If the
gateway wants its own auth header rather than a bearer token, the OpenAI client
reads `OPENAI_CUSTOM_HEADERS` as newline-separated `Name: value` pairs:

```bash
export OPENAI_CUSTOM_HEADERS="apikey: your-consumer-key"
```

Each `base_url` must include whatever prefix `/chat/completions` gets appended to.

### What reasoning levels actually work

The shipped levels are what the **live API** reports, which is narrower than the
GPT-5.6 model pages. Those pages list `max`; the API refuses it:

```text
400 - Unsupported value: 'reasoning_effort' does not support 'max' with this model.
Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'.
```

`minimal` is refused the same way. Both Luna and Sol report the identical set.

There is a second, undocumented restriction that matters more here. On
`/v1/chat/completions`, **any effort above `none` is rejected while function tools
are present**:

```text
400 - Function tools with reasoning_effort are not supported for gpt-5.6-luna in
/v1/chat/completions. To use function tools, use /v1/responses or set
reasoning_effort to 'none'.
```

This agent always has tools, so against OpenAI's own endpoint `none` is currently
the only level that works — which is why it is the shipped `default_reasoning`. The
same levels may well work through a gateway that routes to `/v1/responses`, and
`config.json` is per-endpoint precisely so you can declare that where it is true.
Verify against your own endpoint rather than trusting either the model pages or this
paragraph.

## How it works

`build_agent` assembles one Strands `Agent` and everything else falls out of it:

- **Tools.** `execute_shell` and the tools in `stranded_tools.py` are plain Python
  functions carrying the `@tool` decorator. Strands derives the JSON schema the model
  sees from each function's type hints and docstring, so there are no hand-written
  tool schemas to keep in sync.
- **Skills.** `skills/` and `skills/_builtin/` are handed to Strands' `AgentSkills`
  plugin. It advertises each `SKILL.md`'s metadata in the system prompt and loads the
  full instructions on demand, so adding a skill is still just adding a directory.
- **Approval.** The `Approval` class is a Strands `InterventionHandler`. Approval mode
  `all` returns `Proceed`, `deny` returns `Deny`, and `ask` returns `Confirm`. In the
  terminal the confirmation is answered in-process; in the browser it becomes an
  interrupt that pauses the agent until you click Approve or Deny.
- **Events.** The same handler reports every tool call before it runs and turns each
  finished tool result into the plan, goal, and source updates the terminal and
  browser display.
- **State.** Plans and goals live in `agent.state`, and the conversation lives in
  `agent.messages`. Both are saved as plain JSON, so a session file is readable, and
  nothing about a conversation is held server-side by the model vendor.

Because history is local rather than vendor-held, pointing STRANDed at a different
endpoint is a config entry rather than a rewrite.

Shell execution, web searches, and web fetches follow the approval mode. Plans and
goals are local framework state and do not require approval. STRANDed reports tool
calls, reasoning summaries, plans, goals, and web sources as they occur.

## Lineage

STRANDed is the successor to Solith, which spoke to the OpenAI Responses API directly
with its own hand-rolled transport.
Moving to Strands deleted the hand-written JSON tool schemas, the server-sent-event
parser, the tool-call dispatch loop, the approval resume bookkeeping, the separate
plan and goal state file, and every other piece of provider-specific machinery —
about 250 lines, or a fifth of the project. Solith was in turn inspired by and derived
from [`pnegahdar/nano`](https://github.com/pnegahdar/nano) by Parham Negahdar.

## Screenshot

![Screenshot of the STRANDed Agent](/images/screenshot.png)

## License

STRANDed is released under the [MIT License](LICENSE.md). The license and original
copyright notice are retained in recognition of the upstream Nano project.

## Author

Robert Buccigrossi, Ph.D.  
CIO, TCG, Inc.  
CTO, SkepticCTO LLC
