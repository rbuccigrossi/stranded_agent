# STRANDed Agent

STRANDed is a small, single-user agent harness built on the
[AWS Strands Agents SDK](https://github.com/strands-agents/harness-sdk). The name is
the point: it is *stranded* on purpose, kept as thin as an agent can be while still
being genuinely useful, so the whole system stays easy to read in one sitting.

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

Run the test suite from the project root:

```text
python -m unittest discover -s tests -v
```

The tests use a scripted Strands model provider, so they exercise the real agent loop,
tool executor, and approval path without a network call or an API key.

## Configuration

The default model is `5.6 Luna` with `Light` reasoning, reached through Strands'
OpenAI Responses provider. Model, reasoning, and approval mode can be set from the
command line, the web interface, or the environment:

| Flag | Environment variable | Values |
| --- | --- | --- |
| `--model` | `STRANDED_MODEL` | any model id the provider accepts |
| `--provider` | `STRANDED_PROVIDER` | `openai`, `anthropic`, `bedrock`, `gemini`, `ollama` |
| `--reasoning` | `STRANDED_REASONING` | `Light`, `Medium`, `Heavy` |
| `--approval` | `STRANDED_APPROVAL` | `ask`, `all`, `deny` |
| | `STRANDED_MAX_STEPS` | turns allowed in one invocation (default 200) |
| | `STRANDED_MAX_GOAL_ITERATIONS` | unprompted goal continuations (default 5) |

`-c` continues the most recent session in this directory and `-s` picks one from a
list.

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

Because history is local rather than vendor-held, changing providers is a flag rather
than a rewrite — every entry in `PROVIDERS` satisfies the same Strands `Model`
interface.

Shell execution, web searches, and web fetches follow the approval mode. Plans and
goals are local framework state and do not require approval. STRANDed reports tool
calls, reasoning summaries, plans, goals, and web sources as they occur.

## Lineage

STRANDed is the successor to Solith, which spoke to the OpenAI Responses API directly.
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
