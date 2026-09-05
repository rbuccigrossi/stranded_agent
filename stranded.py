#!/usr/bin/env python3
"""STRANDed Agent: a minimal, inspectable shell agent built on the AWS Strands SDK.

Talks to any OpenAI-compatible Chat Completions endpoint. Each model in config.json
names its own endpoint, so a plain OpenAI key and a gateway such as APISIX -- which
often exposes a separate route per model -- are the same code path.
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from strands import Agent, tool
from strands.interventions import Confirm, Deny, InterventionHandler, Proceed
from strands.models.openai import OpenAIModel
from strands.types.agent import Limits
from strands.vended_plugins.skills import AgentSkills

import stranded_tools

try:
    import readline
except ImportError:  # pragma: no cover - unavailable on some platforms
    readline = None


ROOT_DIR = Path(__file__).resolve().parent
SKILLS_DIR = ROOT_DIR / "skills"
BUILTIN_SKILLS_DIR = SKILLS_DIR / "_builtin"
TOOLS_DIR = ROOT_DIR / "tools"
SESSIONS_FILE = Path(os.getenv("STRANDED_SESSIONS", str(ROOT_DIR / ".stranded_sessions.json")))
DEFAULT_CONFIG = ROOT_DIR / "config.json"
EXAMPLE_CONFIG = ROOT_DIR / "config.json.example"


def default_config_path() -> Path:
    """Where to read the catalogue from.

    ``config.json`` is yours and is not in version control, so a fresh clone has
    only the committed example; fall back to it rather than failing to start. An
    explicit ``STRANDED_CONFIG`` always wins, and is an error if it is missing.
    """
    explicit = os.getenv("STRANDED_CONFIG")
    if explicit:
        return Path(explicit)
    return DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else EXAMPLE_CONFIG


CONFIG_FILE = default_config_path()

DEFAULT_APPROVAL = "ask"
APPROVAL_MODES = ("ask", "all", "deny")
MAX_STEPS = int(os.getenv("STRANDED_MAX_STEPS", "200"))
MAX_GOAL_ITERATIONS = int(os.getenv("STRANDED_MAX_GOAL_ITERATIONS", "5"))

#: Tools that reach outside the conversation and therefore follow the approval mode.
APPROVAL_TOOL_NAMES = {"execute_shell", "web_search", "web_fetch"}

GOAL_PROMPT = (
    "Continue working toward the active goal. Take the next necessary action; "
    "do not stop merely to report progress. Use tools or mark the goal complete "
    "or blocked when appropriate."
)

_TTY = sys.stderr.isatty()

# Reasoning blocks are dropped when history is replayed, and the provider says so on
# every request. That is expected here, so keep it out of the transcript.
logging.getLogger("strands.models.openai").setLevel(logging.ERROR)


def ensure_environment() -> None:
    """Create project-owned extension directories and expose tools to commands."""
    SKILLS_DIR.mkdir(exist_ok=True)
    TOOLS_DIR.mkdir(exist_ok=True)
    tool_path = str(TOOLS_DIR)
    if tool_path not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = tool_path + os.pathsep + os.environ.get("PATH", "")


ensure_environment()

if readline:
    readline.parse_and_bind("\\C-l: clear-screen")


def color(code: int, value: str) -> str:
    return f"\033[{code}m{value}\033[0m" if _TTY else value


def catalog() -> Dict[str, Any]:
    """The models on offer, read from config.json.

    Both interfaces read this, so the terminal and the browser always present the
    same choices, and adding a model never touches the code.
    """
    with CONFIG_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def models() -> List[Dict[str, Any]]:
    return catalog()["models"]


def model_entry(name: str) -> Dict[str, Any]:
    """One model's id, endpoint, and reasoning levels, looked up by its display name."""
    offered = models()
    for entry in offered:
        if entry["name"] == name:
            return entry
    names = ", ".join(item["name"] for item in offered)
    raise ValueError(f"unknown model {name!r}; config.json offers: {names}")


@dataclass(frozen=True)
class AgentConfig:
    """Everything the user can change about a run, validated against config.json.

    Empty fields fall back to the catalogue's default, so ``AgentConfig()`` is
    always the configured default.
    """

    model: str = ""
    reasoning: Optional[str] = None
    approval_mode: str = DEFAULT_APPROVAL
    max_steps: int = MAX_STEPS

    def __post_init__(self) -> None:
        data = catalog()
        object.__setattr__(self, "model",
                           self.model or data.get("default_model") or data["models"][0]["name"])
        entry = model_entry(self.model)
        levels = entry.get("reasoning") or []
        chosen = self.reasoning or entry.get("default_reasoning") or (levels[0] if levels else None)
        if levels and chosen not in levels:
            raise ValueError(f"{self.model} accepts reasoning: {', '.join(levels)}")
        object.__setattr__(self, "reasoning", chosen if levels else None)
        object.__setattr__(self, "approval_mode", self.approval_mode.strip().lower())
        if self.approval_mode not in APPROVAL_MODES:
            raise ValueError(f"approval must be one of: {', '.join(APPROVAL_MODES)}")


def env_config(**overrides: Any) -> AgentConfig:
    """Build a configuration from the environment, with non-empty overrides on top."""
    settings: Dict[str, Any] = {
        "model": os.getenv("STRANDED_MODEL", ""),
        "reasoning": os.getenv("STRANDED_REASONING"),
        "approval_mode": os.getenv("STRANDED_APPROVAL", DEFAULT_APPROVAL),
    }
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return AgentConfig(**settings)


@dataclass
class Turn:
    """The outcome of one call into the agent."""

    status: str  # complete | approval_required | stopped | error
    answer: str = ""
    reasoning: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    approval: Optional[Dict[str, str]] = None
    error: Optional[str] = None


def api_key() -> str:
    """The bearer token. Behind a gateway this is the gateway's own credential."""
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
    if not key:
        raise RuntimeError("set OPENAI_API_KEY (or OPEN_API_KEY)")
    return key


def system_message() -> str:
    docs = ", ".join(sorted(path.name for path in ROOT_DIR.glob("*.md"))) or "none"
    return f"""You are the STRANDed Agent, a general-purpose agent with shell, planning, goal, and public web tools.
Use the tools to inspect, edit, install, test, search, automate, and answer.
Be concise, tenacious, and relentlessly useful. Keep taking shell steps until done or blocked.
Output short plain-text snippets optimized for terminal reading; no markdown rendering or syntax highlighting.
Never run destructive commands unless explicitly requested.
For multi-step work, create and maintain a visible plan. Echo plan updates as you proceed.
Create a goal only for explicit ongoing objectives or bounded iteration; continue active goals until they are complete, blocked, or bounded by the goal/framework limit.
Use web_search and web_fetch for current public information, and cite the source URLs you use.
project root: {ROOT_DIR}
cwd: {os.getcwd()}
platform: {platform.platform()}
python: {sys.version.split()[0]}
shell: {os.getenv('SHELL', '')}
Important docs (read as needed): {docs}
Project tools directory: {TOOLS_DIR}
"""


@tool
def execute_shell(command: str, description: str, cwd: Optional[str] = None,
                  timeout: int = 60, env: Optional[Dict[str, str]] = None) -> str:
    """Run a shell command with inherited environment.

    Args:
        command: The command to run.
        description: Why this command is useful right now, in 5-10 words.
        cwd: Directory to run the command in; defaults to the current directory.
        timeout: Seconds to wait before killing the command.
        env: Extra environment variables for this command.
    """
    if not 5 <= len(description.split()) <= 10:
        return "bad arguments: description must be 5-10 words"
    run_env = {**os.environ, **(env or {})}
    tool_path = str(TOOLS_DIR)
    if tool_path not in run_env.get("PATH", "").split(os.pathsep):
        run_env["PATH"] = tool_path + os.pathsep + run_env.get("PATH", "")
    try:
        process = subprocess.run(
            command, shell=True, cwd=os.path.abspath(cwd or os.getcwd()), env=run_env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
        )
        return f"$ {command}\nexit {process.returncode}\n{process.stdout}"[-12000:]
    except subprocess.TimeoutExpired as error:
        return f"$ {command}\ntimeout after {timeout}s\n{error.stdout or ''}"[-12000:]


def tool_detail(name: str, args: Dict[str, Any]) -> str:
    """One line describing a tool call, used for approval prompts and logs."""
    if name == "execute_shell":
        return f"execute_shell: $ {args.get('command', '')}"
    if name == "web_search":
        return f"web_search: query: {args.get('query', '')}"
    if name == "web_fetch":
        return f"web_fetch: url: {args.get('url', '')}"
    return name


#: Tool name -> the key to read out of its JSON result, and the event to raise for it.
RESULT_EVENTS = {"create_plan": ("plan", "plan_update"), "update_plan": ("plan", "plan_update"),
                 "create_goal": ("goal", "goal_update"), "get_goal": ("goal", "goal_update"),
                 "update_goal": ("goal", "goal_update"), "web_search": ("sources", "web_sources"),
                 "web_fetch": ("sources", "web_sources")}


def tool_events(name: str, result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn a finished tool result into the plan, goal, and source events the UIs show."""
    if name not in RESULT_EVENTS or not result:
        return []
    try:
        payload = json.loads("".join(block.get("text", "") for block in result.get("content", [])))
    except (json.JSONDecodeError, AttributeError):
        return []
    key, event_type = RESULT_EVENTS[name]
    return [{"type": event_type, key: payload.get(key), "explanation": payload.get("explanation", "")}]


class Approval(InterventionHandler):
    """Gate side-effecting tools and report tool activity to the caller.

    ``ask`` lets an interactive caller answer in-process; leaving it unset makes
    the agent pause with an interrupt that an external UI resolves later.
    """

    name = "approval"

    def __init__(self, config: AgentConfig,
                 on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                 ask: Optional[Callable[[str, Dict[str, Any]], bool]] = None) -> None:
        self.config = config
        self.on_event = on_event
        self.ask = ask

    def _emit(self, event: Dict[str, Any]) -> None:
        if self.on_event:
            self.on_event(event)

    def before_tool_call(self, event: Any, **kwargs: Any) -> Any:
        name, args = event.tool_use["name"], event.tool_use["input"]
        self._emit({"type": "tool_call", "name": name, "arguments": args})
        if name not in APPROVAL_TOOL_NAMES or self.config.approval_mode == "all":
            return Proceed()
        if self.config.approval_mode == "deny":
            return Deny(reason="denied by user")
        detail = tool_detail(name, args)
        if self.ask:
            return Confirm(prompt=detail, response=self.ask(name, args))
        return Confirm(prompt=detail)

    def after_tool_call(self, event: Any, **kwargs: Any) -> Any:
        for derived in tool_events(event.tool_use["name"], event.result):
            self._emit(derived)
        return Proceed()


def build_model(config: AgentConfig) -> OpenAIModel:
    """Point the Chat Completions client at the endpoint this model declares.

    Each model carries its own ``base_url``, because a gateway commonly exposes one
    route per model; omit it to use the OpenAI client's default. A model with no
    reasoning levels is sent no ``reasoning_effort`` at all, which is what models
    that reject the argument need.
    """
    entry = model_entry(config.model)
    client_args: Dict[str, Any] = {"api_key": api_key()}
    if entry.get("base_url"):
        client_args["base_url"] = entry["base_url"]
    return OpenAIModel(client_args=client_args, model_id=entry["id"],
                       params={"reasoning_effort": config.reasoning} if config.reasoning else {})


def build_agent(config: AgentConfig, messages: Optional[List[Dict[str, Any]]] = None,
                state: Optional[Dict[str, Any]] = None,
                on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                ask: Optional[Callable[[str, Dict[str, Any]], bool]] = None) -> Agent:
    """Build an agent, resuming from a previously saved message history and state.

    The approval handler is kept on ``agent.approval`` because it is also this
    harness's event sink, and each caller streams to a different place.
    """
    approval = Approval(config, on_event, ask)
    state = dict(state or {})
    skills = state.get("agent_skills")
    if isinstance(skills, dict):
        # The skills plugin injects its catalogue into each fresh system prompt and
        # warns if a previous injection is missing, so keep the active skills but
        # drop the record of where the last copy went.
        state["agent_skills"] = {key: value for key, value in skills.items()
                                 if key != "last_injected_xml"}
    agent = Agent(
        model=build_model(config),
        system_prompt=system_message(),
        tools=[execute_shell, *stranded_tools.TOOLS],
        plugins=[AgentSkills(skills=[str(SKILLS_DIR), str(BUILTIN_SKILLS_DIR)])],
        interventions=[approval],
        messages=list(messages or []),
        state=state,
        callback_handler=None,
    )
    agent.approval = approval
    return agent


def _usage(result: Any) -> Dict[str, int]:
    usage = getattr(result.metrics, "accumulated_usage", {}) or {}
    return {"input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0)}


def _answer(result: Any) -> str:
    return "".join(block.get("text", "") for block in (result.message or {}).get("content", []))


def run_turn(agent: Agent, prompt: Any, config: AgentConfig,
             on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> Turn:
    """Stream one turn of the Strands agent loop and summarize how it ended."""
    agent.approval.on_event = on_event
    reasoning: List[str] = []

    async def stream() -> Any:
        result = None
        async for event in agent.stream_async(prompt, limits=Limits(turns=config.max_steps)):
            if "result" in event:
                result = event["result"]
            elif "reasoningText" in event:
                reasoning.append(event["reasoningText"])
                if on_event:
                    on_event({"type": "reasoning_delta", "text": event["reasoningText"]})
            elif "data" in event and on_event:
                on_event({"type": "text_delta", "text": event["data"]})
        return result

    try:
        result = asyncio.run(stream())
    except Exception as error:
        return Turn("error", error=f"{type(error).__name__}: {error}",
                    reasoning="".join(reasoning))
    usage = _usage(result)
    if on_event:
        on_event({"type": "usage", "usage": usage})
    if result.stop_reason == "interrupt" and result.interrupts:
        pending = result.interrupts[0]
        return Turn("approval_required", usage=usage, reasoning="".join(reasoning),
                    approval={"id": pending.id, "detail": str(pending.reason)})
    if result.stop_reason == "limit_turns":
        return Turn("stopped", "stopped: too many tool calls", "".join(reasoning), usage)
    return Turn("complete", _answer(result), "".join(reasoning), usage)


def run_agent(agent: Agent, prompt: Any, config: AgentConfig,
              on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> Turn:
    """Run a turn, then keep going while an explicit goal is still active."""
    for iteration in range(MAX_GOAL_ITERATIONS + 1):
        turn = run_turn(agent, prompt, config, on_event)
        if turn.status != "complete":
            return turn
        goal = agent.state.get("goal") or {}
        if goal.get("status") != "active":
            return turn
        limit = min(config.max_steps, int(goal.get("max_steps", MAX_GOAL_ITERATIONS)))
        if iteration >= limit:
            return Turn("stopped", "stopped: goal iteration limit reached",
                        turn.reasoning, turn.usage)
        if on_event:
            on_event({"type": "goal_iteration", "iteration": iteration + 1, "limit": limit})
        prompt = GOAL_PROMPT
    return turn


def resume_prompt(interrupt_id: str, approved: bool) -> List[Dict[str, Any]]:
    """Build the prompt that answers a pending approval interrupt."""
    return [{"interruptResponse": {"interruptId": interrupt_id, "response": bool(approved)}}]


def display_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Flatten Strands messages into the transcript entries the browser renders."""
    shown = []
    for message in messages:
        content = message.get("content", [])
        text = "".join(block["text"] for block in content if "text" in block)
        reasoning = "".join(block["reasoningContent"]["reasoningText"].get("text", "")
                            for block in content if "reasoningContent" in block)
        if text or reasoning:
            shown.append({"role": message.get("role", "assistant"), "content": text,
                          "reasoning": reasoning})
    return shown


def load_sessions() -> List[Dict[str, Any]]:
    try:
        with SESSIONS_FILE.open(encoding="utf-8") as handle:
            value = json.load(handle)
            return value if isinstance(value, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def save_session(agent: Agent, label: str, config: AgentConfig) -> None:
    """Persist the conversation as plain JSON so it can be read and resumed anywhere."""
    sessions = [session for session in load_sessions()
                if not (session.get("label") == label and session.get("cwd") == os.getcwd())]
    sessions.append({"label": label[:80], "cwd": os.getcwd(), "ts": int(time.time()),
                     "model": config.model, "reasoning": config.reasoning,
                     "messages": agent.messages, "state": agent.state.get()})
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SESSIONS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(sessions[-50:], handle, indent=2, default=str)


def pick_session() -> Dict[str, Any]:
    sessions = [session for session in load_sessions() if session.get("cwd") == os.getcwd()][-10:]
    if not sessions:
        raise SystemExit("no sessions in this directory")
    for index, session in enumerate(reversed(sessions)):
        age = int(time.time()) - session.get("ts", int(time.time()))
        label = f"{age // 60}m" if age < 3600 else f"{age // 3600}h" if age < 86400 else f"{age // 86400}d"
        print(f"  {color(90, str(index))}  {session.get('label', 'untitled')}  {color(90, label + ' ago')}")
    try:
        choice = input(f"{color(1, 'stranded')}{color(90, '#')} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)
    try:
        return sessions[-(int(choice) + 1)]
    except (ValueError, IndexError) as error:
        raise SystemExit("invalid session") from error


def terminal_approval(config: AgentConfig) -> Callable[[str, Dict[str, Any]], bool]:
    """Ask on the terminal, remembering an 'approve all' answer for the rest of the run."""
    state = {"mode": config.approval_mode}

    def ask(name: str, args: Dict[str, Any]) -> bool:
        print(f"\n{color(90, '# ' + name)}", file=sys.stderr)
        print(color(32, tool_detail(name, args)), file=sys.stderr)
        for key in ("description", "cwd", "timeout", "env"):
            if args.get(key) not in (None, "", {}):
                print(color(90, f"{key}: {args[key]}"), file=sys.stderr)
        if state["mode"] == "all":
            return True
        try:
            choice = input(f"Approve? {color(32, '[y] Approve')}  "
                           f"{color(33, '[a] Approve All')}  {color(31, '[n] Deny')}: ").strip().lower()
        except EOFError:
            return False
        if choice in ("a", "all"):
            state["mode"] = "all"
            return True
        return choice in ("y", "yes")

    return ask


_reasoning_open = False


def cli_event(event: Dict[str, Any]) -> None:
    """Render one harness event on the terminal.

    Reasoning, where a model emits it, arrives token by token; the label is printed
    once and the rest of the summary streams in after it.
    """
    global _reasoning_open
    kind = event["type"]
    if kind == "reasoning_delta":
        if not _reasoning_open:
            print(f"\n{color(90, '[reasoning] ')}", end="", file=sys.stderr, flush=True)
            _reasoning_open = True
        print(color(90, event["text"]), end="", file=sys.stderr, flush=True)
        return
    _reasoning_open = False
    if kind == "tool_call":
        print(f"\n{color(90, '[tool] ' + event['name'])}", file=sys.stderr)
        print(color(32, tool_detail(event["name"], event.get("arguments", {}))), file=sys.stderr)
    elif kind == "text_delta":
        print(event["text"], end="", flush=True)
    elif kind == "plan_update":
        print("\nplan:", file=sys.stderr)
        for step in event.get("plan") or []:
            marker = {"completed": "x", "in_progress": ">"}.get(step.get("status"), " ")
            print(f"  [{marker}] {step.get('step', '')}", file=sys.stderr)
    elif kind == "goal_update" and event.get("goal"):
        goal = event["goal"]
        print(f"\ngoal [{goal.get('status', 'unknown')}]: {goal.get('objective', '')}", file=sys.stderr)
    elif kind == "goal_iteration":
        print(f"\ngoal iteration {event['iteration']} of {event['limit']}", file=sys.stderr)
    elif kind == "web_sources":
        for source in event.get("sources") or []:
            print(f"  - {source.get('title', '')}: {source.get('url', '')}", file=sys.stderr)


def run_cli_turn(agent: Agent, prompt: str, config: AgentConfig) -> Turn:
    """Run one prompt in the terminal, where approvals are answered in-process."""
    turn = run_agent(agent, prompt, config, cli_event)
    print(f"\n{color(90, 'tokens: ' + str(turn.usage.get('total_tokens', 0)))}")
    return turn


def repl(agent: Agent, config: AgentConfig, label: Optional[str]) -> None:
    print(color(1, "stranded") + " repl " + color(90, "(:q quit)"))
    while True:
        try:
            prompt = input(color(36, "stranded > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt.lower() in (":q", "quit", "exit"):
            return
        turn = run_cli_turn(agent, prompt, config)
        if turn.status != "complete":
            print(turn.error or turn.answer or turn.status, file=sys.stderr)
            continue
        label = label or prompt
        save_session(agent, label, config)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                        help="continue the most recent session")
    parser.add_argument("-s", "--session", action="store_true", help="pick a saved session")
    parser.add_argument("--model", help="model name from config.json (env STRANDED_MODEL)")
    parser.add_argument("--reasoning", help="reasoning effort the model allows "
                                            "(env STRANDED_REASONING)")
    parser.add_argument("--approval", choices=APPROVAL_MODES,
                        help=f"tool approval mode (default {DEFAULT_APPROVAL}; "
                             "env STRANDED_APPROVAL)")
    parser.add_argument("--list", action="store_true", help="show config.json and exit")
    parser.add_argument("prompt", nargs="*")
    return parser.parse_args(argv)


def print_catalog() -> None:
    """Show what config.json offers, so --model and --reasoning can be typed correctly."""
    print(color(90, f"# {CONFIG_FILE}"))
    default = AgentConfig().model
    for entry in models():
        marker = "  (default)" if entry["name"] == default else ""
        levels = ", ".join(entry.get("reasoning") or []) or "no reasoning setting"
        print(f"{color(1, entry['name'])}{marker}")
        print(f"  {color(90, entry['id'])} at "
              f"{entry.get('base_url') or 'https://api.openai.com/v1'}")
        print(f"  reasoning: {levels}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    try:
        if args.list:
            print_catalog()
            return
        config = env_config(model=args.model, reasoning=args.reasoning,
                            approval_mode=args.approval)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{error}\n(reading {CONFIG_FILE}; "
                         f"copy {EXAMPLE_CONFIG.name} to start from the shipped catalogue)")
    session: Dict[str, Any] = {}
    if args.session or args.continue_session:
        session = pick_session() if args.session else next(
            (item for item in reversed(load_sessions()) if item.get("cwd") == os.getcwd()), {})
        if not session:
            raise SystemExit("no sessions in this directory")
        print(color(90, f"resuming: {session.get('label')}"))
    agent = build_agent(config, session.get("messages"), session.get("state"),
                        on_event=cli_event, ask=terminal_approval(config))
    prompt = " ".join(args.prompt)
    if not prompt:
        repl(agent, config, session.get("label"))
        return
    turn = run_cli_turn(agent, prompt, config)
    if turn.status != "complete":
        raise SystemExit(turn.error or turn.answer or turn.status)
    save_session(agent, session.get("label") or prompt, config)


if __name__ == "__main__":
    main()
