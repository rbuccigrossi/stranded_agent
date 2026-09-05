import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stranded
import stranded_tools
from tests.scripted_model import ScriptedModel, text_turn, tool_turn


@contextlib.contextmanager
def catalog_of(data):
    """Run the block against a config.json written just for this test."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(stranded, "CONFIG_FILE", path):
            yield


def scripted_agent(config: stranded.AgentConfig, *turns, **kwargs) -> stranded.Agent:
    """Build a real agent whose only stand-in is the model provider."""
    model = ScriptedModel(*turns)
    with patch.object(stranded, "build_model", return_value=model):
        agent = stranded.build_agent(config, **kwargs)
    agent.scripted_model = model
    return agent


class ConfigTests(unittest.TestCase):
    def test_defaults_come_from_the_catalog(self):
        config = stranded.AgentConfig()
        self.assertEqual(config.model, "GPT 5.6 Luna")
        self.assertEqual(config.reasoning, "none")
        self.assertEqual(config.approval_mode, "ask")

    def test_shipped_catalog_matches_the_live_api(self):
        entries = {entry["name"]: entry for entry in stranded.models()}
        self.assertEqual(sorted(entries), ["GPT 5.6 Luna", "GPT 5.6 Sol"])
        self.assertEqual(entries["GPT 5.6 Luna"]["id"], "gpt-5.6-luna")
        self.assertEqual(entries["GPT 5.6 Sol"]["id"], "gpt-5.6-sol")
        for entry in entries.values():
            self.assertEqual(entry["reasoning"], ["none", "low", "medium", "high", "xhigh"])

    def test_every_offered_level_is_accepted(self):
        for level in ["none", "low", "medium", "high", "xhigh"]:
            with self.subTest(level=level):
                config = stranded.AgentConfig(model="GPT 5.6 Sol", reasoning=level)
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    model = stranded.build_model(config)
                self.assertEqual(model.get_config()["params"], {"reasoning_effort": level})

    def test_unknown_model_and_level_are_rejected_by_name(self):
        with self.assertRaisesRegex(ValueError, "unknown model"):
            stranded.AgentConfig(model="GPT 5.6 Terra")
        with self.assertRaisesRegex(ValueError, "accepts reasoning"):
            stranded.AgentConfig(reasoning="max")
        with self.assertRaisesRegex(ValueError, "approval must be"):
            stranded.AgentConfig(approval_mode="maybe")

    def test_model_id_and_effort_reach_the_chat_completions_client(self):
        config = stranded.AgentConfig(model="GPT 5.6 Sol", reasoning="high")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            model = stranded.build_model(config)
        self.assertEqual(type(model).__name__, "OpenAIModel")
        self.assertEqual(model.get_config()["model_id"], "gpt-5.6-sol")
        self.assertEqual(model.get_config()["params"], {"reasoning_effort": "high"})
        self.assertEqual(model.client_args, {"api_key": "test-key"})

    def test_each_model_carries_its_own_endpoint(self):
        """The gateway case: one route per model, selected by picking the model."""
        gateway = {"default_model": "Luna via APISIX", "models": [
            {"name": "Luna via APISIX", "id": "gpt-5.6-luna",
             "base_url": "https://gw.example/luna/v1",
             "reasoning": ["none", "high"], "default_reasoning": "high"},
            {"name": "Sol via APISIX", "id": "gpt-5.6-sol",
             "base_url": "https://gw.example/sol/v1", "reasoning": []}]}
        with catalog_of(gateway), patch.dict(os.environ, {"OPENAI_API_KEY": "gw-token"}):
            luna = stranded.build_model(stranded.AgentConfig())
            sol = stranded.build_model(stranded.AgentConfig(model="Sol via APISIX"))
            self.assertEqual(stranded.AgentConfig().reasoning, "high")
            # A model with no reasoning levels is sent no reasoning_effort at all.
            self.assertIsNone(stranded.AgentConfig(model="Sol via APISIX").reasoning)
        self.assertEqual(luna.client_args["base_url"], "https://gw.example/luna/v1")
        self.assertEqual(sol.client_args["base_url"], "https://gw.example/sol/v1")
        self.assertEqual(luna.get_config()["params"], {"reasoning_effort": "high"})
        self.assertEqual(sol.get_config()["params"], {})

    def test_a_model_without_base_url_uses_the_client_default(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            model = stranded.build_model(stranded.AgentConfig())
        self.assertNotIn("base_url", model.client_args)

    def test_env_config_reads_the_environment_and_honors_overrides(self):
        environment = {"STRANDED_MODEL": "GPT 5.6 Sol", "STRANDED_REASONING": "xhigh",
                       "STRANDED_APPROVAL": "all"}
        with patch.dict(os.environ, environment, clear=False):
            config = stranded.env_config()
            self.assertEqual(config.model, "GPT 5.6 Sol")
            self.assertEqual(config.reasoning, "xhigh")
            self.assertEqual(config.approval_mode, "all")
            # An explicit flag wins; a None override leaves the environment alone.
            overridden = stranded.env_config(reasoning="high", model=None)
            self.assertEqual(overridden.reasoning, "high")
            self.assertEqual(overridden.model, "GPT 5.6 Sol")


class AgentTests(unittest.TestCase):
    def test_agent_exposes_the_expected_tools(self):
        agent = scripted_agent(stranded.AgentConfig())
        self.assertEqual(set(agent.tool_names),
                         {"execute_shell", "create_plan", "update_plan", "create_goal",
                          "get_goal", "update_goal", "web_search", "web_fetch", "skills"})

    def test_project_and_builtin_skills_are_discovered(self):
        agent = scripted_agent(stranded.AgentConfig())
        prompt = agent.system_prompt if isinstance(agent.system_prompt, str) else ""
        self.assertIn("STRANDed Agent", prompt)
        skills = sorted(path.parent.name for path in
                        stranded.BUILTIN_SKILLS_DIR.glob("*/SKILL.md"))
        self.assertEqual(skills, ["goals", "plans", "web-scrape", "web-search"])

    def test_builtin_skills_have_metadata(self):
        for path in sorted(stranded.BUILTIN_SKILLS_DIR.glob("*/SKILL.md")):
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"))
            self.assertIn(f"name: {path.parent.name}\n", text)
            self.assertIn("description:", text.split("---\n", 2)[1])

    def test_project_tools_are_on_path(self):
        self.assertIn(str(stranded.TOOLS_DIR), os.environ["PATH"].split(os.pathsep))


class ApprovalTests(unittest.TestCase):
    class FakeEvent:
        def __init__(self, name, args):
            self.tool_use = {"name": name, "input": args, "toolUseId": "call_1"}
            self.result = None

    def action_for(self, mode, name="execute_shell", args=None):
        events = []
        handler = stranded.Approval(stranded.AgentConfig(approval_mode=mode), events.append)
        action = handler.before_tool_call(self.FakeEvent(name, args or {"command": "ls"}))
        return action, events

    def test_modes_map_to_interventions(self):
        self.assertEqual(self.action_for("all")[0].type, "proceed")
        self.assertEqual(self.action_for("deny")[0].type, "deny")
        self.assertEqual(self.action_for("ask")[0].type, "confirm")

    def test_state_tools_never_need_approval(self):
        action, events = self.action_for("ask", "create_plan", {"steps": []})
        self.assertEqual(action.type, "proceed")
        self.assertEqual(events[0]["type"], "tool_call")

    def test_interactive_callers_answer_in_process(self):
        handler = stranded.Approval(stranded.AgentConfig(), None, ask=lambda name, args: True)
        action = handler.before_tool_call(self.FakeEvent("execute_shell", {"command": "ls"}))
        self.assertEqual(action.response, True)

    def test_tool_detail_summarizes_each_gated_tool(self):
        self.assertEqual(stranded.tool_detail("execute_shell", {"command": "ls"}),
                         "execute_shell: $ ls")
        self.assertEqual(stranded.tool_detail("web_search", {"query": "hi"}),
                         "web_search: query: hi")
        self.assertEqual(stranded.tool_detail("web_fetch", {"url": "https://x"}),
                         "web_fetch: url: https://x")


class ToolEventTests(unittest.TestCase):
    @staticmethod
    def result(text):
        return {"content": [{"text": text}]}

    def test_plan_goal_and_sources_become_ui_events(self):
        plan = self.result('{"plan":[{"step":"Look","status":"in_progress"}]}')
        self.assertEqual(stranded.tool_events("update_plan", plan)[0]["plan"][0]["status"],
                         "in_progress")
        goal = self.result('{"goal":{"objective":"Finish","status":"active"}}')
        self.assertEqual(stranded.tool_events("create_goal", goal)[0]["goal"]["status"], "active")
        sources = self.result('{"sources":[{"title":"T","url":"https://x"}]}')
        self.assertEqual(stranded.tool_events("web_search", sources)[0]["type"], "web_sources")

    def test_other_tools_and_bad_output_produce_nothing(self):
        self.assertEqual(stranded.tool_events("execute_shell", self.result("$ ls")), [])
        self.assertEqual(stranded.tool_events("update_plan", self.result("not json")), [])


class LoopTests(unittest.TestCase):
    def test_plan_tool_writes_agent_state_and_emits_an_update(self):
        agent = scripted_agent(
            stranded.AgentConfig(),
            tool_turn("create_plan", "call_1",
                      {"steps": [{"step": "Inspect the repository", "status": "in_progress"},
                                 {"step": "Run the tests"}]}),
            text_turn("plan created"))
        events = []
        turn = stranded.run_agent(agent, "make a plan", stranded.AgentConfig(), events.append)
        self.assertEqual(turn.status, "complete")
        self.assertEqual(turn.answer, "plan created")
        self.assertEqual(agent.state.get("plan")[0]["status"], "in_progress")
        self.assertTrue(any(event["type"] == "plan_update" for event in events))
        self.assertTrue(any(event["type"] == "tool_call" and event["name"] == "create_plan"
                            for event in events))
        self.assertEqual(turn.usage["total_tokens"], 10)

    def test_active_goal_continues_until_it_is_completed(self):
        agent = scripted_agent(
            stranded.AgentConfig(),
            tool_turn("create_goal", "call_1", {"objective": "Finish the task", "max_steps": 3}),
            text_turn("I will continue."),
            tool_turn("update_goal", "call_2", {"status": "complete", "explanation": "done"}),
            text_turn("finished"))
        events = []
        turn = stranded.run_agent(agent, "start", stranded.AgentConfig(), events.append)
        self.assertEqual(turn.status, "complete")
        self.assertEqual(turn.answer, "finished")
        self.assertEqual(agent.state.get("goal")["status"], "complete")
        self.assertTrue(any(event["type"] == "goal_iteration" for event in events))

    def test_approval_pauses_the_loop_and_resumes_with_the_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "ran.txt"
            agent = scripted_agent(
                stranded.AgentConfig(),
                tool_turn("execute_shell", "call_1",
                          {"command": f'echo ran > "{marker}"',
                           "description": "write a marker file for the test"}),
                text_turn("all done"))
            config = stranded.AgentConfig()
            turn = stranded.run_agent(agent, "run it", config)
            self.assertEqual(turn.status, "approval_required")
            self.assertIn("execute_shell: $ echo ran", turn.approval["detail"])
            self.assertFalse(marker.exists())

            turn = stranded.run_agent(agent, stranded.resume_prompt(turn.approval["id"], True),
                                      config)
            self.assertEqual(turn.status, "complete")
            self.assertEqual(turn.answer, "all done")
            self.assertTrue(marker.exists())

    def test_denied_tool_does_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "ran.txt"
            config = stranded.AgentConfig(approval_mode="deny")
            agent = scripted_agent(
                config,
                tool_turn("execute_shell", "call_1",
                          {"command": f'echo ran > "{marker}"',
                           "description": "write a marker file for the test"}),
                text_turn("nothing happened"))
            turn = stranded.run_agent(agent, "run it", config)
            self.assertEqual(turn.status, "complete")
            self.assertFalse(marker.exists())

    def test_shell_tool_requires_a_useful_description(self):
        self.assertEqual(stranded.execute_shell("echo hi", "short"),
                         "bad arguments: description must be 5-10 words")
        output = stranded.execute_shell("echo hi", "print a friendly greeting for the test")
        self.assertIn("exit 0", output)

    def test_model_errors_are_reported_not_raised(self):
        agent = scripted_agent(stranded.AgentConfig())
        with patch.object(agent, "stream_async", side_effect=RuntimeError("boom")):
            turn = stranded.run_agent(agent, "hello", stranded.AgentConfig())
        self.assertEqual(turn.status, "error")
        self.assertIn("boom", turn.error)


class TranscriptTests(unittest.TestCase):
    def test_display_messages_flattens_text_and_reasoning(self):
        messages = [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "thinking"}}},
                {"text": "hi"}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "call_1"}}]},
        ]
        shown = stranded.display_messages(messages)
        self.assertEqual([item["role"] for item in shown], ["user", "assistant"])
        self.assertEqual(shown[1], {"role": "assistant", "content": "hi", "reasoning": "thinking"})

    def test_sessions_round_trip_through_json(self):
        agent = scripted_agent(stranded.AgentConfig(), text_turn("saved"))
        config = stranded.AgentConfig()
        stranded.run_agent(agent, "remember this", config)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(stranded, "SESSIONS_FILE", Path(directory) / "sessions.json"):
                stranded.save_session(agent, "a label", config)
                saved = stranded.load_sessions()
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["label"], "a label")
        self.assertEqual(stranded.display_messages(saved[0]["messages"])[0]["content"],
                         "remember this")


class WebToolTests(unittest.TestCase):
    def test_html_parsers_extract_text_and_search_results(self):
        document = stranded_tools.DocumentParser()
        document.feed("<html><title>Example</title><script>ignore()</script>"
                      "<h1>Hello</h1><p>Readable text</p><a href='https://example.com'>Link</a></html>")
        self.assertEqual(document.title, "Example")
        self.assertIn("Readable text", document.lines)
        self.assertEqual(document.links[0]["url"], "https://example.com")

        results = stranded_tools.DuckDuckGoParser()
        results.feed("<a class='result__a' href='https://example.com'>Example result</a>"
                     "<div class='result__snippet'>A useful snippet.</div>")
        self.assertEqual(results.results[0]["title"], "Example result")
        self.assertEqual(results.results[0]["url"], "https://example.com")

    def test_local_and_private_addresses_are_refused(self):
        for url in ("http://localhost/x", "https://127.0.0.1/x", "ftp://example.com", "http://x.local/"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                stranded_tools._safe_url(url)

    def test_tool_schemas_come_from_the_type_hints(self):
        schema = stranded_tools.create_plan.tool_spec["inputSchema"]["json"]
        self.assertEqual(schema["required"], ["steps"])
        self.assertEqual(schema["$defs"]["PlanStep"]["properties"]["status"]["enum"],
                         ["pending", "in_progress", "completed"])
        search = stranded_tools.web_search.tool_spec["inputSchema"]["json"]
        self.assertEqual(search["properties"]["max_results"]["type"], "integer")


if __name__ == "__main__":
    unittest.main()
