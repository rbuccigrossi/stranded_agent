import http.client
import json
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

import stranded
import stranded_web
from tests.scripted_model import ScriptedModel, text_turn, tool_turn


class WebServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        patcher = patch.object(stranded_web, "WEB_SESSIONS_FILE",
                               Path(self.directory.name) / "sessions.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        stranded_web.SESSIONS.clear()
        stranded_web.AGENTS.clear()
        self.addCleanup(stranded_web.SESSIONS.clear)
        self.addCleanup(stranded_web.AGENTS.clear)
        self.server = HTTPServer(("127.0.0.1", 0), stranded_web.StrandedHandler)
        thread = threading.Thread(target=self.server.serve_forever)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def post(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        connection.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
        body = connection.getresponse().read().decode("utf-8")
        connection.close()
        return [json.loads(block[5:]) for block in body.split("\n\n")
                if block.startswith("data:")] or body

    def get(self, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        connection.request("GET", path)
        body = connection.getresponse().read()
        connection.close()
        return json.loads(body)

    @staticmethod
    def scripted(*turns):
        return patch.object(stranded, "build_model", return_value=ScriptedModel(*turns))

    def test_two_messages_complete_and_are_persisted(self):
        with self.scripted(text_turn("turn one"), text_turn("turn two")):
            first = self.post("/api/chat", {"prompt": "first", "config": {"approval": "deny"}})
            session_id = next(iter(stranded_web.SESSIONS))
            second = self.post("/api/chat", {"session_id": session_id, "prompt": "second"})
        self.assertEqual(first[-1]["status"], "complete")
        self.assertEqual(first[-1]["answer"], "turn one")
        self.assertEqual(second[-1]["answer"], "turn two")
        self.assertEqual(len(stranded_web.SESSIONS[session_id]["messages"]), 4)
        self.assertEqual(stranded_web.SESSIONS[session_id]["label"], "first")

        view = self.get("/api/sessions/" + session_id)
        self.assertEqual([item["content"] for item in view["messages"]],
                         ["first", "turn one", "second", "turn two"])
        self.assertEqual(json.loads(Path(self.directory.name, "sessions.json").read_text())[0]["id"],
                         session_id)

    def test_approval_round_trip_runs_the_command(self):
        marker = Path(self.directory.name) / "ran.txt"
        turns = (tool_turn("execute_shell", "call_1",
                           {"command": f'echo ran > "{marker}"',
                            "description": "write a marker file for the test"}),
                 text_turn("all done"))
        with self.scripted(*turns):
            events = self.post("/api/chat", {"prompt": "run it", "config": {"approval": "ask"}})
            session_id = next(iter(stranded_web.SESSIONS))
            pending = next(event for event in events if event["type"] == "approval_required")
            self.assertIn("execute_shell: $ echo ran", pending["approval"]["detail"])
            self.assertFalse(marker.exists())

            resumed = self.post("/api/approve", {"session_id": session_id,
                                                 "interrupt_id": pending["approval"]["id"],
                                                 "approved": True})
        self.assertEqual(resumed[-1]["status"], "complete")
        self.assertEqual(resumed[-1]["answer"], "all done")
        self.assertTrue(marker.exists())

    def test_expired_approval_is_rejected(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        connection.request("POST", "/api/approve", json.dumps({"session_id": "missing"}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 404)
        connection.close()

    def test_catalog_is_served_for_the_dropdowns(self):
        served = self.get("/api/catalog")
        self.assertEqual(served, stranded.catalog())
        luna = served["models"][0]
        self.assertEqual(luna["name"], "GPT 5.6 Luna")
        self.assertIn("xhigh", luna["reasoning"])

    def test_browser_choices_are_validated_and_stored(self):
        with self.scripted(text_turn("done")):
            self.post("/api/chat", {"prompt": "hi", "config": {
                "model": "GPT 5.6 Sol", "reasoning": "high", "approval": "deny"}})
        stored = next(iter(stranded_web.SESSIONS.values()))["config"]
        self.assertEqual(stored, {"model": "GPT 5.6 Sol",
                                  "reasoning": "high", "approval": "deny"})
        self.assertEqual(stored, self.get("/api/sessions")[0]["config"])

    def test_a_rejected_choice_returns_400(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        connection.request("POST", "/api/chat", json.dumps(
            {"prompt": "hi", "config": {"model": "GPT 5.6 Sol", "reasoning": "max"}}),
            {"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        connection.close()
        self.assertEqual(response.status, 400)
        self.assertIn("accepts reasoning", body)

    def test_page_renders_the_stranded_name_and_event_handlers(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        connection.request("GET", "/")
        page = connection.getresponse().read().decode("utf-8")
        connection.close()
        self.assertIn("<title>STRANDed Agent</title>", page)
        for fragment in ("function scrollChat()", "function showToolCall(call)",
                         "function loadCatalog()", "function syncReasoning(chosen)",
                         'id="model"', 'id="reasoning"',
                         "function showPlan(plan)", "function showGoal(goal)",
                         "function showSources(sources)", "insertBefore(d,x.body)",
                         "insertBefore(box,x.body)", "min-height:0", "overflow-y:auto"):
            self.assertIn(fragment, page)


if __name__ == "__main__":
    unittest.main()
