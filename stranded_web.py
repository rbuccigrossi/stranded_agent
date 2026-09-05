#!/usr/bin/env python3
"""A single-threaded, standard-library browser UI for stranded.py."""

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from strands import Agent

import stranded


HOST = "127.0.0.1"
PORT = 8765
WEB_SESSIONS_FILE = stranded.ROOT_DIR / ".stranded_web_sessions.json"
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSIONS_LOCK = threading.Lock()

#: Live ``(config, agent)`` pairs by session id. An approval pauses the
#: Strands agent mid-loop, so the same object has to be there when the browser answers.
AGENTS: Dict[str, Any] = {}


def load_sessions() -> Dict[str, Dict[str, Any]]:
    try:
        with WEB_SESSIONS_FILE.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, list):
            return {item["id"]: item for item in value if isinstance(item, dict) and item.get("id")}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}


SESSIONS.update(load_sessions())


def save_sessions() -> None:
    with SESSIONS_LOCK:
        WEB_SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(SESSIONS.values(), key=lambda item: item.get("updated", 0))[-50:]
        with WEB_SESSIONS_FILE.open("w", encoding="utf-8") as handle:
            json.dump(ordered, handle, indent=2, default=str)


def session_config(config: stranded.AgentConfig) -> Dict[str, Any]:
    """The configuration fields the browser shows and sends back."""
    return {"model": config.model, "reasoning": config.reasoning,
            "approval": config.approval_mode}


def new_session() -> Dict[str, Any]:
    now = int(time.time())
    session = {
        "id": uuid.uuid4().hex,
        "label": "New chat",
        "created": now,
        "updated": now,
        "config": session_config(stranded.env_config()),
        "messages": [],
        "state": {},
    }
    SESSIONS[session["id"]] = session
    save_sessions()
    return session


def session_summary(session: Dict[str, Any]) -> Dict[str, Any]:
    return {key: session.get(key) for key in ("id", "label", "updated", "config")}


def session_view(session: Dict[str, Any]) -> Dict[str, Any]:
    """What the browser needs to redraw a conversation it is opening."""
    return {**session_summary(session),
            "messages": stranded.display_messages(session.get("messages") or [])}


def config_from_payload(payload: Dict[str, Any], session: Dict[str, Any]) -> stranded.AgentConfig:
    """Validate the browser's choices against config.json, falling back to the environment."""
    incoming = {**(session.get("config") or {}), **(payload.get("config") or {})}
    config = stranded.env_config(model=incoming.get("model"),
                                 reasoning=incoming.get("reasoning"),
                                 approval_mode=incoming.get("approval"))
    session["config"] = session_config(config)
    return config


def agent_for(session: Dict[str, Any], config: stranded.AgentConfig) -> Agent:
    """Return this session's live agent, rebuilding it when the configuration changes."""
    cached_config, cached_agent = AGENTS.get(session["id"], (None, None))
    if cached_agent is not None and cached_config == config:
        return cached_agent
    agent = stranded.build_agent(config, session.get("messages"), session.get("state"))
    AGENTS[session["id"]] = (config, agent)
    return agent


def record(session: Dict[str, Any], agent: Agent) -> None:
    """Copy the agent's conversation and state back onto the stored session."""
    session["messages"] = json.loads(json.dumps(agent.messages, default=str))
    session["state"] = agent.state.get()
    session["updated"] = int(time.time())
    save_sessions()


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>STRANDed Agent</title>
<style>
:root{color-scheme:dark;font:15px/1.45 system-ui,sans-serif;--bg:#171717;--panel:#202020;--line:#363636;--muted:#a5a5a5;--accent:#8ab4f8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eee;height:100vh;display:flex;overflow:hidden}
button,input,select{font:inherit;color:inherit}button{border:1px solid var(--line);background:#292929;border-radius:8px;padding:9px 12px;cursor:pointer}button:hover{background:#353535}
.layout{display:grid;grid-template-columns:260px 1fr;width:100%;min-height:0}.side{background:var(--panel);border-right:1px solid var(--line);padding:16px;display:flex;flex-direction:column;gap:14px;min-height:0}.brand{font-size:20px;font-weight:650;letter-spacing:.02em}.new{width:100%;text-align:left}.sessions{overflow:auto;display:flex;flex-direction:column;gap:4px}.session{border:0;background:transparent;text-align:left;color:#ddd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.session.active{background:#353535}.hint{color:var(--muted);font-size:12px;margin-top:auto}
.main{display:flex;flex-direction:column;min-width:0;min-height:0}.top{border-bottom:1px solid var(--line);padding:12px 22px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}.top label{color:var(--muted);font-size:12px}.control{display:flex;align-items:center;gap:5px}input,select{background:#292929;border:1px solid var(--line);border-radius:7px;padding:7px 9px;min-width:120px}.model{min-width:190px}.token{margin-left:auto;color:var(--muted);font-size:12px}.chat{flex:1;min-height:0;overflow-y:auto;padding:28px max(22px,calc((100% - 850px)/2));display:flex;flex-direction:column;gap:18px}.msg{max-width:850px;width:100%;white-space:pre-wrap}.msg.user{align-self:flex-end;background:#2d3d56;border-radius:13px;padding:12px 15px;max-width:680px}.msg .role{color:var(--muted);font-size:12px;margin-bottom:5px}.body{margin:0;font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}.reasoning{margin-top:10px;border-left:2px solid #555;padding-left:10px;color:#aaa;font-size:13px}.approval{margin-top:12px;border:1px solid #806b3b;background:#332d20;border-radius:10px;padding:12px}.approval code{display:block;color:#f0d594;margin:7px 0;white-space:pre-wrap}.approval button{margin-right:7px}.composer{padding:14px max(22px,calc((100% - 850px)/2));border-top:1px solid var(--line);display:flex;gap:8px}.composer textarea{flex:1;resize:none;min-height:46px;max-height:160px;background:#292929;border:1px solid var(--line);border-radius:10px;padding:12px;color:#eee;font:inherit}.empty{color:var(--muted);text-align:center;margin:auto}.error{color:#ff9b9b}.small{font-size:12px;color:var(--muted)}
@media(max-width:700px){.layout{grid-template-columns:1fr}.side{display:none}.top{padding:10px}.chat,.composer{padding-left:12px;padding-right:12px}.token{width:100%;margin-left:0}}
</style></head>
<body><div class="layout"><aside class="side"><div class="brand">STRANDed</div><button class="new" id="new">＋ New chat</button><div class="sessions" id="sessions"></div><div class="hint">Single-thread local UI<br>Tool approval stays in your hands.</div></aside>
<main class="main"><header class="top"><div class="control"><label for="model">model</label><select id="model" class="model"></select></div><div class="control"><label for="reasoning">reasoning</label><select id="reasoning"></select></div><div class="control"><label for="approval">approval</label><select id="approval"><option value="ask">Ask</option><option value="all">All</option><option value="deny">Deny</option></select></div><div class="token" id="tokens">tokens: 0</div></header><section class="chat" id="chat"><div class="empty" id="empty">Start a conversation with the STRANDed Agent.</div></section><form class="composer" id="composer"><textarea id="prompt" placeholder="Message the agent…" autofocus></textarea><button id="send">Send</button></form></main></div>
<script>
const $=id=>document.getElementById(id);let activeId=null,assistant=null,pendingId=null,catalog={models:[]};
function fill(select,values,chosen){select.replaceChildren(...values.map(v=>{const o=document.createElement('option');o.value=o.textContent=v;return o}));if(values.includes(chosen))select.value=chosen}
function syncReasoning(chosen){const m=(catalog.models||[]).find(x=>x.name===$('model').value)||{};const levels=m.reasoning||[];fill($('reasoning'),levels,chosen||m.default_reasoning);$('reasoning').disabled=!levels.length}
function syncModels(model,reasoning){fill($('model'),(catalog.models||[]).map(m=>m.name),model);syncReasoning(reasoning)}
async function loadCatalog(){catalog=await (await fetch('/api/catalog')).json();syncModels(catalog.default_model)}
function applyConfig(c){syncModels(c.model,c.reasoning);$('approval').value=c.approval}
function scrollChat(){requestAnimationFrame(()=>{$('chat').scrollTop=$('chat').scrollHeight})}
function showToolCall(call){assistant=null;if($('empty'))$('empty').remove();const box=document.createElement('div');box.className='small';box.textContent='Tool: '+(call.detail||call.name||'unknown');$('chat').append(box);scrollChat()}
function showPlan(plan){assistant=null;if($('empty'))$('empty').remove();const box=document.createElement('details');box.className='reasoning';box.open=true;const s=document.createElement('summary');s.textContent='Plan update';box.append(s,...(plan||[]).map(x=>{const d=document.createElement('div');d.textContent=(x.status==='completed'?'✓ ':x.status==='in_progress'?'→ ':'○ ')+(x.step||'');return d}));$('chat').append(box);scrollChat()}
function showGoal(goal){assistant=null;if(!goal)return;if($('empty'))$('empty').remove();const box=document.createElement('div');box.className='small';box.textContent=`Goal [${goal.status}]: ${goal.objective}`;$('chat').append(box);scrollChat()}
function showSources(sources){if(!(sources||[]).length)return;assistant=null;if($('empty'))$('empty').remove();const box=document.createElement('div');box.className='small';box.append(document.createTextNode('Sources: '),...(sources||[]).map(x=>{const a=document.createElement('a');a.href=x.url;a.target='_blank';a.rel='noopener';a.textContent=x.title||x.url;return a}));$('chat').append(box);scrollChat()}
function addMessage(role,text=''){if($('empty'))$('empty').remove();const box=document.createElement('article');box.className='msg '+role;const r=document.createElement('div');r.className='role';r.textContent=role==='user'?'You':'STRANDed';const b=document.createElement('pre');b.className='body';b.textContent=text;box.append(r,b);$('chat').append(box);scrollChat();return {box,body:b}}
function showReasoning(x,text){let d=x.box.querySelector('.reasoning');if(!d){d=document.createElement('details');d.className='reasoning';const s=document.createElement('summary');s.textContent='Reasoning summary';d.append(s,document.createElement('div'));x.box.insertBefore(d,x.body)}d.lastChild.textContent+=text;scrollChat()}
function renderMessage(m){const x=addMessage(m.role,m.content||'');if(m.reasoning)showReasoning(x,m.reasoning);return x}
function renderSessions(items){$('sessions').replaceChildren(...items.map(s=>{const b=document.createElement('button');b.className='session'+(s.id===activeId?' active':'');b.textContent=s.label||'New chat';b.onclick=()=>openSession(s.id);return b}))}
async function loadSessions(){const r=await fetch('/api/sessions');renderSessions(await r.json())}
async function openSession(id){const r=await fetch('/api/sessions/'+id);const s=await r.json();activeId=id;assistant=null;pendingId=null;$('chat').replaceChildren();s.messages.forEach(renderMessage);applyConfig(s.config);loadSessions()}
async function newChat(){const r=await fetch('/api/new',{method:'POST'});const s=await r.json();await openSession(s.id)}
function startAssistant(){if(!assistant)assistant=addMessage('assistant','');return assistant}
function showApproval(a){const x=startAssistant();const box=document.createElement('div');box.className='approval';const p=document.createElement('div');p.textContent='Approval required';const c=document.createElement('code');c.textContent=a.detail||'';const yes=document.createElement('button');yes.textContent='Approve';const no=document.createElement('button');no.textContent='Deny';yes.onclick=()=>decide(true,box);no.onclick=()=>decide(false,box);box.append(p,c,yes,no);x.box.insertBefore(box,x.body);scrollChat()}
async function decide(ok,box){box.querySelectorAll('button').forEach(b=>b.disabled=true);await stream('/api/approve',{session_id:activeId,interrupt_id:pendingId,approved:ok},false)}
async function stream(url,payload,newTurn=true){
if(newTurn){assistant=null;pendingId=null}
$('send').disabled=true;
let response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
if(!response.body){$('send').disabled=false;return}
let reader=response.body.getReader(),decoder=new TextDecoder(),buffer='',finished=false;
function handle(block){
 const line=block.split('\n').find(x=>x.startsWith('data:'));if(!line)return;
 const e=JSON.parse(line.slice(5));
 if(e.type==='session'){activeId=e.session_id;loadSessions()}
 else if(e.type==='tool_call'){showToolCall(e)}
 else if(e.type==='text_delta'){startAssistant().body.textContent+=e.text;scrollChat()}
 else if(e.type==='reasoning_delta'){showReasoning(startAssistant(),e.text)}
 else if(e.type==='plan_update'){showPlan(e.plan)}
 else if(e.type==='goal_update'){showGoal(e.goal)}
 else if(e.type==='goal_iteration'){showGoal({status:`iteration ${e.iteration}/${e.limit}`,objective:'Continuing toward the active goal'})}
 else if(e.type==='web_sources'){showSources(e.sources)}
 else if(e.type==='usage'){$('tokens').textContent='tokens: '+(e.usage.total_tokens||0)}
 else if(e.type==='approval_required'){pendingId=e.approval.id;showApproval(e.approval)}
 else if(e.type==='done'){if(e.answer){const x=startAssistant();if(!x.body.textContent)x.body.textContent=e.answer}if(e.status==='error'){const x=startAssistant();x.body.classList.add('error');x.body.textContent=e.error||'Request failed'}if(e.reasoning){const x=startAssistant();if(!x.box.querySelector('.reasoning'))showReasoning(x,e.reasoning)}if(e.usage)$('tokens').textContent='tokens: '+(e.usage.total_tokens||0);loadSessions();scrollChat();finished=true}
}
while(!finished){const q=await reader.read();if(q.done)break;buffer+=decoder.decode(q.value,{stream:true});let parts=buffer.split('\n\n');buffer=parts.pop();parts.forEach(handle)}
if(reader.cancel)await reader.cancel();
$('send').disabled=false;$('prompt').focus()
}
$('composer').onsubmit=e=>{e.preventDefault();const p=$('prompt').value.trim();if(!p)return;addMessage('user',p);$('prompt').value='';stream('/api/chat',{session_id:activeId,prompt:p,config:{model:$('model').value,reasoning:$('reasoning').value,approval:$('approval').value}})};
$('model').onchange=()=>syncReasoning();
$('new').onclick=newChat;$('prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('composer').requestSubmit()}};loadCatalog().then(loadSessions);
</script></body></html>'''


class StrandedHandler(BaseHTTPRequestHandler):
    server_version = "StrandedWeb/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length) or b"{}")
        return value if isinstance(value, dict) else {}

    def event(self, value: Dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(value, default=str)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/catalog":
            self.send_json(stranded.catalog())
        elif path == "/api/sessions":
            items = sorted(SESSIONS.values(), key=lambda item: item.get("updated", 0), reverse=True)
            self.send_json([session_summary(item) for item in items])
        elif path.startswith("/api/sessions/"):
            session = SESSIONS.get(path.rsplit("/", 1)[-1])
            self.send_json(session_view(session) if session else {"error": "session not found"},
                           200 if session else 404)
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/new":
                self.send_json(session_summary(new_session()))
            elif path == "/api/chat":
                self.chat(self.read_json())
            elif path == "/api/approve":
                self.approve(self.read_json())
            else:
                self.send_json({"error": "not found"}, 404)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, 400)

    def chat(self, payload: Dict[str, Any]) -> None:
        session = SESSIONS.get(payload.get("session_id")) or new_session()
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self.send_json({"error": "prompt is required"}, 400)
            return
        if not session.get("messages"):
            session["label"] = prompt[:80]
        self.run_turn(session, config_from_payload(payload, session), prompt)

    def approve(self, payload: Dict[str, Any]) -> None:
        session = SESSIONS.get(payload.get("session_id"))
        interrupt_id = payload.get("interrupt_id")
        if not session or session["id"] not in AGENTS or not interrupt_id:
            self.send_json({"error": "approval request expired"}, 404)
            return
        self.run_turn(session, AGENTS[session["id"]][0],
                      stranded.resume_prompt(str(interrupt_id), bool(payload.get("approved"))))

    def run_turn(self, session: Dict[str, Any], config: stranded.AgentConfig, prompt: Any) -> None:
        """Stream one agent turn to the browser as server-sent events."""
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.event({"type": "session", "session_id": session["id"]})

        def emit(value: Dict[str, Any]) -> None:
            if value["type"] == "tool_call":
                value = {**value, "detail": stranded.tool_detail(value["name"], value["arguments"])}
            self.event(value)

        agent = agent_for(session, config)
        turn = stranded.run_agent(agent, prompt, config, emit)
        record(session, agent)
        if turn.status == "approval_required":
            self.event({"type": "approval_required", "approval": turn.approval})
        self.event({"type": "done", "status": turn.status, "answer": turn.answer,
                    "reasoning": turn.reasoning, "usage": turn.usage, "error": turn.error})


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), StrandedHandler)
    print(f"STRANDed Agent web UI listening at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
