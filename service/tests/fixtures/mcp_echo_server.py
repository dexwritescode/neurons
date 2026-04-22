#!/usr/bin/env python3
"""Minimal MCP server over stdio for integration testing.
Implements JSON-RPC 2.0 (one message per line).
Exposes one tool: 'echo' — returns its 'text' argument as the result.
"""
import json
import sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue

    method = msg.get("method", "")
    msg_id = msg.get("id")

    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "echo-test-server", "version": "1.0"}
        }})

    elif method == "notifications/initialized":
        pass  # no response needed

    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": [
            {
                "name": "echo",
                "description": "Echoes the text argument back",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            }
        ]}})

    elif method == "tools/call":
        params = msg.get("params", {})
        args   = params.get("arguments", {})
        text   = args.get("text", "")
        send({"jsonrpc": "2.0", "id": msg_id, "result": {
            "content": [{"type": "text", "text": text}],
            "isError": False
        }})

    else:
        if msg_id is not None:
            send({"jsonrpc": "2.0", "id": msg_id,
                  "error": {"code": -32601, "message": "Method not found"}})
