"""The offline LLM calls (Stages 1-3): one cached ChatClient per JSON schema.

Structured output is enforced by the server (Ollama's OpenAI endpoint honours `json_schema`), so
a reply that does not parse is an error, not something to repair."""
import json

from cook_ad.llm.client import ChatClient


def client_for(llm_cfg, schema, name, max_requests=None):
    return ChatClient(
        model=llm_cfg["model"], base_url=llm_cfg["base_url"], cache_dir=llm_cfg["cache_dir"],
        temperature=llm_cfg.get("temperature", 0.0), max_requests=max_requests,
        response_format={"type": "json_schema",
                         "json_schema": {"name": name, "strict": True, "schema": schema}},
    )


def ask_json(client, system, user):
    return json.loads(client.complete([{"role": "system", "content": system},
                                       {"role": "user", "content": user}]))
