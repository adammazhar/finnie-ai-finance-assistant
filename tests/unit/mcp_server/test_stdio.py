"""stdio transport: the server as Claude Desktop runs it, in a real subprocess."""

from __future__ import annotations

import os
import sys
from typing import Any

import anyio
from mcp import Client, StdioServerParameters

from src.core.config import PROJECT_ROOT


def test_stdio_server_lists_and_calls_tools(tmp_path):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        # Claude Desktop starts the process from elsewhere; PYTHONPATH finds the code
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "HF_HUB_OFFLINE": "1"},
    )

    async def main() -> tuple[set[str], dict[str, Any], str]:
        async with Client(params, read_timeout_seconds=60) as client:
            tools = await client.list_tools()
            result = await client.call_tool("explain_tax_account", {"account_type": "hsa"})
            glossary = await client.read_resource("finnie://glossary")
            return (
                {t.name for t in tools.tools},
                result.structured_content,
                glossary.contents[0].text,
            )

    names, hsa, glossary = anyio.run(main)
    assert len(names) == 6 and "search_financial_knowledge" in names
    assert hsa["name"] == "HSA" or "Health" in hsa["name"]
    assert glossary.startswith("# Finnie glossary")
