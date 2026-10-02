# Finnie MCP Server

Finnie's market data, portfolio analytics, goal projections, and knowledge base, as an
[MCP](https://modelcontextprotocol.io) server. A client such as Claude Desktop or Claude
Code does the reasoning; Finnie supplies the data and calculations. The server never calls
Finnie's own LLM, so no API key for OpenAI or Anthropic is needed to use it.

- Specification: MCP **2026-07-28** (stateless Streamable HTTP). Clients that still use
  the older `initialize` handshake (2025-03-26 to 2025-11-25) also work.
- SDK: official Python SDK `mcp==2.2.0` (`MCPServer`).
- Code: `src/mcp_server/server.py` (tools, resources, prompt), `http.py` (HTTP transport and
  bearer token), `__main__.py` (launcher).

## What it offers

| Type | Name | What it returns |
|---|---|---|
| tool | `get_stock_quote(ticker)` | Price, change, exact price time, live/delayed/last close, market open or closed |
| tool | `get_market_overview()` | S&P 500, Nasdaq-100, Dow, Russell 2000 levels with their tracking ETFs; 11 sector ETFs; a short mood summary |
| tool | `analyze_portfolio(holdings, risk_tolerance?)` | Value, weights, asset and sector allocation, diversification, risk, fund fees, past-year risk measures |
| tool | `project_financial_goal(target_amount, years, ...)` | Monte Carlo odds of reaching the goal, P10/median/P90, the monthly amount for an 80% chance |
| tool | `search_financial_knowledge(query, category?, limit?)` | The best knowledge base passages with their sources and a resource link |
| tool | `explain_tax_account(account_type)` | How a 401(k), IRA, Roth, HSA, 529, or taxable account is taxed, with this year's IRS limits |
| resource | `finnie://articles/{article_id}` | A whole article in markdown, with sources |
| resource | `finnie://glossary` | The glossary, A to Z |
| prompt | `explain_like_beginner(topic)` | A plain-language explanation grounded in Finnie's sources |

Every tool has a typed output schema and returns structured content. Every result carries
the education-only disclaimer, and market data carries its time and freshness. Failures
come back as MCP tool errors with a plain message, never a stack trace.

## Transports

| | stdio | Streamable HTTP |
|---|---|---|
| Start | The client starts it: `python -m src.mcp_server` | You start it: `python -m src.mcp_server --http` |
| For | Claude Desktop | Claude Code, scripts, any HTTP MCP client |
| Address | the process's stdin/stdout | `http://127.0.0.1:8765/mcp` (`mcp` section of `config.yaml`) |
| Auth | none needed: only the user who started the process can talk to it | `Authorization: Bearer <MCP_API_TOKEN>`, otherwise **401** |

Logs go to stderr (stdout carries the protocol in stdio mode). They never contain tool
arguments, results, or the token.

## 1. Claude Desktop (stdio, Windows)

**Before you start:** the project's `.venv` is set up (`pip install -r requirements.txt`) and
the knowledge base index is built (`python scripts/build_index.py`, see the README).

1. **Open the config file.** In Claude Desktop, go to **Settings → Developer → Edit Config**.
   That opens the folder with `claude_desktop_config.json`, normally
   `%APPDATA%\Claude\claude_desktop_config.json`. (If Claude Desktop came from the Microsoft
   Store, use the Edit Config button rather than that path, because the file lives in the
   app's own package folder.) If the file doesn't exist yet, create it.

2. **Add Finnie.** Put this in the file. If it already has an `mcpServers` section, add only
   the `"finnie": {...}` entry inside it.

   ```json
   {
     "mcpServers": {
       "finnie": {
         "command": "C:\\SynologyDrive\\home\\projects\\ai\\finnie-ai-finance-assistant\\.venv\\Scripts\\python.exe",
         "args": ["-m", "src.mcp_server"],
         "env": {
           "PYTHONPATH": "C:\\SynologyDrive\\home\\projects\\ai\\finnie-ai-finance-assistant",
           "HF_HUB_OFFLINE": "1"
         }
       }
     }
   }
   ```

   - `command` is the `.venv` Python, so Finnie's packages are used. JSON needs every
     backslash doubled.
   - Claude Desktop starts the process from its own folder. `PYTHONPATH` lets Python find
     `src`, and the server then switches to the project folder, so `config.yaml`, `.env`,
     and `data/` are found.
   - `HF_HUB_OFFLINE=1` loads the embedding model from the local cache without contacting
     Hugging Face.
   - For a different checkout, replace the folder in both places.

3. **Fully quit Claude Desktop.** Closing the window isn't enough: the app keeps running in
   the background and doesn't reread the config. Click the **^** (Show hidden icons) arrow
   at the right of the taskbar, right-click the **Claude** icon, and choose **Quit**. If
   you're not sure it's closed, check Task Manager for `Claude.exe` and end it.

4. **Start Claude Desktop again.**

5. **Check that the tools appear.** In a new chat, click the **+** (or the tools/connectors
   icon) under the message box. **finnie** should be listed with its 6 tools. Clicking it
   shows `get_stock_quote`, `get_market_overview`, `analyze_portfolio`,
   `project_financial_goal`, `search_financial_knowledge`, and `explain_tax_account`.
   **Settings → Developer** also lists **finnie** as *running*.

6. **Try it.** Ask, for example:
   - "Use Finnie to get a quote for VTI and tell me when the price is from."
   - "With Finnie, project saving toward $50,000 in 10 years with $5,000 saved and $300 a month."
   - "Search Finnie's knowledge base for how expense ratios work, and cite the sources."

   Claude asks permission the first time it uses each tool. The beginner prompt is under
   **+ → finnie → explain_like_beginner**, and the articles and glossary are under the same
   menu as attachable resources.

**If finnie doesn't appear or shows an error**

- Read the server's log: `%APPDATA%\Claude\logs\mcp-server-finnie.log` (Claude Desktop's
  own MCP log is `mcp.log` in the same folder). **Settings → Developer → finnie** shows
  the most recent error too.
- `python.exe` not found: check the path in `command` with
  `dir C:\SynologyDrive\home\projects\ai\finnie-ai-finance-assistant\.venv\Scripts\python.exe`.
- `No module named src`: check `PYTHONPATH` points at the project folder.
- Invalid JSON: a missing comma or a single backslash. Paste the file into a JSON validator.
- Run the exact command yourself from another folder. It should wait silently for input;
  press Ctrl+C to stop it:

  ```powershell
  cd $env:USERPROFILE
  $env:PYTHONPATH = "C:\SynologyDrive\home\projects\ai\finnie-ai-finance-assistant"
  C:\SynologyDrive\home\projects\ai\finnie-ai-finance-assistant\.venv\Scripts\python.exe -m src.mcp_server
  ```

- The first knowledge base search takes a few seconds while the embedding model loads.

## 2. Streamable HTTP on localhost (bearer token)

1. **Create a token** and put it in `.env` (never commit it; `.env` is git-ignored):

   ```powershell
   .venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

   ```dotenv
   MCP_API_TOKEN=<the value printed above>
   ```

   The server refuses to start without a token, or with one shorter than 32 characters
   (`mcp.min_token_length`).

2. **Start the server** and leave it running:

   ```powershell
   .venv\Scripts\python -m src.mcp_server --http
   # Finnie MCP server on http://127.0.0.1:8765/mcp (bearer token required)
   ```

   `--port 9000` overrides the port in `config.yaml`.

3. **Run the demo client** in another terminal:

   ```powershell
   .venv\Scripts\python scripts/mcp_client_demo.py
   .venv\Scripts\python scripts/mcp_client_demo.py --ticker AAPL   # also a live quote
   ```

   It sends a request with no token (401), one with a wrong token (401), then connects with
   the token from `.env`, lists the tools, and calls `explain_tax_account`. Abridged output:

   ```text
   1. No token:
     HTTP 401  WWW-Authenticate: Bearer realm="finnie"
     {"error": "missing_token", "error_description": "A valid bearer token (MCP_API_TOKEN) is required."}

   2. Wrong token:
     HTTP 401  WWW-Authenticate: Bearer realm="finnie", error="invalid_token"
     {"error": "invalid_token", "error_description": "A valid bearer token (MCP_API_TOKEN) is required."}

   3. With the token: connected (protocol 2026-07-28)
     6 tools:
      - get_stock_quote: Stock or fund quote
      ...
     call explain_tax_account(account_type="roth_ira"):
     { "name": "Roth IRA", "growth": "Tax-free", ... "tax_year": 2026, ... }
   ```

4. **Connect Claude Code** to it. Local scope keeps the token in your user settings
   (`~/.claude.json`), not in the repository. In PowerShell, from the project folder:

   ```powershell
   $token = (Select-String -Path .env -Pattern '^MCP_API_TOKEN=(.+)$').Matches[0].Groups[1].Value
   claude mcp add --transport http --scope local finnie http://127.0.0.1:8765/mcp --header "Authorization: Bearer $token"
   claude mcp list
   # finnie: http://127.0.0.1:8765/mcp (HTTP) - ✔ Connected
   ```

   Then, in Claude Code, ask something like "use finnie to get a VTI quote", or run `/mcp`
   to see the server and its tools. Remove it with `claude mcp remove finnie -s local`. If
   you change the token, remove the server and add it again.

**What the HTTP server enforces**

| Request | Response |
|---|---|
| No `Authorization` header | `401`, `WWW-Authenticate: Bearer realm="finnie"` |
| Wrong token, or a scheme other than Bearer | `401`, `WWW-Authenticate: Bearer realm="finnie", error="invalid_token"` |
| Valid token, `Origin` that isn't a localhost page | `403` (DNS-rebinding protection) |
| Valid token, `Host` that isn't localhost | `421` |
| Valid token, `GET` / `DELETE` | `405` (stateless server: no stream to open or session to end) |
| Valid token, `POST` | the MCP response |

The token check runs first, in constant time (`hmac.compare_digest`), and covers every path
and method. The server listens on 127.0.0.1 only. A static token suits one person on one
machine; for a server other people reach over a network, the production design is OAuth
with Auth0 (DESIGN.md §9.4), not built.

## 3. MCP Inspector (optional)

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) needs Node.js:

```powershell
npx @modelcontextprotocol/inspector
```

Choose **Streamable HTTP**, URL `http://127.0.0.1:8765/mcp`, and add the header
`Authorization: Bearer <token>`. Or choose **STDIO** with the command and arguments from the
Claude Desktop entry above.

## Tests

`tests/unit/mcp_server/`:

- `test_server.py`: every tool, resource, and the prompt through an in-memory MCP client,
  including error results and that failed tool arguments stay out of logs.
- `test_stdio.py`: the stdio transport in a real subprocess started from another folder, as
  Claude Desktop does.
- `test_http.py`: the HTTP transport on a real uvicorn server on loopback. It covers 401
  without a token and with wrong tokens, 405, 403 for a foreign `Origin`, 421 for a foreign
  `Host`, the older `initialize` handshake, and a real MCP client with and without the
  token.
- `test_main.py`: the launcher, including refusing to start without a token.
