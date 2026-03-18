# Arbos

<p align="center">
  Arbos is a <a href="https://ghuntley.com/loop/">Ralph-loop</a> combined with a Telegram bot.<br>
  It loops a goal through Claude Code, powered by Anthropic with an OpenRouter fallback.
</p>

## The Design

Arbos loops a `GOAL.md` through a coding agent, step after step, with no memory between steps except `STATE.md`.

```
                                 ┌────── [GOAL.md] ────────┐
                                 ▼                         │
            ┌──────────┐     ┌───────┐                     │
            │ Telegram │◄───►│ Agent │─────────────────────┘
            └──────────┘     └───────┘
```

## Providers

| Priority | Provider | Model | Auth |
|----------|----------|-------|------|
| Primary | **Anthropic** | `claude-sonnet-4-6` | Claude Code Pro (OAuth) |
| Fallback | **OpenRouter** | `stepfun/step-3.5-flash:free` | API key |

When the Anthropic quota is exceeded, Arbos automatically switches to the OpenRouter free model. At each new step, it retries Anthropic first.

## Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) with Pro auth (`claude login`)
- [Telegram Bot token](https://core.telegram.org/bots#how-do-i-create-a-bot)
- [OpenRouter API key](https://openrouter.ai) (for fallback)
- Python 3.10+, `pm2`

## Getting started

```sh
git clone https://github.com/JulAius/arbos_genesis.git
cd arbos_genesis
cp .env.example .env
# Edit .env with your tokens
python3 -m venv .venv && source .venv/bin/activate
pip install -r <(grep -oP '"\K[^"]+' pyproject.toml | head -20) 2>/dev/null || pip install requests httpx uvicorn fastapi pyTelegramBotAPI python-dotenv cryptography
pm2 start .arbos-launch.sh --name arbos
```

## Usage

Send `/goal` to your Telegram bot:

```
/goal
Build a trading system that predicts BTC direction on a 15-minute horizon.
```

### Telegram commands

| Command | Description |
|---------|-------------|
| `/goal <text>` | Set the goal for a slot |
| `/status` | Show current goals and step counts |
| `/pause` / `/resume` | Pause/resume a goal |
| `/restart` | Restart the process via pm2 |
| `/update` | Git pull and restart |
| `/clear` | Reset context and state |

## How it works

1. Each **step** is a single `claude -p` invocation with full tool access
2. Steps run back-to-back on success, with exponential backoff on failure
3. `STATE.md` is the only memory between steps — if it's not written there, it's forgotten
4. The Telegram bot relays operator messages and streams agent responses
5. SIGINT/SIGTERM are handled gracefully — no crash restarts

## Configuration

See `.env.example` for all options:

```env
PROVIDER=anthropic              # primary provider
CLAUDE_MODEL=claude-sonnet-4-6  # primary model
FALLBACK_PROVIDER=openrouter    # fallback on quota exceeded
FALLBACK_MODEL=stepfun/step-3.5-flash:free
OPENROUTER_API_KEY=sk-or-...    # fallback API key
TAU_BOT_TOKEN=...               # Telegram bot token
TELEGRAM_OWNER_ID=...           # your Telegram user ID
AUTO_PUSH=true                  # auto-push profitable changes
PUSH_MIN_ACCURACY=0.60          # minimum accuracy to push
PUSH_MIN_SHARPE=0               # minimum Sharpe to push
```

## Auto-push

When `AUTO_PUSH=true`, Arbos automatically commits and pushes code changes to GitHub after each successful step — **only if** the latest results meet profitability thresholds:

- `accuracy >= PUSH_MIN_ACCURACY` (default 0.60)
- `sharpe >= PUSH_MIN_SHARPE` (default 0)

The agent can also force a push by creating a `.autopush` file (with an optional commit message inside).

Excluded from auto-push: `.env`, `context/`, `logs/`.

---

MIT
