## Running

```sh
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the venv and activate it
uv venv 
source .venv/bin/activate

# Install this repo.
uv pip install -e .

# Add your goal with required perms and API keys.
echo "HL_SECRET_KEY=f1...." >> .env
echo "make me money on hyperliquid" >> GOAL.md

# Run the agent.
python3 agent.py
```
