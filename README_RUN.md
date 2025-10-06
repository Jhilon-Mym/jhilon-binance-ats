Quick run instructions

1) Create a `.env` file or set environment variables directly.

   - Create a `.env` file in the project root and add any values you need (for example: `LIVE_TRADES=true`, `BINANCE_API_KEY=...`, `BINANCE_API_SECRET=...`).
   - Alternatively, export the required environment variables in your shell or CI pipeline instead of using a `.env` file.

3) Activate your virtualenv and install requirements if needed:

   .venv\\.venv\\Scripts\\Activate.ps1
   python -m pip install -r requirements.txt

4) Run bot in paper mode (default in the `scripts/run_bot.py`):

   python scripts/run_bot.py

For testing, you can run the unit tests:

   python -m pytest -q
