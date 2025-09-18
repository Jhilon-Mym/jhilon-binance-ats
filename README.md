# Binance Auto Trading Bot (ATS Pro)

A professional, web-based auto trading bot for Binance with AI/ML signal generation, backtesting, and a secure dashboard UI.

## Features
- Secure login/logout and password change
- Start/stop bot, monitor status, and view logs in real time
- AI/ML-based signal generation (XGBoost, RandomForest, TA)
- Backtesting, profit/loss tracking, and equity curve plotting
- Downloadable profit reports
- Modern, responsive web UI (Flask backend + vanilla JS frontend)

## Quick Start
1. **Clone the repo:**
   ```
   git clone <your-repo-url>
   cd binance_auto_trading_bot-jewel
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Configure credentials:**
   - Edit `webui/backend/user_auth.txt` with your email and password.
   - Set up your Binance API keys in the appropriate config file or environment variables.
4. **Run the server:**
   ```
   cd webui/backend
   set PYTHONPATH=../..; python app.py
   ```
   (On Linux/Mac: `export PYTHONPATH=../.. && python app.py`)
5. **Open the UI:**
   - Go to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Folder Structure
- `src/` — Core trading logic, strategies, AI/ML code
- `webui/backend/` — Flask backend, API endpoints
- `webui/frontend/` — HTML/CSS/JS dashboard
- `tools/` — Utilities, scripts, and helpers
- `training/` — Model training scripts
- `models/` — Model artifacts and scalers
- `tests/` — Unit tests

## License
MIT License

---
For support or contributions, open an issue or pull request on GitHub.
