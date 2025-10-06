import logging
import os
import warnings

def configure_logging():
    """Configure root logging for the application.

    Respects LOG_LEVEL env var (default INFO). Ensures the 'safe_client'
    logger inherits the same handlers/level so its messages appear in the
    application logs.
    """
    level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    # Preserve existing handlers if already configured (useful in tests)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    else:
        logging.getLogger().setLevel(level)

    # Ensure our safe_client logger uses the same level and propagates to root
    sc = logging.getLogger('safe_client')
    sc.setLevel(level)
    sc.propagate = True

    # Also make the module-level named logger consistent
    logging.getLogger('src').setLevel(level)

    # Suppress known deprecation warnings originating from the websockets
    # package (used by python-binance). These are noise for our tests and
    # come from third-party dependencies; suppress them specifically.
    try:
        warnings.filterwarnings('ignore', message='.*WebSocketClientProtocol is deprecated.*', module='binance.ws.websocket_api')
        warnings.filterwarnings('ignore', message='.*websockets.legacy is deprecated.*', module='websockets.legacy')
        warnings.filterwarnings('ignore', message='.*websockets.WebSocketClientProtocol is deprecated.*', module='websockets')
    except Exception:
        pass

# call on import to keep behaviour consistent for CLI entrypoints
configure_logging()
