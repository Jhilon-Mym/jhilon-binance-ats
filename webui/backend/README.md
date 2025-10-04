Reconcile and Session Authentication

This folder contains the web UI backend for the bot. The manual reconciliation endpoint is protected by session-based login and has a token fallback.

Session-based usage

1. Start the web UI server.
2. Login via the web UI or POST to `/api/login` with a JSON body {"email": "you@example.com", "password": "..."}.
3. The server will set a session cookie. Subsequent POST requests to `/api/reconcile` using the same cookie will be accepted.

Token fallback (RECONCILE_TOKEN)

If you prefer a token-based approach (for automation), set the `RECONCILE_TOKEN` environment variable in the `.env` file. Then send the token in the request body as `{ "token": "<token>" }` or in the request header `X-RECONCILE-TOKEN`.

PowerShell examples

# Login (saves cookie automatically in Invoke-RestMethod)
$creds = @{ email = 'admin@example.com'; password = 'yourpassword' } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/login -Method Post -Body $creds -ContentType 'application/json' -SessionVariable s

# Call reconcile using session cookie saved in $s
Invoke-RestMethod -Uri http://localhost:8000/api/reconcile -Method Post -Body (@{ timeout=20 } | ConvertTo-Json) -ContentType 'application/json' -WebSession $s

# Or call reconcile using token header
Invoke-RestMethod -Uri http://localhost:8000/api/reconcile -Method Post -Body (@{ token = 'mytoken' } | ConvertTo-Json) -ContentType 'application/json' -Headers @{ 'X-RECONCILE-TOKEN' = 'mytoken' }


Production / Security notes
---------------------------

If you plan to run the web UI in a production environment, consider the
following security recommendations:

- FLASK_SECRET: set a strong secret in environment variable `FLASK_SECRET`
	(e.g. a 32+ character random string). This prevents session tampering.

- HTTPS: always serve the UI over HTTPS when exposed to untrusted networks.
	Use a reverse proxy (nginx, Caddy) or platform-provided TLS termination.

- Secure cookies: set secure cookie flags in production. In Flask, you can
	configure `SESSION_COOKIE_SECURE=True` and `SESSION_COOKIE_HTTPONLY=True`.

- Session lifetime: set a reasonable session lifetime to reduce risk of
	stale sessions. Example: `app.permanent_session_lifetime = timedelta(hours=2)`
	(the project sets a session on login; consider making sessions `permanent`),
	and require re-login after expiry.

- RECONCILE_TOKEN: if you use token fallback for automation, store it
	securely (not in source control) and rotate it periodically. Treat it like
	any secret API key.

- Logs & secrets: avoid printing API keys or secrets into logs. The UI
	attempts to mask keys in `bot_run.log` where possible, but keep secrets
	out of logs.

Deployment checklist
- Export `FLASK_SECRET` into the environment used by the UI.
- Ensure `.env` contains the correct API keys for testnet/mainnet as needed.
- Use HTTPS and set secure cookie flags in production.

