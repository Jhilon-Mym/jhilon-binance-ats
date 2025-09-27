// config_api.js - API for config get/set
export async function getConfig() {
  const res = await fetch('/api/config');
  if (!res.ok) throw new Error('Failed to fetch config');
  return await res.json();
}

export async function setConfig(cfg) {
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg)
  });
  if (!res.ok) throw new Error('Failed to set config');
  return await res.json();
}
