import React, { useState, useEffect } from 'react';

const DEFAULTS = {
  SYMBOL: 'BTCUSDT',
  INTERVAL: '5m',
  HISTORY_PRELOAD: 600,
  BUY_USDT_PER_TRADE: 10,
  MIN_USDT_BAL: 20,
  MIN_PROFIT_TO_CLOSE: 0.015,
  FAST_SMA: 9,
  SLOW_SMA: 21,
  ATR_LEN: 14,
  EMA_HTF: 200,
  ATR_SL_MULT: 1.0,
  ATR_TP_MULT: 1.0,
  AI_MIN_CONFIDENCE_OVERRIDE: 0.75,
  AI_OVERRIDE_PROB: 0.85,
  INDICATOR_CONFIRM_COUNT: 2,
  MIN_COMBINED_SCORE: 0.05,
  MIN_AI_WEIGHT: 0.0,
  WIN_PROB_MIN: 0.75,
};

const SYMBOL_OPTIONS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'];
const INTERVAL_OPTIONS = ['1m', '5m', '15m', '1h', '4h', '1d'];

const FIELD_META = {
  SYMBOL: { type: 'select', options: SYMBOL_OPTIONS },
  INTERVAL: { type: 'select', options: INTERVAL_OPTIONS },
  HISTORY_PRELOAD: { type: 'int', min: 100, max: 5000 },
  BUY_USDT_PER_TRADE: { type: 'float', min: 1, max: 10000 },
  MIN_USDT_BAL: { type: 'float', min: 1, max: 10000 },
  MIN_PROFIT_TO_CLOSE: { type: 'float', min: 0.001, max: 0.1 },
  FAST_SMA: { type: 'int', min: 2, max: 50 },
  SLOW_SMA: { type: 'int', min: 5, max: 200 },
  ATR_LEN: { type: 'int', min: 5, max: 50 },
  EMA_HTF: { type: 'int', min: 20, max: 500 },
  ATR_SL_MULT: { type: 'float', min: 0.1, max: 10 },
  ATR_TP_MULT: { type: 'float', min: 0.1, max: 10 },
  AI_MIN_CONFIDENCE_OVERRIDE: { type: 'float', min: 0.5, max: 1.0 },
  AI_OVERRIDE_PROB: { type: 'float', min: 0.5, max: 1.0 },
  INDICATOR_CONFIRM_COUNT: { type: 'int', min: 1, max: 5 },
  MIN_COMBINED_SCORE: { type: 'float', min: 0.0, max: 1.0 },
  MIN_AI_WEIGHT: { type: 'float', min: 0.0, max: 1.0 },
  WIN_PROB_MIN: { type: 'float', min: 0.5, max: 1.0 },
};

export default function ConfigModal({
  isOpen, onClose, onApply, onRestore, isBotRunning, currentConfig
}) {
  const [values, setValues] = useState({ ...DEFAULTS, ...currentConfig });
  const [dirty, setDirty] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [restoreFields, setRestoreFields] = useState(Object.keys(DEFAULTS));
  const [showRestore, setShowRestore] = useState(false);

  useEffect(() => {
    setValues({ ...DEFAULTS, ...currentConfig });
    setDirty(false);
  }, [isOpen, currentConfig]);

  const handleChange = (k, v) => {
    setValues({ ...values, [k]: v });
    setDirty(true);
  };

  const validate = (k, v) => {
    const meta = FIELD_META[k];
    if (!meta) return true;
    if (meta.type === 'int') {
      if (!Number.isInteger(Number(v))) return false;
      if (meta.min !== undefined && Number(v) < meta.min) return false;
      if (meta.max !== undefined && Number(v) > meta.max) return false;
    } else if (meta.type === 'float') {
      if (isNaN(Number(v))) return false;
      if (meta.min !== undefined && Number(v) < meta.min) return false;
      if (meta.max !== undefined && Number(v) > meta.max) return false;
    }
    return true;
  };

  const handleApply = () => setShowConfirm(true);
  const confirmApply = () => { setShowConfirm(false); onApply(values); };
  const cancelApply = () => setShowConfirm(false);

  const handleRestore = () => setShowRestore(true);
  const confirmRestore = () => {
    const restored = { ...values };
    restoreFields.forEach(f => { restored[f] = DEFAULTS[f]; });
    setValues(restored);
    setShowRestore(false);
    setDirty(true);
  };
  const cancelRestore = () => setShowRestore(false);

  if (!isOpen) return null;

  return (
    <div className="modal config-modal">
      <h2>Configuration</h2>
      <form>
        {Object.keys(DEFAULTS).map(k => (
          <div key={k} className="form-row">
            <label>{k}
              <span className="meta">{FIELD_META[k]?.min !== undefined ? `Min: ${FIELD_META[k].min}` : ''} {FIELD_META[k]?.max !== undefined ? `Max: ${FIELD_META[k].max}` : ''}</span>
            </label>
            {FIELD_META[k]?.type === 'select' ? (
              <select value={values[k]} onChange={e => handleChange(k, e.target.value)} disabled={isBotRunning}>
                {FIELD_META[k].options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
              </select>
            ) : (
              <input
                type="number"
                value={values[k]}
                min={FIELD_META[k]?.min}
                max={FIELD_META[k]?.max}
                step={FIELD_META[k]?.type === 'int' ? 1 : 0.001}
                onChange={e => handleChange(k, e.target.value)}
                disabled={isBotRunning}
                className={validate(k, values[k]) ? '' : 'invalid'}
              />
            )}
          </div>
        ))}
      </form>
      <div className="button-row">
        <button onClick={handleApply} disabled={!dirty || isBotRunning}>Apply</button>
        <button onClick={handleRestore} disabled={!dirty || isBotRunning}>Restore Default</button>
        <button onClick={onClose}>Cancel</button>
      </div>
      {showConfirm && (
        <div className="modal confirm-modal">
          <p>Are you sure you want to apply these changes?</p>
          <button onClick={confirmApply}>OK</button>
          <button onClick={cancelApply}>Cancel</button>
        </div>
      )}
      {showRestore && (
        <div className="modal restore-modal">
          <p>Select fields to restore to default:</p>
          <div>
            <label><input type="checkbox" checked={restoreFields.length === Object.keys(DEFAULTS).length} onChange={e => setRestoreFields(e.target.checked ? Object.keys(DEFAULTS) : [])}/> Select All</label>
          </div>
          {Object.keys(DEFAULTS).map(k => (
            <div key={k}>
              <label><input type="checkbox" checked={restoreFields.includes(k)} onChange={e => setRestoreFields(e.target.checked ? [...restoreFields, k] : restoreFields.filter(f => f !== k))}/> {k}</label>
            </div>
          ))}
          <button onClick={confirmRestore}>OK</button>
          <button onClick={cancelRestore}>Cancel</button>
        </div>
      )}
      {isBotRunning && (
        <div className="modal info-modal">
          <p>First, close the bot. Then try again.</p>
        </div>
      )}
    </div>
  );
}
