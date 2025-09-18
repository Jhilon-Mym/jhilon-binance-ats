# Binance Bot WebUI (বাংলা)

## দ্রুত শুরু

1. **প্রশিক্ষণ (Train):**
   ```powershell
   & D:/binance_bot_webui_ready/.venv/Scripts/python.exe d:/binance_bot_webui_ready/training/train_model.py
   ```
   এতে `models/model.pkl` ও `models/scaler.pkl` আলাদা তৈরি হবে।

2. **ইন্সপেক্ট/ডিবাগ:**
   ```powershell
   & D:/binance_bot_webui_ready/.venv/Scripts/python.exe d:/binance_bot_webui_ready/tools/inspect_model.py
   & D:/binance_bot_webui_ready/.venv/Scripts/python.exe d:/binance_bot_webui_ready/tools/inspect_labels.py
   & D:/binance_bot_webui_ready/.venv/Scripts/python.exe d:/binance_bot_webui_ready/tools/debug_signal.py
   ```
   - inspect_model.py: model/scaler লোড ও prediction চেক
   - inspect_labels.py: label distribution ও feature summary
   - debug_signal.py: signal/win_prob/strategy ডিবাগ

3. **লাইভ বট চালানো:**
   ```powershell
   & D:/binance_bot_webui_ready/.venv/Scripts/python.exe d:/binance_bot_webui_ready/src/bot.py
   ```
   - .env ফাইলে API key, secret, testnet/live ঠিক করুন

## গুরুত্বপূর্ণ
- `models/model.pkl` ও `models/scaler.pkl` আলাদা ফাইল
- signal/win_prob লজিক টিউন করা হয়েছে (threshold=0.7)
- inspect_model.py ও inspect_labels.py দিয়ে দ্রুত যাচাই করুন
- PowerShell-এ সরাসরি python স্ক্রিপ্ট চালান

## সমস্যা হলে
- .env, requirements.txt, API key/config চেক করুন
- ডিবাগ স্ক্রিপ্ট চালিয়ে আউটপুট দেখুন
- কোনো প্রশ্ন থাকলে, বাংলায় লিখে আমাকে জানান

---

**শুভকামনা! আপনার বট এখন প্রস্তুত!**

# আপডেটেড ট্রেডিং বট (Fixed Build)

এই প্যাকেজে `src/` ফোল্ডারের সম্পূর্ণ ফাইলগুলো আপডেট করা হয়েছে। নিচে কী কী সমস্যা ঠিক করা হয়েছে এবং কীভাবে রান করবেন তা ধাপে ধাপে দেয়া হলো।

## কী কী ফিক্স/উন্নয়ন করা হয়েছে

1) **`win_prob` থ্রেশহোল্ডে আটকে ট্রেড না হওয়া (স্থায়ী সমাধান)**  
   - `src/bot.py` এর `execute_trade()` ফাংশনে AI কনফিডেন্স থ্রেশহোল্ডকে *reason-aware* করা হয়েছে।  
   - এখন `apply_strategy()` যদি `reason='indicators_confirm'` দেয়, তাহলে **AI_MIN_CONFIDENCE_OVERRIDE** দিয়ে আবার ব্লক করা হবে না।  
   - শুধুমাত্র `reason='ai_confident'` হলে সফট থ্রেশহোল্ড (ডিফল্ট 0.75) চেক হবে। ফলে আপনার স্ক্রিনশটে দেখা `win_prob 0.600 < threshold 0.80` কারণে আর Buy/Sell আটকে থাকবে না।

2) **অ্যাকাউন্ট ব্যালেন্স-সেফ ট্রেড এক্সিকিউশন**  
   - BUY করার আগে এখন USDT ফ্রি ব্যালেন্স চেক করা হয় (`MIN_USDT_BAL` এবং `BUY_USDT_PER_TRADE` উভয় কভার করে)।  
   - SELL করার আগে বাস্তবিক BTC ফ্রি ব্যালেন্স চেক করা হয়; এক্সচেঞ্জের `minQty/stepSize` মেনে কিউটি ফরম্যাট করা হয়।  

3) **কম দামে কিনে দাম বাড়লে বিক্রি (Profit-only exit)**  
   - `src/trade_manager.py` এর `ManagedTrade.update()` লজিক আগের মতই **ট্রেইলিং স্টপ + কেবল প্রফিটে ক্লোজ** করে।  
   - SELL সিগনালে বট স্পট ওয়ালেটের `BTC` ফ্রি ব্যালেন্স থেকেই সেল প্লেস করে; নতুন "শর্ট" ওপেন করার চেষ্টা করে না। 

4) **বট রিস্টার্টের পর পেন্ডিং অর্ডার সিংক (স্থায়ী সমাধান)**  
   - `src/utils.py` এ নতুন `reconcile_pending_with_exchange()` মেথড যোগ করা হয়েছে।  
   - বট চালুর সময় (`main()` এর শুরুতে) এটি একবার চালু হয়:  
     - লোকাল `pending_trades.json` এর প্রতিটি ওপেন ট্রেডের `orderId` দিয়ে Binance-এ স্ট্যাটাস চেক করে।  
     - `CANCELED/REJECTED/EXPIRED` হলে লোকাল পেন্ডিং থেকে সরিয়ে দেয়।  
     - `PARTIALLY_FILLED/FILLED` হলে `qty`, `cummulativeQuoteQty` ইত্যাদি দিয়ে লোকাল কপি আপডেট করে।  

5) **লট সাইজ/মিন-কিউটির সাথে কড়া কমপ্লায়েন্স**  
   - BUY/SELL দুই দিকেই কিউটি এক্সচেঞ্জের `LOT_SIZE` (minQty/stepSize) অনুযায়ী `ROUND_DOWN` করা হয়।

## কীভাবে রান করবেন

1. `.env` ফাইল তৈরি/আপডেট করুন (উদাহরণ):
   ```
   BINANCE_API_KEY=YOUR_KEY
   BINANCE_API_SECRET=YOUR_SECRET
   USE_TESTNET=true           # লাইভ মেইননেটে গেলে false করুন
   SYMBOL=BTCUSDT
   BUY_USDT_PER_TRADE=20
   MIN_USDT_BAL=10
   FAST_SMA=9
   SLOW_SMA=21
   ATR_LEN=14
   EMA_HTF=200
   ATR_SL_MULT=1.0
   ATR_TP_MULT=1.25
   AI_MIN_CONFIDENCE_OVERRIDE=0.75
   AI_OVERRIDE_PROB=0.80
   INDICATOR_CONFIRM_COUNT=2
   ```

2. ডিপেন্ডেন্সি ইন্সটল করুন:
   ```
   pip install -r requirements.txt
   ```

3. বট চালান:
   ```
   python -m src.bot
   ```
   - স্টার্টআপে বট প্রথমে **ব্যালেন্স প্রিন্ট** করবে এবং **পেন্ডিং অর্ডার রিকনসাইল** করবে।
   - ওয়েবসকেট ক্যান্ডেল আসলেই স্ট্রাটেজি রান হবে এবং ট্রেড এক্সিকিউট করবে।

> **মেইননেট** চালাতে চাইলে `.env` এ `USE_TESTNET=false` দিন এবং লাইভ API কী ব্যবহার করুন।

## ফাইল লিস্ট

- `src/bot.py` — **আপডেটেড**: ট্রেড এক্সিকিউশন, ব্যালেন্স চেক, থ্রেশহোল্ড ফিক্স, রিস্টার্টে রিকনসাইল কল।
- `src/utils.py` — **আপডেটেড**: `reconcile_pending_with_exchange()` যোগ।
- `src/strategy.py`, `src/hybrid_ai.py`, `src/indicators.py`, `src/trade_manager.py`, `src/config.py`, `src/websocket_bot.py` — অপরিবর্তিত/ছোটখাটো ফরম্যাটিং, কাজ আগের মতই।
- `requirements.txt`

## টিপস

- আপনার স্ক্রিনশটে দেখা `win_prob=0.600` আসে ইন্ডিকেটর-ফলব্যাক থেকে (SMA+MACD)। এখন এর জন্য ট্রেড **ব্লক হবে না** কারণ `reason='indicators_confirm'` হলে সফট থ্রেশহোল্ড আর প্রযোজ্য নয়।  
- যেকোনো সময় লগ/ফাইল: `bot.log`, `pending_trades.json`, `orders.log`, `trade_history.json` দেখে ট্রাবলশুট করতে পারবেন।

---

**শুধু এই ZIP-টা আপনার প্রজেক্টে কপি করে পুরনো `src/` রিপ্লেস করলেই হবে।**
