//+------------------------------------------------------------------+
//|                                          BG_MultiExecutor.mq5    |
//|                                  Copyright 2026, Quantum Library |
//|                           Blue Guardian Multi-Account Executor   |
//|                                   v2.1 - Production Ready        |
//+------------------------------------------------------------------+
#property copyright "Quantum Library"
#property link      "https://www.mql5.com"
#property version   "2.10"
#property description "Blue Guardian Multi-Account Signal Executor"
#property description "Production-ready with full safety checks"

//--- Input parameters (SET PER ACCOUNT)
input string   AccountName      = "BG_INSTANT_1";     // Account identifier
input int      MagicNumber      = 100001;             // Unique magic per account
input double   MaxLotSize       = 0.5;                // Maximum lot size
input double   DailyDDLimit     = 4.0;                // Daily drawdown limit %
input double   MaxDDLimit       = 8.0;                // Max drawdown limit %
input int      Slippage         = 50;                 // Max slippage points
input bool     TradeEnabled     = false;              // SAFETY: Must enable manually
input int      CheckInterval    = 10;                 // Signal check interval (seconds)
input double   MaxSpreadPoints  = 100;                // Max spread to allow trade

//--- Global variables
datetime       g_lastCheck = 0;
double         g_startBalance = 0;
double         g_highWaterMark = 0;
double         g_dailyStartBalance = 0;
datetime       g_lastDayReset = 0;
int            g_tradesToday = 0;
bool           g_blocked = false;
string         g_blockReason = "";
string         g_lastSignalTimestamp = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("BLUE GUARDIAN MULTI-EXECUTOR v2.1");
    Print("Account Name: ", AccountName);
    Print("Account ID: ", AccountInfoInteger(ACCOUNT_LOGIN));
    Print("Magic Number: ", MagicNumber);
    Print("Max Lot: ", MaxLotSize);
    Print("Daily DD Limit: ", DailyDDLimit, "%");
    Print("Max DD Limit: ", MaxDDLimit, "%");
    Print("Max Spread: ", MaxSpreadPoints, " points");
    Print("Trading Enabled: ", TradeEnabled);
    Print("========================================");

    g_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    g_highWaterMark = g_startBalance;
    g_dailyStartBalance = g_startBalance;
    g_lastDayReset = TimeCurrent();

    // Safety check
    if(g_startBalance <= 0)
    {
        Print("ERROR: Invalid starting balance. Cannot initialize.");
        return INIT_FAILED;
    }

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("BG Executor shutting down. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check interval
    if(TimeCurrent() - g_lastCheck < CheckInterval) return;
    g_lastCheck = TimeCurrent();

    // Daily reset
    CheckDailyReset();

    // Update high water mark
    UpdateHighWaterMark();

    // Check drawdown limits
    if(!CheckDrawdownLimits())
    {
        if(!g_blocked)
        {
            g_blocked = true;
            Print("BLOCKED: ", g_blockReason);
        }
        return;
    }

    // Read and execute signals
    ReadAndExecuteSignals();
}

//+------------------------------------------------------------------+
//| Daily reset check                                                |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
    MqlDateTime now, last;
    TimeToStruct(TimeCurrent(), now);
    TimeToStruct(g_lastDayReset, last);

    if(now.day != last.day || now.mon != last.mon || now.year != last.year)
    {
        g_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        g_tradesToday = 0;
        g_lastDayReset = TimeCurrent();
        g_blocked = false;
        g_blockReason = "";
        Print("Daily reset. New balance baseline: $", DoubleToString(g_dailyStartBalance, 2));
    }
}

//+------------------------------------------------------------------+
//| Update high water mark for accurate drawdown calculation         |
//+------------------------------------------------------------------+
void UpdateHighWaterMark()
{
    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    if(currentBalance > g_highWaterMark)
    {
        g_highWaterMark = currentBalance;
    }
}

//+------------------------------------------------------------------+
//| Check drawdown limits                                            |
//+------------------------------------------------------------------+
bool CheckDrawdownLimits()
{
    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);

    // Use minimum of balance and equity for safety
    double current = MathMin(currentBalance, equity);

    // Guard against division by zero
    if(g_dailyStartBalance <= 0 || g_highWaterMark <= 0)
    {
        g_blockReason = "Invalid balance baseline - cannot calculate drawdown";
        return false;
    }

    // Daily drawdown check (from day's starting balance)
    double dailyDD = ((g_dailyStartBalance - current) / g_dailyStartBalance) * 100;
    if(dailyDD >= DailyDDLimit * 0.9)  // 90% of limit = warning zone
    {
        g_blockReason = StringFormat("Daily DD %.2f%% approaching limit %.2f%%", dailyDD, DailyDDLimit);
        if(dailyDD >= DailyDDLimit)
        {
            return false;
        }
    }

    // Max drawdown check (from high water mark)
    double maxDD = ((g_highWaterMark - current) / g_highWaterMark) * 100;
    if(maxDD >= MaxDDLimit * 0.9)
    {
        g_blockReason = StringFormat("Max DD %.2f%% approaching limit %.2f%%", maxDD, MaxDDLimit);
        if(maxDD >= MaxDDLimit)
        {
            return false;
        }
    }

    return true;
}

//+------------------------------------------------------------------+
//| Read signals and execute                                         |
//+------------------------------------------------------------------+
void ReadAndExecuteSignals()
{
    // Build signal file path (use common folder for VPS compatibility)
    string signalFile = "signal_" + AccountName + ".json";

    int fileHandle = FileOpen(signalFile, FILE_READ|FILE_TXT|FILE_ANSI|FILE_COMMON);
    if(fileHandle == INVALID_HANDLE)
    {
        // Try standard files folder as fallback
        fileHandle = FileOpen(signalFile, FILE_READ|FILE_TXT|FILE_ANSI);
        if(fileHandle == INVALID_HANDLE) return;
    }

    string json = "";
    while(!FileIsEnding(fileHandle))
    {
        json += FileReadString(fileHandle);
    }
    FileClose(fileHandle);

    if(json == "") return;

    // Check timestamp to avoid duplicate execution
    string signalTimestamp = ExtractStringValue(json, "timestamp");
    if(signalTimestamp == g_lastSignalTimestamp && signalTimestamp != "")
    {
        return;  // Already processed this signal
    }

    // Check for our symbol
    string currentSymbol = _Symbol;
    if(StringFind(json, "\"" + currentSymbol + "\"") < 0) return;

    // Verify magic number matches
    int jsonMagic = (int)ExtractDoubleValue(json, "magic_number");
    if(jsonMagic != 0 && jsonMagic != MagicNumber)
    {
        Print("WARNING: Signal magic ", jsonMagic, " doesn't match our magic ", MagicNumber);
        return;
    }

    // Parse action and confidence
    string action = ExtractStringValue(json, "action");
    double confidence = ExtractDoubleValue(json, "confidence");
    double signalLotSize = ExtractDoubleValue(json, "max_lot_size");

    // Use smaller of signal lot size and our max
    double lotSize = MathMin(signalLotSize > 0 ? signalLotSize : MaxLotSize, MaxLotSize);

    // Check if blocked by brain
    string status = ExtractStringValue(json, "status");
    if(status == "BLOCKED")
    {
        string reason = ExtractStringValue(json, "block_reason");
        Print("Brain blocked: ", reason);
        return;
    }

    // Update last processed timestamp
    g_lastSignalTimestamp = signalTimestamp;

    if(!TradeEnabled)
    {
        Print("DRY RUN [", AccountName, "] >> ", action, " (", DoubleToString(confidence*100, 1), "%)");
        return;
    }

    // Execute based on action
    if(action == "BUY")
    {
        ExecuteTrade(ORDER_TYPE_BUY, lotSize);
    }
    else if(action == "SELL")
    {
        ExecuteTrade(ORDER_TYPE_SELL, lotSize);
    }
}

//+------------------------------------------------------------------+
//| Execute Trade with full safety checks                            |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE type, double lots)
{
    // Check if position already exists for this magic
    if(PositionExists()) return;

    // Symbol trade mode check
    ENUM_SYMBOL_TRADE_MODE tradeMode = (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE);
    if(tradeMode == SYMBOL_TRADE_MODE_DISABLED)
    {
        Print("ERROR: Trading disabled for ", _Symbol);
        return;
    }
    if(tradeMode == SYMBOL_TRADE_MODE_CLOSEONLY)
    {
        Print("ERROR: Symbol ", _Symbol, " is close-only");
        return;
    }

    // Spread check
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    if(spread > MaxSpreadPoints)
    {
        Print("SKIP: Spread ", spread, " exceeds max ", MaxSpreadPoints);
        return;
    }

    // Refresh prices
    MqlTick tick;
    if(!SymbolInfoTick(_Symbol, tick))
    {
        Print("ERROR: Could not get current tick");
        return;
    }

    double price = (type == ORDER_TYPE_BUY) ? tick.ask : tick.bid;

    // Calculate SL/TP based on ATR
    double atr = CalculateATR(14);
    double sl_dist = atr * 1.5;
    double tp_dist = atr * 3.0;

    // Get stop level requirement
    int stopLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double minDistance = stopLevel * point;

    // Ensure SL/TP meet minimum requirements
    if(sl_dist < minDistance) sl_dist = minDistance * 1.5;
    if(tp_dist < minDistance) tp_dist = minDistance * 1.5;

    double sl, tp;
    if(type == ORDER_TYPE_BUY)
    {
        sl = price - sl_dist;
        tp = price + tp_dist;
    }
    else
    {
        sl = price + sl_dist;
        tp = price - tp_dist;
    }

    // Validate and normalize lot size
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lots = MathMax(lots, minLot);
    lots = MathMin(lots, maxLot);
    lots = MathFloor(lots / lotStep) * lotStep;
    lots = NormalizeDouble(lots, 2);

    // Margin check
    double marginRequired;
    if(!OrderCalcMargin(type, _Symbol, lots, price, marginRequired))
    {
        Print("ERROR: Could not calculate margin");
        return;
    }

    double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
    if(marginRequired > freeMargin * 0.8)  // Use max 80% of free margin
    {
        Print("ERROR: Insufficient margin. Required: ", marginRequired, " Free: ", freeMargin);
        return;
    }

    // Get proper filling mode
    ENUM_ORDER_TYPE_FILLING filling = GetFillingMode(_Symbol);

    // Build order request
    MqlTradeRequest request;
    MqlTradeResult result;
    ZeroMemory(request);
    ZeroMemory(result);

    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lots;
    request.type = type;
    request.price = price;
    request.sl = NormalizeDouble(sl, _Digits);
    request.tp = NormalizeDouble(tp, _Digits);
    request.deviation = Slippage;
    request.magic = MagicNumber;
    request.comment = "BG_" + AccountName;
    request.type_time = ORDER_TIME_GTC;
    request.type_filling = filling;

    if(!OrderSend(request, result))
    {
        Print("ERROR [", AccountName, "]: OrderSend failed - ", GetLastError());
        return;
    }

    if(result.retcode == TRADE_RETCODE_DONE)
    {
        string typeStr = (type == ORDER_TYPE_BUY) ? "BUY" : "SELL";
        Print("SUCCESS [", AccountName, "]: ", typeStr, " ", lots, " lots @ ", price);
        Print("  SL: ", sl, " | TP: ", tp);
        g_tradesToday++;
    }
    else
    {
        Print("ERROR [", AccountName, "]: ", result.comment, " (", result.retcode, ")");
    }
}

//+------------------------------------------------------------------+
//| Get appropriate filling mode for symbol                          |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode(string symbol)
{
    uint filling = (uint)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);

    if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
        return ORDER_FILLING_FOK;

    if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
        return ORDER_FILLING_IOC;

    return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Check if position exists for our magic                          |
//+------------------------------------------------------------------+
bool PositionExists()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
               PositionGetString(POSITION_SYMBOL) == _Symbol)
            {
                return true;
            }
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| Calculate ATR with proper fallback                               |
//+------------------------------------------------------------------+
double CalculateATR(int period)
{
    MqlRates rates[];
    ArraySetAsSeries(rates, true);

    int copied = CopyRates(_Symbol, PERIOD_M5, 0, period + 1, rates);
    if(copied < period + 1)
    {
        // Fallback: use spread-based estimate
        double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
        return spread * 10;  // Conservative fallback
    }

    double trSum = 0;
    for(int i = 0; i < period; i++)
    {
        double high = rates[i].high;
        double low = rates[i].low;
        double prevClose = rates[i+1].close;

        double tr = MathMax(high - low,
                   MathMax(MathAbs(high - prevClose),
                          MathAbs(low - prevClose)));
        trSum += tr;
    }

    double atr = trSum / period;

    // Sanity check
    if(atr <= 0)
    {
        double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
        return spread * 10;
    }

    return atr;
}

//+------------------------------------------------------------------+
//| JSON Helpers                                                     |
//+------------------------------------------------------------------+
string ExtractStringValue(string json, string key)
{
    int keyPos = StringFind(json, "\"" + key + "\"");
    if(keyPos < 0) return "";

    int colonPos = StringFind(json, ":", keyPos);
    if(colonPos < 0) return "";

    int valueStart = StringFind(json, "\"", colonPos);
    if(valueStart < 0) return "";

    int valueEnd = StringFind(json, "\"", valueStart + 1);
    if(valueEnd < 0) return "";

    return StringSubstr(json, valueStart + 1, valueEnd - valueStart - 1);
}

double ExtractDoubleValue(string json, string key)
{
    int keyPos = StringFind(json, "\"" + key + "\"");
    if(keyPos < 0) return 0;

    int colonPos = StringFind(json, ":", keyPos);
    if(colonPos < 0) return 0;

    string remaining = StringSubstr(json, colonPos + 1, 50);
    StringTrimLeft(remaining);

    // Find end of number
    int endPos = 0;
    for(int i = 0; i < StringLen(remaining); i++)
    {
        ushort c = StringGetCharacter(remaining, i);
        if((c >= '0' && c <= '9') || c == '.' || c == '-')
        {
            endPos = i + 1;
        }
        else if(endPos > 0)
        {
            break;
        }
    }

    if(endPos > 0)
    {
        return StringToDouble(StringSubstr(remaining, 0, endPos));
    }

    return 0;
}
//+------------------------------------------------------------------+
