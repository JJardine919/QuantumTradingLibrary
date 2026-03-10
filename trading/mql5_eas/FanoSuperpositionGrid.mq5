//+------------------------------------------------------------------+
//|                                      FanoSuperpositionGrid.mq5   |
//|          Fano Plane Octonion Superposition Grid Trading System    |
//|                                 QuantumChildren / DooDoo 2026    |
//|                                                                  |
//| 7 Fibonacci lookbacks mapped to Fano plane points (e1..e7).     |
//| Octonion product AB captures lead/lag across all timescales.      |
//| Jordan-Shadow decomposition: consensus vs conflict vs chaos.     |
//| Bayesian pattern association provides 30% weight tie-breaking.   |
//| Fixed $1 SL / $3 TP per position. NOT martingale.                |
//+------------------------------------------------------------------+
#property copyright "QuantumChildren / DooDoo 2026"
#property link      ""
#property version   "1.00"
#property strict

//--- Include chain: Grid needs Trade.mqh, Bayesian needs Regime needs Decomp needs Octonion
#include <Fano\FanoBayesian.mqh>
#include <Fano\FanoGrid.mqh>
#include <Fano\FanoRisk.mqh>

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT ==="
input int      MagicNumber         = 212001;       // Magic Number
input string   AccountName         = "FANO_ATLAS"; // Account Name

input group "=== FANO REGIME ==="
input int      Lookback1           = 5;            // Fano point e1 (bars)
input int      Lookback2           = 8;            // Fano point e2 (bars)
input int      Lookback3           = 13;           // Fano point e3 (bars)
input int      Lookback4           = 21;           // Fano point e4 (bars)
input int      Lookback5           = 34;           // Fano point e5 (bars)
input int      Lookback6           = 55;           // Fano point e6 (bars)
input int      Lookback7           = 89;           // Fano point e7 (bars)
input double   ConfidenceThreshold = 0.30;         // Min regime confidence to trade
input bool     InvertedMode        = false;        // Invert regime signal

input group "=== GRID ==="
input int      MaxPositions        = 5;            // Max grid depth
input double   SLDollars           = 1.00;         // SL per position (sacred)
input double   TPDollars           = 3.00;         // TP per position (3:1 R:R)
input double   DynTPPercent        = 50.0;         // Partial TP percentage
input double   ATRMultiplier       = 0.0438;       // ATR mult for SL distance
input double   RollSLMult          = 1.5;          // Break-even trigger multiplier
input int      ATRPeriod           = 14;           // ATR period

input group "=== BAYESIAN ==="
input double   RegimeWeight        = 0.70;         // Weight for octonion signal
input double   PatternWeight       = 0.30;         // Weight for Bayesian signal
input int      MinPatternSamples   = 20;           // Min samples before pattern votes

input group "=== RISK ==="
input double   DailyDDLimit        = 4.5;          // Daily drawdown limit %
input double   MaxDDLimit          = 9.0;          // Max drawdown limit %

input group "=== MANAGEMENT ==="
input int      CheckInterval       = 30;           // Signal check interval (seconds)
input bool     TradeEnabled        = true;         // Master enable switch

//+------------------------------------------------------------------+
//| GLOBAL OBJECTS                                                    |
//+------------------------------------------------------------------+
CTrade       g_trade;
FanoRegime   g_regime;
FanoBayesian g_bayesian;
FanoGrid     g_grid;
FanoRisk     g_risk;
datetime     g_lastSignalCheck = 0;
string       g_bayesFile       = "";

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("================================================================");
   Print("  FANO SUPERPOSITION GRID EA v1.0");
   Print("  Account: ", AccountName);
   Print("  Symbol:  ", _Symbol);
   Print("  Magic:   ", MagicNumber);
   Print("  Mode:    ", InvertedMode ? "INVERTED" : "NORMAL");
   Print("  SL=$", DoubleToString(SLDollars,2),
         " TP=$", DoubleToString(TPDollars,2),
         " ATRmult=", DoubleToString(ATRMultiplier,4));
   Print("  Max Grid: ", MaxPositions, " | Confidence: ", DoubleToString(ConfidenceThreshold,2));
   Print("  Regime Weight: ", DoubleToString(RegimeWeight,2),
         " | Pattern Weight: ", DoubleToString(PatternWeight,2));
   Print("================================================================");

   //--- Initialize regime detection with custom lookback periods
   int lb[7];
   lb[0] = Lookback1; lb[1] = Lookback2; lb[2] = Lookback3; lb[3] = Lookback4;
   lb[4] = Lookback5; lb[5] = Lookback6; lb[6] = Lookback7;

   if(!g_regime.Init(_Symbol, PERIOD_CURRENT, lb))
   {
      Print("ERROR: FanoRegime Init failed");
      return INIT_FAILED;
   }

   //--- Initialize Bayesian pattern tracker
   g_bayesian.Init(MinPatternSamples);
   g_bayesFile = StringFormat("FanoBayes_%s_%d.bin", _Symbol, MagicNumber);
   if(!g_bayesian.LoadFromFile(g_bayesFile))
      Print("Bayesian cold start - learning from scratch");
   else
      Print("Bayesian counters loaded from ", g_bayesFile);

   //--- Initialize grid
   g_grid.Init(_Symbol, MagicNumber);
   g_grid.SyncWithBroker();

   //--- Initialize risk
   g_risk.Init();

   //--- Configure trade object
   g_trade.SetExpertMagicNumber(MagicNumber);
   g_trade.SetDeviationInPoints(30);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   g_lastSignalCheck = 0;

   Print("FANO EA initialized. Grid synced: ", g_grid.count, " positions.");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   //--- Save Bayesian counters
   if(g_bayesian.SaveToFile(g_bayesFile))
      Print("Bayesian counters saved to ", g_bayesFile);
   else
      Print("WARNING: Failed to save Bayesian counters");

   //--- Release indicator handles
   g_regime.Deinit();

   Print("FANO EA deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- ALWAYS: manage hidden SL/TP/partial on every tick
   g_grid.ManagePositions(g_trade);

   //--- Throttle signal checks
   if(TimeCurrent() - g_lastSignalCheck < CheckInterval)
      return;
   g_lastSignalCheck = TimeCurrent();

   //--- Risk check
   g_risk.CheckDailyReset();
   g_risk.UpdateHWM();
   if(!g_risk.IsSafe(DailyDDLimit, MaxDDLimit))
      return;

   //--- Master switch
   if(!TradeEnabled)
      return;

   //--- Get close prices for Bayesian (need at least 100 bars)
   double close[];
   ArraySetAsSeries(close, true);
   int copied = CopyClose(_Symbol, PERIOD_CURRENT, 0, 200, close);
   if(copied < 100)
      return;

   //--- Get ATR value
   double atr_buf[];
   ArraySetAsSeries(atr_buf, true);
   if(CopyBuffer(g_regime.atr_handle, 0, 0, 1, atr_buf) < 1)
      return;
   double atr_val = atr_buf[0];
   if(atr_val < 1e-10)
      return;

   //--- Update regime detection (full octonion pipeline)
   g_regime.Update(ConfidenceThreshold, InvertedMode);

   //--- Update Bayesian counters (learn from history)
   g_bayesian.UpdateCounters(close, copied);

   //--- Get Bayesian vote from active triple's lookbacks
   double pattern_vote = g_bayesian.GetTripleVote(g_regime.active_triple, close);

   //--- Combine signals: RegimeWeight * octonion + PatternWeight * Bayesian
   double final_signal = RegimeWeight * g_regime.regime_confidence
                       + PatternWeight * (pattern_vote * 2.0);

   //--- Determine direction
   int direction = 0;
   if(MathAbs(final_signal) >= ConfidenceThreshold)
      direction = (final_signal > 0) ? 1 : -1;

   //--- Compute lot and grid distances (needs ATR and chaos level)
   g_grid.ComputeLotAndDistances(atr_val, g_regime.decomp.chaos_level,
                                  SLDollars, TPDollars, DynTPPercent,
                                  ATRMultiplier, RollSLMult);

   //--- Recalculate hidden SL/TP for existing positions after sync
   //    (handles terminal restart case where SL/TP were zeroed)
   for(int i = 0; i < g_grid.count; i++)
   {
      if(g_grid.positions[i].hidden_sl == 0 || g_grid.positions[i].hidden_tp == 0)
      {
         double entry = g_grid.positions[i].entry_price;
         int dir = g_grid.positions[i].direction;
         if(dir == 1)
         {
            g_grid.positions[i].hidden_sl = entry - g_grid.sl_distance;
            g_grid.positions[i].hidden_tp = entry + g_grid.tp_distance;
            g_grid.positions[i].partial_tp = entry + g_grid.partial_tp_distance;
         }
         else
         {
            g_grid.positions[i].hidden_sl = entry + g_grid.sl_distance;
            g_grid.positions[i].hidden_tp = entry - g_grid.tp_distance;
            g_grid.positions[i].partial_tp = entry - g_grid.partial_tp_distance;
         }
      }
   }

   //--- Grid entry logic
   if(direction != 0)
   {
      if(g_grid.count == 0)
      {
         //--- First entry: open initial grid position
         if(g_grid.CanOpenNew(MaxPositions, direction))
            g_grid.OpenPosition(direction, g_trade);
      }
      else if(g_grid.CanOpenNew(MaxPositions, direction))
      {
         //--- Grid averaging: price moved AGAINST us by grid_spacing
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(g_grid.PriceReachedNextLevel(bid, g_grid.positions[0].direction))
            g_grid.OpenPosition(g_grid.positions[0].direction, g_trade);
      }
   }

   //--- Status print
   Print(StringFormat("FANO | Dir:%+d | Conf:%.3f | Triple:T%d | J:%.3f C:%.3f X:%.2f | Grid:%d/%d | Lot:%.4f | DD:%.2f%%/%.2f%%",
         direction, final_signal,
         g_regime.active_triple + 1,
         g_regime.decomp.jordan_strength,
         g_regime.decomp.commutator_strength,
         g_regime.decomp.chaos_level,
         g_grid.count, MaxPositions, g_grid.base_lot,
         g_risk.DailyDD(), g_risk.TotalDD()));
}
//+------------------------------------------------------------------+
