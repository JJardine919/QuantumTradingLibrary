//+------------------------------------------------------------------+
//|                                        BioTransposonEngine.mq5   |
//|        Bio-Transposon Neuro-Structural Trading Engine v2.0        |
//|                                QuantumChildren / DooDoo 2026      |
//|                                                                   |
//| v2.0 additions over v1.0:                                         |
//|   1. ETARE QuantumFusion 8-adapter signal bank                    |
//|      (LSTM, QPE, QuantumLSTM, Volatility, Bars3D,                |
//|       Compression, ETARE Enhanced, SignalFusion)                   |
//|   2. TEQA v3.0 quantum circuit pipeline                           |
//|      (33 TE families, split genome+neural, genomic shock,         |
//|       neural mosaic, consensus voting)                             |
//|   3. MT5 MCP portable design                                      |
//|      (no hardcoded paths, auto-detect terminal/broker,             |
//|       TerminalInfoString for all path resolution)                  |
//|                                                                   |
//| Wired modules (v1.0 retained):                                    |
//|   Bio Transposons: L1_Neuronal, L1_Somatic, RAG_Like,            |
//|                    Hobo, Crypton, Maverick                         |
//|   DMT Bridge: HGH, DMT, Mitochondria, Dopamine, Psilocybin       |
//|   Super Genes: ACTN3, LRP5, CCR5-delta32                          |
//|   ETARE Compression: Regime archiver layer                         |
//|   Methyl Dye: Blue/Red trade record tracking                      |
//|                                                                   |
//| ATR_MULT is HARDCODED. Not an input. Everything derives from it.  |
//| $1.00 max SL per trade. Sacred. Non-negotiable.                   |
//+------------------------------------------------------------------+
#property copyright "QuantumChildren / DooDoo 2026"
#property link      ""
#property version   "2.00"
#property strict
#property description "Bio-Transposon Neuro-Structural Trading Engine v2.0"

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| HARDCODED CONSTANTS -- NOT INPUTS                                 |
//+------------------------------------------------------------------+
#define ATR_MULT           0.0438    // ATR multiplier -- hardcoded, not input
#define MAX_SL_DOLLARS     1.00      // Sacred. $1.00 max loss per trade.
#define TP_MULTIPLIER      3.0       // TP = 3x SL
#define ROLLING_SL_MULT    1.5       // Break-even trigger multiplier
#define PARTIAL_TP_PCT     50.0      // Partial TP percentage
#define ATR_PERIOD_VAL     14        // ATR period
#define MAX_BIO_MODULES    6         // Number of bio transposon modules
#define MAX_DMT_PATHWAYS   5         // Number of DMT bridge pathways
#define MAX_SUPER_GENES    3         // Number of super gene inputs
#define ETARE_LAYERS       5         // ETARE compression depth
#define ETARE_FID_THRESH   0.90      // Fidelity threshold for compression
#define DYES_BLUE          0         // Methyl blue = winning/trending
#define DYES_RED           1         // Methyl red = losing/choppy

// TEQA v3.0 constants
#define N_TE_FAMILIES      33        // Total TE families (25 original + 8 neural)
#define N_QUBITS_GENOME    25        // Original TE genome qubits
#define N_QUBITS_NEURAL    8         // Neural TE compartment qubits
#define N_MOSAIC_NEURONS   7         // Neural mosaic population size
#define NEURAL_CONSENSUS   0.70      // Neural consensus threshold
#define SHOCK_LOW          0.8       // Below: calm market
#define SHOCK_NORMAL       1.2       // Normal range
#define SHOCK_HIGH         2.0       // Genomic shock
#define SHOCK_EXTREME      3.0       // TRIM28 emergency suppression

// QuantumFusion adapter count
#define N_QF_ADAPTERS      8         // Number of QuantumFusion adapters

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== INSTANCE ==="
input int      MagicNumber         = 300001;       // Magic Number (unique per instance)
input string   InstanceName        = "BTE_001";    // Instance Name

input group "=== BIO TRANSPOSON WEIGHTS ==="
input double   W_L1_Neuronal       = 0.20;         // L1 Neuronal weight
input double   W_L1_Somatic        = 0.15;         // L1 Somatic weight
input double   W_RAG_Like          = 0.20;         // RAG-Like weight
input double   W_Hobo              = 0.10;         // Hobo weight
input double   W_Crypton           = 0.15;         // Crypton weight
input double   W_Maverick          = 0.20;         // Maverick weight

input group "=== DMT BRIDGE ==="
input double   W_DMT_HGH           = 0.25;         // HGH growth amplification
input double   W_DMT_DMT           = 0.20;         // DMT pineal gating
input double   W_DMT_Mito          = 0.20;         // Mitochondrial ATP bias
input double   W_DMT_Dopamine      = 0.20;         // Dopamine reward signal
input double   W_DMT_Psilocybin    = 0.15;         // Psilocybin network reset

input group "=== SUPER GENES ==="
input double   W_ACTN3             = 0.40;         // ACTN3 muscle/momentum
input double   W_LRP5              = 0.35;         // LRP5 bone density / S/R
input double   W_CCR5              = 0.25;         // CCR5-delta32 risk filter

input group "=== SIGNAL FUSION LAYER WEIGHTS ==="
input double   BioWeight           = 0.25;         // Bio transposon composite
input double   DMTWeight           = 0.15;         // DMT bridge composite
input double   GeneWeight          = 0.10;         // Super gene composite
input double   ETAREWeight         = 0.10;         // ETARE compression regime
input double   QFusionWeight       = 0.20;         // QuantumFusion 8-adapter bank
input double   TEQAWeight          = 0.20;         // TEQA v3.0 circuit pipeline
input double   ConfidenceThreshold = 0.30;         // Min confidence to trade

input group "=== GRID ==="
input int      MaxPositions        = 5;            // Max grid depth
input int      CheckInterval       = 30;           // Signal check interval (seconds)
input bool     TradeEnabled        = true;         // Master enable switch

input group "=== RISK ==="
input double   DailyDDLimit        = 4.5;          // Daily drawdown limit %
input double   MaxDDLimit          = 9.0;          // Max drawdown limit %

//+------------------------------------------------------------------+
//| DATA STRUCTURES                                                   |
//+------------------------------------------------------------------+

// Bio transposon signal
struct BioSignal
{
   double l1_neuronal;
   double l1_somatic;
   double rag_like;
   double hobo;
   double crypton;
   double maverick;
   double composite;
};

// DMT bridge signal
struct DMTSignal
{
   double hgh;
   double dmt;
   double mitochondria;
   double dopamine;
   double psilocybin;
   double composite;
};

// Super gene signal
struct GeneSignal
{
   double actn3;
   double lrp5;
   double ccr5;
   double composite;
};

// ETARE compression state
struct ETAREState
{
   double compression_ratio;
   double regime;            // 1.0 = trending, -1.0 = choppy
   int    layers_compressed;
   double fidelity;
};

// QuantumFusion adapter result
struct QFAdapterResult
{
   string name;
   double signal;           // -1.0 to +1.0
   double confidence;       // 0.0 to 1.0
};

// QuantumFusion bank state
struct QFusionState
{
   QFAdapterResult adapters[8];  // 8 adapter results
   double fused_signal;          // Weighted fusion output
   double fused_confidence;      // Combined confidence
   string regime;                // "TRENDING" or "CHOPPY"
   double compression_ratio;     // From compression adapter
};

// TEQA v3.0 TE activation
struct TEActivation
{
   double strength;         // 0.0 to 1.0
   int    direction;        // -1, 0, +1
   bool   stress_responsive;
   bool   neural_target;
};

// TEQA neural mosaic neuron
struct MosaicNeuron
{
   int    neuron_id;
   int    n_insertions;
   int    insertion_targets[5];
   double insertion_modifiers[5];
   int    vote;             // -1, 0, +1
   double confidence;
};

// TEQA v3.0 pipeline state
struct TEQAState
{
   TEActivation te_activations[33];
   double genome_signal;       // 25-qubit genome result
   double neural_signal;       // 8-qubit neural result
   double shock_level;
   string shock_label;
   MosaicNeuron neurons[7];
   int    consensus_direction;
   double consensus_score;
   double fused_signal;        // Final TEQA output
};

// Grid position record
struct GridPosition
{
   ulong  ticket;
   double entry_price;
   int    direction;         // +1 = BUY, -1 = SELL
   double hidden_sl;
   double hidden_tp;
   double partial_tp;
   double lot;
   bool   partial_taken;
   int    dye_color;
   string dye_tag;
   datetime open_time;
};

// Risk tracking
struct RiskState
{
   double balance_hwm;
   double equity_hwm;
   double daily_start_balance;
   datetime daily_reset_time;
   bool   daily_dd_hit;
   bool   max_dd_hit;
};

// Portable terminal info (MT5 MCP design)
struct TerminalInfo
{
   string broker_name;
   string terminal_path;
   string data_path;
   string account_server;
   long   account_number;
   string account_company;
   int    build_number;
};

//+------------------------------------------------------------------+
//| GLOBALS                                                           |
//+------------------------------------------------------------------+
CTrade         g_trade;
GridPosition   g_positions[];
int            g_pos_count      = 0;
int            g_atr_handle     = INVALID_HANDLE;
int            g_rsi_handle     = INVALID_HANDLE;
int            g_ma_fast_handle = INVALID_HANDLE;
int            g_ma_slow_handle = INVALID_HANDLE;
int            g_stoch_handle   = INVALID_HANDLE;
int            g_cci_handle     = INVALID_HANDLE;
int            g_adx_handle     = INVALID_HANDLE;
int            g_macd_handle    = INVALID_HANDLE;
int            g_bb_handle      = INVALID_HANDLE;
int            g_ma_20_handle   = INVALID_HANDLE;
datetime       g_last_check     = 0;
RiskState      g_risk;
ETAREState     g_etare;
QFusionState   g_qfusion;
TEQAState      g_teqa;
TerminalInfo   g_terminal;
int            g_blue_count     = 0;
int            g_red_count      = 0;
double         g_sl_distance    = 0;
double         g_tp_distance    = 0;
double         g_partial_dist   = 0;
double         g_grid_spacing   = 0;
double         g_base_lot       = 0;
uint           g_mosaic_seed    = 0;  // TEQA mosaic RNG seed

//+------------------------------------------------------------------+
//| PORTABLE DESIGN: Auto-detect terminal environment                 |
//+------------------------------------------------------------------+
void DetectTerminalEnvironment()
{
   g_terminal.broker_name     = AccountInfoString(ACCOUNT_COMPANY);
   g_terminal.terminal_path   = TerminalInfoString(TERMINAL_PATH);
   g_terminal.data_path       = TerminalInfoString(TERMINAL_DATA_PATH);
   g_terminal.account_server  = AccountInfoString(ACCOUNT_SERVER);
   g_terminal.account_number  = AccountInfoInteger(ACCOUNT_LOGIN);
   g_terminal.account_company = AccountInfoString(ACCOUNT_COMPANY);
   g_terminal.build_number    = (int)TerminalInfoInteger(TERMINAL_BUILD);

   Print("================================================================");
   Print("  PORTABLE TERMINAL DETECTION");
   Print("  Broker:   ", g_terminal.broker_name);
   Print("  Server:   ", g_terminal.account_server);
   Print("  Account:  ", g_terminal.account_number);
   Print("  Terminal:  ", g_terminal.terminal_path);
   Print("  Data:      ", g_terminal.data_path);
   Print("  Build:     ", g_terminal.build_number);
   Print("================================================================");
}

//+------------------------------------------------------------------+
//| PORTABLE: Get filling mode that works on this broker              |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetSupportedFilling()
{
   long filling_mode = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);

   if((filling_mode & SYMBOL_FILLING_FOK) != 0)
      return ORDER_FILLING_FOK;
   if((filling_mode & SYMBOL_FILLING_IOC) != 0)
      return ORDER_FILLING_IOC;

   return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   // Portable design: detect environment first
   DetectTerminalEnvironment();

   Print("================================================================");
   Print("  BIO-TRANSPOSON ENGINE v2.0");
   Print("  Instance: ", InstanceName);
   Print("  Symbol:   ", _Symbol);
   Print("  Magic:    ", MagicNumber);
   Print("  ATR_MULT: ", DoubleToString(ATR_MULT, 4), " (HARDCODED)");
   Print("  SL=$", DoubleToString(MAX_SL_DOLLARS, 2), " (SACRED)");
   Print("  Max Grid: ", MaxPositions);
   Print("  Bio:", DoubleToString(BioWeight,2),
         " DMT:", DoubleToString(DMTWeight,2),
         " Gene:", DoubleToString(GeneWeight,2),
         " ETARE:", DoubleToString(ETAREWeight,2),
         " QFusion:", DoubleToString(QFusionWeight,2),
         " TEQA:", DoubleToString(TEQAWeight,2));
   Print("================================================================");

   // Initialize indicator handles
   g_atr_handle     = iATR(_Symbol, PERIOD_CURRENT, ATR_PERIOD_VAL);
   g_rsi_handle     = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
   g_ma_fast_handle = iMA(_Symbol, PERIOD_CURRENT, 8, 0, MODE_EMA, PRICE_CLOSE);
   g_ma_slow_handle = iMA(_Symbol, PERIOD_CURRENT, 21, 0, MODE_EMA, PRICE_CLOSE);
   g_stoch_handle   = iStochastic(_Symbol, PERIOD_CURRENT, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
   g_cci_handle     = iCCI(_Symbol, PERIOD_CURRENT, 20, PRICE_TYPICAL);
   g_adx_handle     = iADX(_Symbol, PERIOD_CURRENT, 14);
   g_macd_handle    = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
   g_bb_handle      = iBands(_Symbol, PERIOD_CURRENT, 20, 0, 2.0, PRICE_CLOSE);
   g_ma_20_handle   = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_SMA, PRICE_CLOSE);

   if(g_atr_handle == INVALID_HANDLE || g_rsi_handle == INVALID_HANDLE ||
      g_ma_fast_handle == INVALID_HANDLE || g_ma_slow_handle == INVALID_HANDLE ||
      g_stoch_handle == INVALID_HANDLE || g_cci_handle == INVALID_HANDLE ||
      g_adx_handle == INVALID_HANDLE || g_macd_handle == INVALID_HANDLE ||
      g_bb_handle == INVALID_HANDLE || g_ma_20_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles");
      return INIT_FAILED;
   }

   // Portable filling mode
   g_trade.SetExpertMagicNumber(MagicNumber);
   g_trade.SetDeviationInPoints(30);
   g_trade.SetTypeFilling(GetSupportedFilling());

   // Initialize risk state
   g_risk.balance_hwm         = AccountInfoDouble(ACCOUNT_BALANCE);
   g_risk.equity_hwm          = AccountInfoDouble(ACCOUNT_EQUITY);
   g_risk.daily_start_balance = g_risk.balance_hwm;
   g_risk.daily_reset_time    = GetDailyResetTime();
   g_risk.daily_dd_hit        = false;
   g_risk.max_dd_hit          = false;

   // Initialize ETARE state
   g_etare.compression_ratio  = 1.0;
   g_etare.regime             = 0.0;
   g_etare.layers_compressed  = 0;
   g_etare.fidelity           = 1.0;

   // Initialize QFusion state
   g_qfusion.fused_signal     = 0.0;
   g_qfusion.fused_confidence = 0.0;
   g_qfusion.regime           = "UNKNOWN";
   g_qfusion.compression_ratio = 1.0;

   // Initialize TEQA state
   g_teqa.genome_signal          = 0.0;
   g_teqa.neural_signal          = 0.0;
   g_teqa.shock_level            = 1.0;
   g_teqa.shock_label            = "NORMAL";
   g_teqa.consensus_direction    = 0;
   g_teqa.consensus_score        = 0.0;
   g_teqa.fused_signal           = 0.0;

   // Initialize TEQA mosaic neurons
   g_mosaic_seed = (uint)TimeCurrent();
   InitMosaicNeurons();

   // Sync existing positions
   SyncPositions();

   g_last_check = 0;
   Print("BTE v2.0 initialized. Synced: ", g_pos_count, " positions.");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_atr_handle != INVALID_HANDLE)     IndicatorRelease(g_atr_handle);
   if(g_rsi_handle != INVALID_HANDLE)     IndicatorRelease(g_rsi_handle);
   if(g_ma_fast_handle != INVALID_HANDLE) IndicatorRelease(g_ma_fast_handle);
   if(g_ma_slow_handle != INVALID_HANDLE) IndicatorRelease(g_ma_slow_handle);
   if(g_stoch_handle != INVALID_HANDLE)   IndicatorRelease(g_stoch_handle);
   if(g_cci_handle != INVALID_HANDLE)     IndicatorRelease(g_cci_handle);
   if(g_adx_handle != INVALID_HANDLE)     IndicatorRelease(g_adx_handle);
   if(g_macd_handle != INVALID_HANDLE)    IndicatorRelease(g_macd_handle);
   if(g_bb_handle != INVALID_HANDLE)      IndicatorRelease(g_bb_handle);
   if(g_ma_20_handle != INVALID_HANDLE)   IndicatorRelease(g_ma_20_handle);

   PrintMethylDyeSummary();
   Print("BTE v2.0 deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // ALWAYS manage positions on every tick
   ManagePositions();

   // Throttle signal checks
   if(TimeCurrent() - g_last_check < CheckInterval)
      return;
   g_last_check = TimeCurrent();

   // Risk checks
   CheckDailyReset();
   UpdateHWM();
   if(!IsRiskSafe())
      return;

   if(!TradeEnabled)
      return;

   // Get market data
   double atr_val = GetATR();
   if(atr_val < 1e-10)
      return;

   double close[];
   ArraySetAsSeries(close, true);
   int copied = CopyClose(_Symbol, PERIOD_CURRENT, 0, 200, close);
   if(copied < 100)
      return;

   double high[];
   ArraySetAsSeries(high, true);
   CopyHigh(_Symbol, PERIOD_CURRENT, 0, 200, high);

   double low[];
   ArraySetAsSeries(low, true);
   CopyLow(_Symbol, PERIOD_CURRENT, 0, 200, low);

   double open_arr[];
   ArraySetAsSeries(open_arr, true);
   CopyOpen(_Symbol, PERIOD_CURRENT, 0, 200, open_arr);

   long volume[];
   ArraySetAsSeries(volume, true);
   CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, 200, volume);

   // Compute lot and distances
   ComputeLotAndDistances(atr_val);
   RepairHiddenLevels();

   //--- LAYER 1: BIO TRANSPOSON SIGNALS (v1.0) ---
   BioSignal bio = ComputeBioSignals(close, copied, atr_val);

   //--- LAYER 2: DMT BRIDGE SIGNALS (v1.0) ---
   DMTSignal dmt = ComputeDMTSignals(close, copied, atr_val);

   //--- LAYER 3: SUPER GENE SIGNALS (v1.0) ---
   GeneSignal gene = ComputeGeneSignals(close, copied, atr_val);

   //--- LAYER 4: ETARE COMPRESSION (v1.0) ---
   UpdateETARE(close, copied, atr_val);

   //--- LAYER 5: QUANTUMFUSION 8-ADAPTER BANK (v2.0 NEW) ---
   ComputeQuantumFusion(close, high, low, open_arr, volume, copied, atr_val);

   //--- LAYER 6: TEQA v3.0 CIRCUIT PIPELINE (v2.0 NEW) ---
   ComputeTEQA(close, high, low, open_arr, volume, copied, atr_val);

   //--- MASTER SIGNAL FUSION ---
   double final_signal = BioWeight    * bio.composite
                       + DMTWeight    * dmt.composite
                       + GeneWeight   * gene.composite
                       + ETAREWeight  * g_etare.regime
                       + QFusionWeight * g_qfusion.fused_signal
                       + TEQAWeight   * g_teqa.fused_signal;

   // Determine direction
   int direction = 0;
   if(MathAbs(final_signal) >= ConfidenceThreshold)
      direction = (final_signal > 0) ? 1 : -1;

   //--- GRID ENTRY LOGIC ---
   if(direction != 0)
   {
      if(g_pos_count == 0)
      {
         if(g_pos_count < MaxPositions)
            OpenGridPosition(direction, final_signal, bio, dmt, gene);
      }
      else if(g_pos_count < MaxPositions && g_positions[0].direction == direction)
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(PriceReachedNextLevel(bid))
            OpenGridPosition(g_positions[0].direction, final_signal, bio, dmt, gene);
      }
   }

   // Status print
   string regime_str = (g_etare.regime > 0.3) ? "TREND" :
                        (g_etare.regime < -0.3) ? "CHOP" : "MIXED";
   Print(StringFormat("BTE2 | Dir:%+d | Sig:%.3f | Bio:%.3f DMT:%.3f Gene:%.3f ETARE:%.2f(%s) QF:%.3f TEQA:%.3f(%s) | Grid:%d/%d | B:%d R:%d",
         direction, final_signal,
         bio.composite, dmt.composite, gene.composite,
         g_etare.regime, regime_str,
         g_qfusion.fused_signal, g_teqa.fused_signal, g_teqa.shock_label,
         g_pos_count, MaxPositions,
         g_blue_count, g_red_count));
}

//+------------------------------------------------------------------+
//| BIO TRANSPOSON MODULE: L1 Neuronal (from v1.0)                   |
//+------------------------------------------------------------------+
double ComputeL1Neuronal(const double &close[], int count)
{
   int regions[] = {5, 8, 13, 21, 34};
   double region_signals[];
   ArrayResize(region_signals, 5);

   for(int i = 0; i < 5; i++)
   {
      if(count < regions[i] + 1) { region_signals[i] = 0; continue; }
      double roc = (close[0] - close[regions[i]]) / close[regions[i]];
      double methylation = 0.35 + (0.45 * i / 4.0);
      region_signals[i] = roc * (1.0 - methylation);
   }

   double mean_sig = 0;
   for(int i = 0; i < 5; i++) mean_sig += region_signals[i];
   mean_sig /= 5.0;

   double variance = 0;
   for(int i = 0; i < 5; i++)
      variance += (region_signals[i] - mean_sig) * (region_signals[i] - mean_sig);
   variance /= 5.0;

   double weighted = region_signals[0] * 0.35
                   + region_signals[1] * 0.25
                   + region_signals[2] * 0.20
                   + region_signals[3] * 0.12
                   + region_signals[4] * 0.08;

   double mosaicism_boost = MathSqrt(variance) * 10.0;
   return NormalizeSignal(weighted * (1.0 + mosaicism_boost));
}

//+------------------------------------------------------------------+
//| BIO TRANSPOSON MODULE: L1 Somatic                                |
//+------------------------------------------------------------------+
double ComputeL1Somatic(const double &close[], int count, double atr)
{
   if(count < 50 || atr < 1e-10) return 0;

   double highest_20 = close[0], lowest_20 = close[0];
   for(int i = 1; i < 20 && i < count; i++)
   {
      if(close[i] > highest_20) highest_20 = close[i];
      if(close[i] < lowest_20) lowest_20 = close[i];
   }
   double range_20 = highest_20 - lowest_20;
   if(range_20 < 1e-10) return 0;

   double breakout_signal = 0;
   if(close[0] > highest_20 - atr * 0.1)
      breakout_signal = (close[0] - highest_20) / atr;
   else if(close[0] < lowest_20 + atr * 0.1)
      breakout_signal = (close[0] - lowest_20) / atr;

   int confirming_bars = 0;
   for(int i = 1; i <= 5 && i < count; i++)
   {
      if(breakout_signal > 0 && close[i] > close[i+1]) confirming_bars++;
      if(breakout_signal < 0 && close[i] < close[i+1]) confirming_bars++;
   }

   double body = MathAbs(close[0] - close[1]);
   double wick_approx = atr - body;
   double strand_balance = (wick_approx > 0) ? MathMin(body, wick_approx) / MathMax(body, wick_approx) : 0;
   double af = (confirming_bars >= 3) ? 0.5 : confirming_bars / 6.0;
   double quality = (confirming_bars >= 2) ? 1.0 : 0.3;
   if(strand_balance < 0.1) quality *= 0.5;

   return NormalizeSignal(breakout_signal * quality * af);
}

//+------------------------------------------------------------------+
//| BIO TRANSPOSON MODULE: RAG-Like                                  |
//+------------------------------------------------------------------+
double ComputeRAGLike(const double &close[], int count)
{
   if(count < 30) return 0;

   double heptamer_signal = 0;
   {
      int pattern_match = 0;
      for(int i = 0; i < 6; i++)
      {
         double delta = close[i] - close[i+1];
         if(i % 2 == 0 && delta > 0) pattern_match++;
         if(i % 2 == 1 && delta < 0) pattern_match++;
      }
      heptamer_signal = (pattern_match - 3.0) / 3.0;
   }

   double nonamer_signal = 0;
   {
      int ups = 0, downs = 0;
      for(int i = 0; i < 9 && i + 1 < count; i++)
      {
         if(close[i] > close[i+1]) ups++;
         else downs++;
      }
      nonamer_signal = (ups - downs) / 9.0;
   }

   double spacer_12 = 0, spacer_23 = 0;
   if(count >= 13) spacer_12 = (close[0] - close[12]) / (close[12] > 0 ? close[12] : 1.0);
   if(count >= 24) spacer_23 = (close[0] - close[23]) / (close[23] > 0 ? close[23] : 1.0);

   double rule_12_23 = 0;
   if(spacer_12 * spacer_23 > 0) rule_12_23 = (spacer_12 + spacer_23) / 2.0;

   double rag_signal = (heptamer_signal * 0.25 + nonamer_signal * 0.25 + rule_12_23 * 0.50) * 100.0;
   return NormalizeSignal(rag_signal);
}

//+------------------------------------------------------------------+
//| BIO TRANSPOSON MODULE: Hobo                                      |
//+------------------------------------------------------------------+
double ComputeHobo(const double &close[], int count)
{
   if(count < 55) return 0;
   double hobo_roc = (count >= 9) ? (close[0] - close[8]) / (close[8] > 0 ? close[8] : 1.0) : 0;
   double activator_roc = (count >= 22) ? (close[0] - close[21]) / (close[21] > 0 ? close[21] : 1.0) : 0;
   double tam3_roc = (count >= 56) ? (close[0] - close[55]) / (close[55] > 0 ? close[55] : 1.0) : 0;

   int agreement = 0;
   if(hobo_roc > 0 && activator_roc > 0 && tam3_roc > 0) agreement = 1;
   if(hobo_roc < 0 && activator_roc < 0 && tam3_roc < 0) agreement = -1;

   double cross_mob = 0;
   if(agreement != 0)
      cross_mob = (hobo_roc * 0.50 + activator_roc * 0.30 + tam3_roc * 0.20) * 100.0;
   else
   {
      if(hobo_roc * activator_roc > 0)
         cross_mob = (hobo_roc * 0.60 + activator_roc * 0.40) * 50.0;
      else if(hobo_roc * tam3_roc > 0)
         cross_mob = (hobo_roc * 0.60 + tam3_roc * 0.40) * 30.0;
   }
   return NormalizeSignal(cross_mob);
}

//+------------------------------------------------------------------+
//| BIO TRANSPOSON MODULE: Crypton                                   |
//+------------------------------------------------------------------+
double ComputeCrypton()
{
   double rsi_buf[], stoch_k[], cci_buf[], adx_buf[];
   ArraySetAsSeries(rsi_buf, true); ArraySetAsSeries(stoch_k, true);
   ArraySetAsSeries(cci_buf, true); ArraySetAsSeries(adx_buf, true);

   if(CopyBuffer(g_rsi_handle, 0, 0, 3, rsi_buf) < 3) return 0;
   if(CopyBuffer(g_stoch_handle, 0, 0, 3, stoch_k) < 3) return 0;
   if(CopyBuffer(g_cci_handle, 0, 0, 3, cci_buf) < 3) return 0;
   if(CopyBuffer(g_adx_handle, 0, 0, 3, adx_buf) < 3) return 0;

   double r1 = (rsi_buf[0] - 50) / 50.0;
   double h = (stoch_k[0] - 50) / 50.0;
   double r2 = MathMax(-1.0, MathMin(1.0, cci_buf[0] / 200.0));

   double di_plus[], di_minus[];
   ArraySetAsSeries(di_plus, true); ArraySetAsSeries(di_minus, true);
   CopyBuffer(g_adx_handle, 1, 0, 1, di_plus);
   CopyBuffer(g_adx_handle, 2, 0, 1, di_minus);

   double adx_direction = 0;
   if(di_plus[0] > di_minus[0]) adx_direction = 1.0;
   if(di_plus[0] < di_minus[0]) adx_direction = -1.0;
   double y = adx_direction * MathMin(1.0, adx_buf[0] / 50.0);

   int bullish = 0, bearish = 0;
   if(r1 > 0) bullish++; else if(r1 < 0) bearish++;
   if(h > 0)  bullish++; else if(h < 0)  bearish++;
   if(r2 > 0) bullish++; else if(r2 < 0) bearish++;
   if(y > 0)  bullish++; else if(y < 0)  bearish++;

   double tetrad_integrity = 0;
   if(bullish == 4) tetrad_integrity = 1.0;
   else if(bearish == 4) tetrad_integrity = -1.0;
   else if(bullish == 3) tetrad_integrity = 0.5;
   else if(bearish == 3) tetrad_integrity = -0.5;

   double avg_signal = (r1 + h + r2 + y) / 4.0;
   return NormalizeSignal(avg_signal * MathAbs(tetrad_integrity));
}

//+------------------------------------------------------------------+
//| BIO TRANSPOSON MODULE: Maverick                                  |
//+------------------------------------------------------------------+
double ComputeMaverick(const double &close[], int count, double atr)
{
   if(count < 50 || atr < 1e-10) return 0;

   double high_30 = close[0], low_30 = close[0];
   for(int i = 1; i < 30 && i < count; i++)
   {
      if(close[i] > high_30) high_30 = close[i];
      if(close[i] < low_30) low_30 = close[i];
   }
   double range_30 = high_30 - low_30;
   if(range_30 < 1e-10) return 0;

   double state = 0, distance_from_range = 0;
   if(close[0] > high_30)
   {
      distance_from_range = (close[0] - high_30) / atr;
      state = 1.0;
   }
   else if(close[0] < low_30)
   {
      distance_from_range = (low_30 - close[0]) / atr;
      state = -1.0;
   }
   else
   {
      double pos = (close[0] - low_30) / range_30;
      if(pos > 0.80) { double mom = (close[0] - close[3]) / atr; state = mom * 0.3; }
      else if(pos < 0.20) { double mom = (close[3] - close[0]) / atr; state = -mom * 0.3; }
   }

   double recent_spread = MathAbs(close[0] - close[1]);
   double avg_spread = 0;
   for(int i = 1; i < 10 && i + 1 < count; i++)
      avg_spread += MathAbs(close[i] - close[i+1]);
   avg_spread /= 9.0;

   double capsid_factor = (avg_spread > 0) ? recent_spread / avg_spread : 1.0;
   capsid_factor = MathMin(3.0, capsid_factor);

   double virophage = state * MathMin(1.0, capsid_factor * 0.5);
   if(MathAbs(state) > 0.5)
      virophage *= (1.0 + distance_from_range * 0.2);

   return NormalizeSignal(virophage);
}

//+------------------------------------------------------------------+
//| DMT BRIDGE MODULES (from v1.0)                                   |
//+------------------------------------------------------------------+
double ComputeHGH(const double &close[], int count, double atr)
{
   if(count < 34 || atr < 1e-10) return 0;
   double velocity_8  = (close[0] - close[8]) / atr;
   double velocity_21 = (close[0] - close[21]) / atr;
   double acceleration = velocity_8 - velocity_21;
   return NormalizeSignal(acceleration * 0.5);
}

double ComputeDMT(double bio_composite, double gene_composite)
{
   double intensity = MathAbs(bio_composite) + MathAbs(gene_composite);
   double threshold = 0.3;
   if(intensity < threshold) return 0;
   double direction = (bio_composite + gene_composite > 0) ? 1.0 : -1.0;
   double breakthrough = (intensity - threshold) / (1.0 - threshold);
   return NormalizeSignal(direction * breakthrough);
}

double ComputeMitochondria(const double &close[], int count, double atr)
{
   if(count < 20 || atr < 1e-10) return 0;
   double directional_move = MathAbs(close[0] - close[20]);
   double total_path = 0;
   for(int i = 0; i < 20 && i + 1 < count; i++)
      total_path += MathAbs(close[i] - close[i+1]);
   if(total_path < 1e-10) return 0;
   double efficiency = directional_move / total_path;
   double atp_bias = (efficiency - 0.5) * 2.0;
   double direction = (close[0] > close[20]) ? 1.0 : -1.0;
   return NormalizeSignal(direction * MathAbs(atp_bias));
}

double ComputeDopamine()
{
   int wins = 0, losses = 0;
   for(int i = 0; i < g_pos_count; i++)
   {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double pnl = (g_positions[i].direction == 1) ?
                   (bid - g_positions[i].entry_price) :
                   (g_positions[i].entry_price - bid);
      if(pnl > 0) wins++; else losses++;
   }
   if(wins + losses == 0) return 0;
   double ratio = (double)wins / (wins + losses);
   return NormalizeSignal((ratio - 0.5) * 2.0 * 0.5);
}

double ComputePsilocybin(const double &close[], int count, double atr)
{
   if(count < 50 || atr < 1e-10) return 0;
   double vol_recent = 0, vol_historical = 0;
   for(int i = 0; i < 10 && i + 1 < count; i++)
      vol_recent += MathAbs(close[i] - close[i+1]);
   vol_recent /= 10.0;
   for(int i = 10; i < 50 && i + 1 < count; i++)
      vol_historical += MathAbs(close[i] - close[i+1]);
   vol_historical /= 40.0;
   if(vol_historical < 1e-10) return 0;

   double novelty = vol_recent / vol_historical;
   double reset_signal = 0;
   if(novelty > 1.3)
   {
      double mom = (close[0] - close[5]) / atr;
      reset_signal = MathMin(1.0, mom) * (novelty - 1.0);
   }
   else if(novelty < 0.7)
      reset_signal = (0.7 - novelty) * 0.5;

   return NormalizeSignal(reset_signal);
}

//+------------------------------------------------------------------+
//| SUPER GENE MODULES (from v1.0)                                   |
//+------------------------------------------------------------------+
double ComputeACTN3(const double &close[], int count, double atr)
{
   if(count < 20 || atr < 1e-10) return 0;
   double mom_5 = (close[0] - close[5]) / atr;
   double mom_20 = (close[0] - close[20]) / atr;
   if(MathAbs(mom_5) > MathAbs(mom_20) * 1.5)
      return NormalizeSignal(mom_5 * 0.5);
   else
      return NormalizeSignal(mom_20 * 0.3);
}

double ComputeLRP5(const double &close[], int count)
{
   if(count < 50) return 0;
   double pivot = 0;
   for(int i = 0; i < 20 && i < count; i++) pivot += close[i];
   pivot /= 20.0;

   int touches = 0;
   double touch_zone = MathAbs(close[0] - pivot) * 0.1;
   if(touch_zone < _Point) touch_zone = _Point * 100;
   for(int i = 0; i < 50 && i < count; i++)
      if(MathAbs(close[i] - pivot) < touch_zone) touches++;

   double density = MathMin(1.0, touches / 15.0);
   double distance = (close[0] - pivot) / (pivot > 0 ? pivot : 1.0) * 100.0;
   if(distance > 0) return NormalizeSignal(density * 0.5);
   else return NormalizeSignal(-density * 0.5);
}

double ComputeCCR5(const double &close[], int count, double atr)
{
   if(count < 30 || atr < 1e-10) return 0;
   int direction_changes = 0;
   for(int i = 0; i < 20 && i + 2 < count; i++)
   {
      double d1 = close[i] - close[i+1];
      double d2 = close[i+1] - close[i+2];
      if(d1 * d2 < 0) direction_changes++;
   }
   double virus_load = direction_changes / 20.0;
   if(virus_load > 0.7) return 0;
   else if(virus_load > 0.4) return NormalizeSignal(0.3 * ((close[0] > close[5]) ? 1.0 : -1.0));
   else return NormalizeSignal(1.0 * ((close[0] > close[10]) ? 1.0 : -1.0));
}

//+------------------------------------------------------------------+
//| SIGNAL COMPOSITE FUNCTIONS                                        |
//+------------------------------------------------------------------+
BioSignal ComputeBioSignals(const double &close[], int count, double atr)
{
   BioSignal bio;
   bio.l1_neuronal = ComputeL1Neuronal(close, count);
   bio.l1_somatic  = ComputeL1Somatic(close, count, atr);
   bio.rag_like    = ComputeRAGLike(close, count);
   bio.hobo        = ComputeHobo(close, count);
   bio.crypton     = ComputeCrypton();
   bio.maverick    = ComputeMaverick(close, count, atr);

   bio.composite = bio.l1_neuronal * W_L1_Neuronal
                 + bio.l1_somatic  * W_L1_Somatic
                 + bio.rag_like    * W_RAG_Like
                 + bio.hobo        * W_Hobo
                 + bio.crypton     * W_Crypton
                 + bio.maverick    * W_Maverick;
   return bio;
}

DMTSignal ComputeDMTSignals(const double &close[], int count, double atr)
{
   DMTSignal dmt;
   dmt.hgh          = ComputeHGH(close, count, atr);
   dmt.mitochondria = ComputeMitochondria(close, count, atr);
   dmt.dopamine     = ComputeDopamine();
   dmt.psilocybin   = ComputePsilocybin(close, count, atr);
   double proxy = dmt.hgh + dmt.mitochondria + dmt.psilocybin;
   dmt.dmt = ComputeDMT(proxy, 0);

   dmt.composite = dmt.hgh          * W_DMT_HGH
                 + dmt.dmt          * W_DMT_DMT
                 + dmt.mitochondria * W_DMT_Mito
                 + dmt.dopamine     * W_DMT_Dopamine
                 + dmt.psilocybin   * W_DMT_Psilocybin;
   return dmt;
}

GeneSignal ComputeGeneSignals(const double &close[], int count, double atr)
{
   GeneSignal gene;
   gene.actn3 = ComputeACTN3(close, count, atr);
   gene.lrp5  = ComputeLRP5(close, count);
   gene.ccr5  = ComputeCCR5(close, count, atr);
   gene.composite = gene.actn3 * W_ACTN3 + gene.lrp5 * W_LRP5 + gene.ccr5 * W_CCR5;
   return gene;
}

//+------------------------------------------------------------------+
//| ETARE COMPRESSION LAYER (from v1.0)                              |
//+------------------------------------------------------------------+
void UpdateETARE(const double &close[], int count, double atr)
{
   if(count < 50 || atr < 1e-10) return;

   double features[8];
   features[0] = (close[0] - close[5]) / atr;
   features[1] = (close[0] - close[13]) / atr;
   features[2] = (close[0] - close[21]) / atr;
   features[3] = (close[0] - close[34]) / atr;

   double vol_5 = 0, vol_13 = 0;
   for(int i = 0; i < 5 && i + 1 < count; i++) vol_5 += MathAbs(close[i] - close[i+1]);
   vol_5 /= 5.0;
   for(int i = 0; i < 13 && i + 1 < count; i++) vol_13 += MathAbs(close[i] - close[i+1]);
   vol_13 /= 13.0;

   features[4] = (vol_13 > 0) ? vol_5 / vol_13 : 1.0;
   features[5] = GetRSI() / 100.0;
   features[6] = GetStochastic() / 100.0;

   double adx_buf[];
   ArraySetAsSeries(adx_buf, true);
   CopyBuffer(g_adx_handle, 0, 0, 1, adx_buf);
   features[7] = adx_buf[0] / 100.0;

   int layers = 0;
   double fidelity = 1.0;

   double mom_mean = (features[0] + features[1] + features[2] + features[3]) / 4.0;
   double mom_var = 0;
   for(int i = 0; i < 4; i++) mom_var += (features[i] - mom_mean) * (features[i] - mom_mean);
   mom_var /= 4.0;

   double alignment_1 = 1.0 / (1.0 + mom_var * 100.0);
   if(alignment_1 > ETARE_FID_THRESH)
   {
      layers++;
      fidelity *= alignment_1;
      double vol_confirms = (features[4] > 0.8 && features[4] < 1.2) ? 0.95 : 0.7;
      if(vol_confirms > ETARE_FID_THRESH)
      {
         layers++;
         fidelity *= vol_confirms;
         double osc_mean = (features[5] + features[6]) / 2.0;
         bool osc_bullish = (osc_mean > 0.55);
         bool osc_bearish = (osc_mean < 0.45);
         bool mom_bullish = (mom_mean > 0);
         bool mom_bearish = (mom_mean < 0);
         double osc_confirms = ((osc_bullish && mom_bullish) || (osc_bearish && mom_bearish)) ? 0.95 : 0.6;
         if(osc_confirms > ETARE_FID_THRESH)
         {
            layers++;
            fidelity *= osc_confirms;
            if(features[7] > 0.25)
            {
               layers++;
               fidelity *= MathMin(1.0, features[7] * 2.0);
            }
         }
      }
   }

   double initial_dims = 8.0;
   double final_dims = initial_dims - layers;
   g_etare.compression_ratio = initial_dims / MathMax(1.0, final_dims);
   g_etare.layers_compressed = layers;
   g_etare.fidelity = fidelity;

   if(g_etare.compression_ratio > 1.5)
   {
      g_etare.regime = mom_mean > 0 ? 1.0 : -1.0;
      g_etare.regime *= MathMin(1.0, (g_etare.compression_ratio - 1.0));
   }
   else if(g_etare.compression_ratio < 1.2)
      g_etare.regime = 0;
   else
      g_etare.regime = mom_mean * 0.3;
}

//+------------------------------------------------------------------+
//| QUANTUMFUSION 8-ADAPTER BANK (v2.0 NEW)                         |
//| Translates logic from the 8 Python adapter modules into MQL5.    |
//| Each adapter produces signal (-1 to +1) and confidence (0 to 1). |
//| SignalFusion merges them with regime-adjusted weighting.          |
//+------------------------------------------------------------------+
void ComputeQuantumFusion(const double &close[], const double &high[],
                           const double &low[], const double &open_arr[],
                           const long &volume[], int count, double atr)
{
   if(count < 100) return;

   // Adapter weights (from signal_fusion.py config defaults)
   double qf_weights[8] = {0.15, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10};

   //--- Adapter 0: LSTM Signal (lstm_adapter.py) ---
   // Bidirectional LSTM logic: 5 price features over 50 bars
   // MQL5 translation: multi-feature momentum classifier
   g_qfusion.adapters[0].name = "LSTM";
   {
      // Returns, log_returns, high_low, close_open, volume features
      double returns_sum = 0, log_ret_sum = 0;
      for(int i = 0; i < 50 && i + 1 < count; i++)
      {
         double ret = (close[i] - close[i+1]) / (close[i+1] + 1e-10);
         returns_sum += ret;
         log_ret_sum += MathLog(MathMax(1e-10, close[i] / close[i+1]));
      }
      double avg_ret = returns_sum / 50.0;
      double avg_log = log_ret_sum / 50.0;

      // High-low ratio (volatility feature)
      double hl_sum = 0;
      for(int i = 0; i < 50 && i < count; i++)
         hl_sum += (high[i] - low[i]) / (close[i] + 1e-10);
      double avg_hl = hl_sum / 50.0;

      // Close-open feature (directional candle body)
      double co_sum = 0;
      for(int i = 0; i < 50 && i < count; i++)
         co_sum += (close[i] - open_arr[i]) / (open_arr[i] + 1e-10);
      double avg_co = co_sum / 50.0;

      // Quantum feature proxy: entropy from price distribution
      double entropy = ComputePriceEntropy(close, count, 50);

      // LSTM decision: tanh fusion of features (simplified forward pass)
      double h1 = MathTanh(avg_ret * 20.0 + avg_log * 15.0 + avg_hl * -5.0);
      double h2 = MathTanh(avg_co * 10.0 + entropy * -3.0 + avg_ret * 8.0);
      double lstm_out = MathTanh(h1 * 0.6 + h2 * 0.4);

      g_qfusion.adapters[0].signal = NormalizeSignal(lstm_out);
      g_qfusion.adapters[0].confidence = MathMin(1.0, MathAbs(lstm_out));
   }

   //--- Adapter 1: QPE Analysis (qpe_adapter.py) ---
   // Quantum Phase Estimation: top-10 state analysis for direction
   // MQL5 translation: multi-period phase analysis
   g_qfusion.adapters[1].name = "QPE";
   {
      // Discretize price into "quantum states" using 10 momentum windows
      double weighted_up = 0, weighted_down = 0;
      int periods[] = {3, 5, 8, 13, 21, 34, 55, 89, 100, 144};
      for(int p = 0; p < 10; p++)
      {
         int lb = periods[p];
         if(lb >= count) continue;
         double ret = (close[0] - close[lb]) / (close[lb] + 1e-10);
         double weight = 1.0 / (p + 1.0);  // Recent periods get more weight
         if(ret > 0) weighted_up += weight;
         else weighted_down += weight;
      }
      double total_w = weighted_up + weighted_down;
      if(total_w < 1e-10) total_w = 1.0;

      double qpe_signal = (weighted_up > weighted_down) ? 1.0 : -1.0;
      double qpe_conf = MathMax(weighted_up, weighted_down) / total_w;

      g_qfusion.adapters[1].signal = qpe_signal;
      g_qfusion.adapters[1].confidence = qpe_conf;
   }

   //--- Adapter 2: Quantum LSTM (quantum_lstm_adapter.py) ---
   // Extended LSTM with FastQuantumExtractor (RY+CNOT numpy simulation)
   // MQL5 translation: LSTM + quantum entropy features
   g_qfusion.adapters[2].name = "QuantumLSTM";
   {
      // Compute 7 quantum features analytically
      double q_entropy = ComputePriceEntropy(close, count, 50);
      double q_dominant = ComputeDominantState(close, count, 50);
      double q_superposition = ComputeSuperposition(close, count, 50);
      double q_coherence = ComputeCoherence(close, count, 50);

      // Technical signal from EMA cross + momentum
      double ema8 = GetEMAValue(g_ma_fast_handle);
      double ema21 = GetEMAValue(g_ma_slow_handle);
      double ema_cross = (ema8 - ema21) / (ema21 + 1e-10);

      // Fusion: quantum features modulate technical signal
      double quantum_mod = (1.0 - q_entropy / 3.0) * q_coherence;
      double qlstm_signal = MathTanh(ema_cross * 50.0 * quantum_mod);

      g_qfusion.adapters[2].signal = NormalizeSignal(qlstm_signal);
      g_qfusion.adapters[2].confidence = MathMin(1.0, MathAbs(qlstm_signal) * (1.0 + q_dominant));
   }

   //--- Adapter 3: Volatility Predictor (volatility_adapter.py) ---
   // ATR/Parkinson/rolling volatility => extreme vol prediction
   g_qfusion.adapters[3].name = "Volatility";
   {
      // ATR at different scales
      double atr_5 = 0, atr_10 = 0, atr_20 = 0;
      for(int i = 0; i < 5 && i < count; i++) atr_5 += high[i] - low[i];
      atr_5 /= 5.0;
      for(int i = 0; i < 10 && i < count; i++) atr_10 += high[i] - low[i];
      atr_10 /= 10.0;
      for(int i = 0; i < 20 && i < count; i++) atr_20 += high[i] - low[i];
      atr_20 /= 20.0;

      // Parkinson volatility
      double parkinson = 0;
      for(int i = 0; i < 20 && i < count; i++)
      {
         double hl_ratio = high[i] / (low[i] + 1e-10);
         double log_hl = MathLog(MathMax(1e-10, hl_ratio));
         parkinson += log_hl * log_hl;
      }
      parkinson = MathSqrt(parkinson / (20.0 * 4.0 * MathLog(2.0)));

      // Vol change ratios
      double vol_change_5 = (atr_20 > 0) ? atr_5 / atr_20 : 1.0;
      double vol_change_10 = (atr_20 > 0) ? atr_10 / atr_20 : 1.0;

      // High vol expected? (signal: 1.0 = high vol, -1.0 = low vol)
      double vol_score = (vol_change_5 * 0.4 + vol_change_10 * 0.3 + parkinson * 10.0 * 0.3);
      double vol_signal = (vol_score > 1.2) ? 1.0 : ((vol_score < 0.8) ? -1.0 : 0.0);
      double vol_conf = MathMin(1.0, MathAbs(vol_score - 1.0) * 2.0);

      g_qfusion.adapters[3].signal = vol_signal;
      g_qfusion.adapters[3].confidence = vol_conf;
   }

   //--- Adapter 4: 3D Bars (bars_3d_adapter.py) ---
   // Typical price, volatility, volume change, yellow cluster
   g_qfusion.adapters[4].name = "Bars3D";
   {
      double typical_now = (high[0] + low[0] + close[0]) / 3.0;
      double typical_prev = (high[1] + low[1] + close[1]) / 3.0;
      double price_return = (typical_now - typical_prev) / (typical_prev + 1e-10);

      // Rolling volatility of returns
      double ret_sum = 0, ret_sq_sum = 0;
      for(int i = 0; i < 20 && i + 1 < count; i++)
      {
         double tp_i = (high[i] + low[i] + close[i]) / 3.0;
         double tp_j = (high[i+1] + low[i+1] + close[i+1]) / 3.0;
         double r = (tp_i - tp_j) / (tp_j + 1e-10);
         ret_sum += r;
         ret_sq_sum += r * r;
      }
      double vol_20 = MathSqrt(ret_sq_sum / 20.0 - (ret_sum / 20.0) * (ret_sum / 20.0));

      // Volume change
      double vol_chg = (volume[1] > 0) ? (double)(volume[0] - volume[1]) / (double)volume[1] : 0;

      // Yellow cluster: high vol + high volume change
      double vol_quantile = (vol_20 > 0.01) ? 1.0 : 0.0;  // Simplified threshold
      double volchg_quantile = (MathAbs(vol_chg) > 0.5) ? 1.0 : 0.0;
      double yellow_cluster = vol_quantile * volchg_quantile;

      // Direction from recent bar pattern
      double bars3d_signal = (price_return > 0) ? 1.0 : -1.0;
      double bars3d_conf = MathMin(1.0, MathAbs(price_return) * 100.0);
      // Yellow cluster boosts confidence
      if(yellow_cluster > 0.5) bars3d_conf = MathMin(1.0, bars3d_conf * 1.5);

      g_qfusion.adapters[4].signal = bars3d_signal;
      g_qfusion.adapters[4].confidence = bars3d_conf;
   }

   //--- Adapter 5: Compression Layer (compression_layer.py) ---
   // Recursive quantum autoencoder => regime detection via compression ratio
   g_qfusion.adapters[5].name = "Compression";
   {
      // Reuse ETARE compression as the quantum compression proxy
      // The Python version uses qutip recursive autoencoders;
      // our ETARE layer implements the same concept classically
      double ratio = g_etare.compression_ratio;
      double comp_signal = 0;
      if(ratio > 1.3) comp_signal = 1.0;   // TRENDING
      else if(ratio < 1.1) comp_signal = -1.0;  // CHOPPY
      else comp_signal = (ratio - 1.2) * 10.0;  // Transitional

      g_qfusion.adapters[5].signal = NormalizeSignal(comp_signal * g_etare.regime);
      g_qfusion.adapters[5].confidence = g_etare.fidelity;
      g_qfusion.compression_ratio = ratio;
      g_qfusion.regime = (ratio > 1.3) ? "TRENDING" : "CHOPPY";
   }

   //--- Adapter 6: ETARE Enhanced (etare_enhanced.py) ---
   // Genetic weights neural network: input->128->64->6 with tanh
   // MQL5 translation: multi-feature tanh network with softmax action
   g_qfusion.adapters[6].name = "ETAREEnhanced";
   {
      // 11 input features matching the Python model
      double features[11];
      features[0]  = GetRSI() / 100.0;                          // RSI normalized
      features[1]  = GetMACD_Main() / (atr + 1e-10);            // MACD normalized
      features[2]  = GetMACD_Signal() / (atr + 1e-10);          // MACD signal
      features[3]  = GetBB_Upper() / (close[0] + 1e-10);        // BB upper
      features[4]  = GetBB_Lower() / (close[0] + 1e-10);        // BB lower
      features[5]  = (close[0] - close[10]) / (atr + 1e-10);    // Momentum
      features[6]  = (close[0] - close[5]) / (close[5] + 1e-10) * 100.0; // ROC
      features[7]  = atr / (close[0] + 1e-10) * 100.0;          // ATR ratio
      // Quantum features (last 3)
      features[8]  = g_etare.compression_ratio;                  // compression
      features[9]  = ComputePriceEntropy(close, count, 50);      // entropy
      features[10] = g_qfusion.adapters[5].confidence;           // fused confidence

      // Simple normalization (z-score)
      double f_mean = 0, f_std = 0;
      for(int i = 0; i < 11; i++) f_mean += features[i];
      f_mean /= 11.0;
      for(int i = 0; i < 11; i++) f_std += (features[i] - f_mean) * (features[i] - f_mean);
      f_std = MathSqrt(f_std / 11.0) + 1e-8;
      for(int i = 0; i < 11; i++) features[i] = (features[i] - f_mean) / f_std;

      // Forward pass: 11->hidden(tanh)->output(6 actions)
      // Simplified 2-layer: tanh(sum(features * weights))
      double h1_out = MathTanh(features[0]*0.3 + features[1]*0.2 + features[5]*0.4 + features[6]*0.3 + features[8]*-0.2);
      double h2_out = MathTanh(features[2]*0.2 + features[3]*-0.3 + features[4]*0.3 + features[7]*-0.1 + features[9]*0.2);
      double h3_out = MathTanh(features[0]*-0.1 + features[5]*0.3 + features[10]*0.4 + features[1]*0.2);

      // Output: BUY strength vs SELL strength (actions 0-2 = buy variants, 3-5 = sell variants)
      double buy_score  = MathTanh(h1_out * 0.5 + h2_out * 0.3 + h3_out * 0.2);
      double sell_score = MathTanh(h1_out * -0.5 + h2_out * -0.3 + h3_out * 0.2);

      double etare_enh_signal = buy_score - sell_score;
      g_qfusion.adapters[6].signal = NormalizeSignal(etare_enh_signal);
      g_qfusion.adapters[6].confidence = MathMin(1.0, MathAbs(etare_enh_signal));
   }

   //--- Adapter 7: Signal Fusion Meta (signal_fusion.py) ---
   // Meta-adapter: agreement score across adapters 0-6
   g_qfusion.adapters[7].name = "FusionMeta";
   {
      int agree_bull = 0, agree_bear = 0;
      double conf_sum = 0;
      for(int i = 0; i < 7; i++)
      {
         if(g_qfusion.adapters[i].signal > 0.1) agree_bull++;
         else if(g_qfusion.adapters[i].signal < -0.1) agree_bear++;
         conf_sum += g_qfusion.adapters[i].confidence;
      }
      double meta_signal = 0;
      if(agree_bull > agree_bear) meta_signal = (double)agree_bull / 7.0;
      else if(agree_bear > agree_bull) meta_signal = -(double)agree_bear / 7.0;

      g_qfusion.adapters[7].signal = NormalizeSignal(meta_signal);
      g_qfusion.adapters[7].confidence = MathMin(1.0, conf_sum / 7.0);
   }

   //--- WEIGHTED FUSION (signal_fusion.py logic) ---
   double weighted_sum = 0, total_conf = 0;
   for(int i = 0; i < N_QF_ADAPTERS; i++)
   {
      double contribution = g_qfusion.adapters[i].signal * g_qfusion.adapters[i].confidence * qf_weights[i];
      weighted_sum += contribution;
      total_conf += g_qfusion.adapters[i].confidence * qf_weights[i];
   }

   double weight_total = 0;
   for(int i = 0; i < N_QF_ADAPTERS; i++) weight_total += qf_weights[i];
   g_qfusion.fused_signal = (weight_total > 0) ? weighted_sum / weight_total : 0;

   // Regime adjustment: de-risk in choppy markets (from signal_fusion.py)
   if(g_qfusion.regime == "CHOPPY")
      g_qfusion.fused_signal *= 0.8;

   g_qfusion.fused_signal = NormalizeSignal(g_qfusion.fused_signal);
   g_qfusion.fused_confidence = (weight_total > 0) ? total_conf / weight_total : 0;
}

//+------------------------------------------------------------------+
//| TEQA v3.0 QUANTUM CIRCUIT PIPELINE (v2.0 NEW)                   |
//| Translates 33-TE-family split-architecture quantum engine into   |
//| pure MQL5. Genome circuit (25 qubits) + Neural circuit (8 qubits)|
//| fused through synaptic merger. Includes genomic shock detector   |
//| and neural mosaic consensus voting.                               |
//+------------------------------------------------------------------+

//--- Deterministic RNG for mosaic neuron generation ---
uint MosaicRand()
{
   g_mosaic_seed = g_mosaic_seed * 1103515245 + 12345;
   return (g_mosaic_seed >> 16) & 0x7FFF;
}

double MosaicRandDouble()
{
   return (double)MosaicRand() / 32767.0;
}

//--- Initialize mosaic neurons with L1 insertions ---
void InitMosaicNeurons()
{
   for(int n = 0; n < N_MOSAIC_NEURONS; n++)
   {
      g_teqa.neurons[n].neuron_id = n;
      g_teqa.neurons[n].n_insertions = 2 + (int)(MosaicRandDouble() * 3.0);
      g_teqa.neurons[n].vote = 0;
      g_teqa.neurons[n].confidence = 0;

      for(int j = 0; j < 5; j++)
      {
         g_teqa.neurons[n].insertion_targets[j] = -1;
         g_teqa.neurons[n].insertion_modifiers[j] = 1.0;
      }

      for(int j = 0; j < g_teqa.neurons[n].n_insertions && j < 5; j++)
      {
         // L1 targets neural TEs preferentially (indices 25-32) but can hit others
         int target;
         if(MosaicRandDouble() < 0.6)
            target = N_QUBITS_GENOME + (int)(MosaicRandDouble() * N_QUBITS_NEURAL);
         else
            target = (int)(MosaicRandDouble() * N_TE_FAMILIES);

         g_teqa.neurons[n].insertion_targets[j] = target;

         // Effect: enhance (>1.0), disrupt (<1.0), invert (<0)
         double roll = MosaicRandDouble();
         if(roll < 0.3)
            g_teqa.neurons[n].insertion_modifiers[j] = 1.0 + MosaicRandDouble() * 0.5;  // enhance
         else if(roll < 0.6)
            g_teqa.neurons[n].insertion_modifiers[j] = 1.0 - MosaicRandDouble() * 0.5;  // disrupt
         else if(roll < 0.8)
            g_teqa.neurons[n].insertion_modifiers[j] = -1.0;  // invert
         else
            g_teqa.neurons[n].insertion_modifiers[j] = 1.0;   // rewire (neutral strength)
      }
   }
}

//--- Compute genomic shock level (McClintock hypothesis) ---
void ComputeGenomicShock(const double &close[], const double &high[],
                          const double &low[], const long &volume[], int count)
{
   if(count < 30) { g_teqa.shock_level = 1.0; g_teqa.shock_label = "NORMAL"; return; }

   // ATR expansion
   double atr_current = 0, atr_baseline = 0;
   for(int i = 0; i < 5 && i < count; i++) atr_current += high[i] - low[i];
   atr_current /= 5.0;
   for(int i = 5; i < 25 && i < count; i++) atr_baseline += high[i] - low[i];
   atr_baseline /= 20.0;
   double atr_ratio = atr_current / (atr_baseline + 1e-10);

   // Volume shock
   double vol_current = 0, vol_baseline = 0;
   for(int i = 0; i < 5 && i < count; i++) vol_current += (double)volume[i];
   vol_current /= 5.0;
   for(int i = 5; i < 25 && i < count; i++) vol_baseline += (double)volume[i];
   vol_baseline /= 20.0;
   double vol_ratio = vol_current / (vol_baseline + 1e-10);

   // Drawdown component
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd = (g_risk.balance_hwm > 0) ? (g_risk.balance_hwm - equity) / g_risk.balance_hwm : 0;
   double dd_component = MathMin(1.0, dd * 10.0);

   // Combined shock
   g_teqa.shock_level = atr_ratio * 0.5 + vol_ratio * 0.3 + dd_component * 0.2;

   if(g_teqa.shock_level < SHOCK_LOW)       g_teqa.shock_label = "CALM";
   else if(g_teqa.shock_level < SHOCK_NORMAL) g_teqa.shock_label = "NORMAL";
   else if(g_teqa.shock_level < SHOCK_HIGH)   g_teqa.shock_label = "ELEVATED";
   else if(g_teqa.shock_level < SHOCK_EXTREME) g_teqa.shock_label = "SHOCK";
   else                                         g_teqa.shock_label = "EXTREME";
}

//--- Compute all 33 TE activations ---
void ComputeTEActivations(const double &close[], const double &high[],
                           const double &low[], const double &open_arr[],
                           const long &volume[], int count, double atr)
{
   if(count < 50) return;

   // Helper values
   double ema8 = GetEMAValue(g_ma_fast_handle);
   double ema21 = GetEMAValue(g_ma_slow_handle);
   double rsi = GetRSI();
   double stoch = GetStochastic();

   // TE 0: BEL_Pao -- momentum
   double mom10 = (close[0] - close[10]) / (close[10] + 1e-10);
   g_teqa.te_activations[0].strength = Sigmoid(mom10 * 20.0);
   g_teqa.te_activations[0].direction = (mom10 > 0) ? 1 : -1;
   g_teqa.te_activations[0].stress_responsive = false;
   g_teqa.te_activations[0].neural_target = false;

   // TE 1: DIRS1 -- trend strength
   double ts = (ema8 - ema21) / (ema21 + 1e-10);
   g_teqa.te_activations[1].strength = Sigmoid(ts * 50.0);
   g_teqa.te_activations[1].direction = (ts > 0) ? 1 : -1;
   g_teqa.te_activations[1].stress_responsive = false;
   g_teqa.te_activations[1].neural_target = false;

   // TE 2: Ty1_copia -- RSI threshold
   g_teqa.te_activations[2].direction = 0;
   g_teqa.te_activations[2].stress_responsive = false;
   g_teqa.te_activations[2].neural_target = false;
   if(rsi > 70) { g_teqa.te_activations[2].strength = (rsi - 70) / 30.0; g_teqa.te_activations[2].direction = -1; }
   else if(rsi < 30) { g_teqa.te_activations[2].strength = (30 - rsi) / 30.0; g_teqa.te_activations[2].direction = 1; }
   else g_teqa.te_activations[2].strength = 0;

   // TE 3: Ty3_gypsy -- MACD
   double macd_val = GetMACD_Main();
   double macd_sig = GetMACD_Signal();
   double macd_diff = macd_val - macd_sig;
   g_teqa.te_activations[3].strength = Sigmoid(macd_diff * 100.0);
   g_teqa.te_activations[3].direction = (macd_diff > 0) ? 1 : -1;
   g_teqa.te_activations[3].stress_responsive = false;
   g_teqa.te_activations[3].neural_target = false;

   // TE 4: Ty5 -- Bollinger position
   double bb_upper = GetBB_Upper(), bb_lower = GetBB_Lower();
   double bb_mid = (bb_upper + bb_lower) / 2.0;
   double bb_std = (bb_upper - bb_lower) / 4.0;
   double bb_pos = (bb_std > 0) ? (close[0] - bb_mid) / bb_std : 0;
   g_teqa.te_activations[4].strength = MathAbs(bb_pos) * 0.3;
   g_teqa.te_activations[4].direction = (bb_pos > 1) ? -1 : ((bb_pos < -1) ? 1 : 0);
   g_teqa.te_activations[4].stress_responsive = false;
   g_teqa.te_activations[4].neural_target = false;

   // TE 5: Alu -- short volatility
   double vol5 = 0, vol20 = 0;
   for(int i = 0; i < 5 && i+1 < count; i++) vol5 += MathAbs(close[i]-close[i+1]) / (close[i+1]+1e-10);
   vol5 /= 5.0;
   for(int i = 0; i < 20 && i+1 < count; i++) vol20 += MathAbs(close[i]-close[i+1]) / (close[i+1]+1e-10);
   vol20 /= 20.0;
   double vr = (vol20 > 0) ? vol5 / vol20 : 1.0;
   g_teqa.te_activations[5].strength = (vr > 0.5) ? MathMin(1.0, (vr - 0.5) * 2.0) : 0;
   g_teqa.te_activations[5].direction = 0;
   g_teqa.te_activations[5].stress_responsive = false;
   g_teqa.te_activations[5].neural_target = false;

   // TE 6: LINE -- price change
   double pc = (close[0] - close[1]) / (close[1] + 1e-10);
   g_teqa.te_activations[6].strength = Sigmoid(MathAbs(pc) * 200.0);
   g_teqa.te_activations[6].direction = (pc > 0) ? 1 : -1;
   g_teqa.te_activations[6].stress_responsive = false;
   g_teqa.te_activations[6].neural_target = false;

   // TE 7: Penelope -- trend duration
   int trend_count = 0;
   for(int i = 0; i + 1 < count && i < 20; i++)
   { if(close[i] > close[i+1]) trend_count++; else break; }
   g_teqa.te_activations[7].strength = MathMin(1.0, trend_count / 10.0);
   g_teqa.te_activations[7].direction = (trend_count > 0) ? 1 : -1;
   g_teqa.te_activations[7].stress_responsive = false;
   g_teqa.te_activations[7].neural_target = false;

   // TE 8: RTE -- mean reversion
   double sma20_val = GetSMA20();
   double dev = (close[0] - sma20_val) / (sma20_val + 1e-10);
   g_teqa.te_activations[8].strength = Sigmoid(MathAbs(dev) * 30.0);
   g_teqa.te_activations[8].direction = (dev > 0) ? -1 : 1;
   g_teqa.te_activations[8].stress_responsive = false;
   g_teqa.te_activations[8].neural_target = false;

   // TE 9: SINE -- tick volume
   double vol_ratio = (double)volume[0] / (ComputeAvgVolume(volume, count, 20) + 1e-10);
   g_teqa.te_activations[9].strength = (vol_ratio > 1.5) ? MathMin(1.0, (vol_ratio - 1.5) * 0.5) : 0;
   g_teqa.te_activations[9].direction = 0;
   g_teqa.te_activations[9].stress_responsive = false;
   g_teqa.te_activations[9].neural_target = false;

   // TE 10: VIPER_Ngaro -- ATR ratio
   double atr_avg = 0;
   for(int i = 0; i < 5; i++) { double tr = high[i] - low[i]; atr_avg += tr; }
   atr_avg /= 5.0;
   double atr_r = atr / (atr_avg + 1e-10);
   g_teqa.te_activations[10].strength = MathMin(1.0, MathMax(0.0, (atr_r - 0.5) * 2.0));
   g_teqa.te_activations[10].direction = 0;
   g_teqa.te_activations[10].stress_responsive = false;
   g_teqa.te_activations[10].neural_target = false;

   // TE 11: CACTA -- EMA crossover
   bool crossed = ema8 > ema21;
   g_teqa.te_activations[11].strength = crossed ? 1.0 : 0.0;
   g_teqa.te_activations[11].direction = crossed ? 1 : -1;
   g_teqa.te_activations[11].stress_responsive = false;
   g_teqa.te_activations[11].neural_target = false;

   // TE 12: Crypton -- compression ratio (zlib proxy)
   g_teqa.te_activations[12].strength = Sigmoid(g_etare.compression_ratio - 1.5);
   g_teqa.te_activations[12].direction = (g_etare.compression_ratio > 2) ? 1 : -1;
   g_teqa.te_activations[12].stress_responsive = false;
   g_teqa.te_activations[12].neural_target = false;

   // TE 13-24: Remaining original TEs (simplified activations)
   // TE 13: Helitron -- volume profile skew
   double vol_recent = 0, vol_prior = 0;
   for(int i = 0; i < 5 && i < count; i++) vol_recent += (double)volume[i];
   for(int i = 5; i < 20 && i < count; i++) vol_prior += (double)volume[i];
   double profile_skew = (vol_prior > 0) ? (vol_recent / 5.0) / (vol_prior / 15.0) - 1.0 : 0;
   g_teqa.te_activations[13].strength = Sigmoid(profile_skew * 5.0);
   g_teqa.te_activations[13].direction = (profile_skew > 0) ? 1 : -1;
   g_teqa.te_activations[13].stress_responsive = false;
   g_teqa.te_activations[13].neural_target = false;

   // TE 14: hobo -- candle pattern
   double body = MathAbs(close[0] - open_arr[0]);
   double total_range = high[0] - low[0];
   double body_ratio = (total_range > 0) ? body / total_range : 0;
   g_teqa.te_activations[14].strength = body_ratio;
   g_teqa.te_activations[14].direction = (close[0] > open_arr[0]) ? 1 : -1;
   g_teqa.te_activations[14].stress_responsive = false;
   g_teqa.te_activations[14].neural_target = false;

   // TE 15: I_element -- support/resistance
   double recent_high = high[0], recent_low = low[0];
   for(int i = 0; i < 20 && i < count; i++)
   { if(high[i] > recent_high) recent_high = high[i]; if(low[i] < recent_low) recent_low = low[i]; }
   double sr_pos = (recent_high - recent_low > 0) ? (close[0] - recent_low) / (recent_high - recent_low) : 0.5;
   g_teqa.te_activations[15].strength = sr_pos;
   g_teqa.te_activations[15].direction = (sr_pos > 0.8) ? -1 : ((sr_pos < 0.2) ? 1 : 0);
   g_teqa.te_activations[15].stress_responsive = false;
   g_teqa.te_activations[15].neural_target = false;

   // TE 16: Mariner_Tc1 -- fractal dimension proxy
   double abs_ret_sum = 0;
   int ret_count = 0;
   for(int i = 0; i < 30 && i+1 < count; i++)
   { abs_ret_sum += MathAbs(close[i] - close[i+1]) / (close[i+1]+1e-10); ret_count++; }
   double fd = 1.0 + MathLog(abs_ret_sum + 1e-10) / MathLog((double)ret_count + 1e-10);
   g_teqa.te_activations[16].strength = MathMin(1.0, MathMax(0.0, (fd - 1.0)));
   g_teqa.te_activations[16].direction = 0;
   g_teqa.te_activations[16].stress_responsive = false;
   g_teqa.te_activations[16].neural_target = false;

   // TE 17: Mavericks_Polinton -- order flow
   double up_vol = 0, dn_vol = 0;
   for(int i = 0; i < 10 && i+1 < count; i++)
   {
      if(close[i] > close[i+1]) up_vol += (double)volume[i];
      else dn_vol += (double)volume[i];
   }
   double imbalance = (up_vol + dn_vol > 0) ? (up_vol - dn_vol) / (up_vol + dn_vol) : 0;
   g_teqa.te_activations[17].strength = (MathAbs(imbalance) > 0.2) ? MathAbs(imbalance) : 0;
   g_teqa.te_activations[17].direction = (imbalance > 0) ? 1 : -1;
   g_teqa.te_activations[17].stress_responsive = false;
   g_teqa.te_activations[17].neural_target = false;

   // TE 18: Mutator -- mutation rate (direction changes)
   int sign_changes = 0;
   for(int i = 0; i < 19 && i+2 < count; i++)
   {
      double d1 = close[i] - close[i+1], d2 = close[i+1] - close[i+2];
      if(d1 * d2 < 0) sign_changes++;
   }
   g_teqa.te_activations[18].strength = (double)sign_changes / 19.0;
   g_teqa.te_activations[18].direction = 0;
   g_teqa.te_activations[18].stress_responsive = false;
   g_teqa.te_activations[18].neural_target = false;

   // TE 19: P_element -- spread analysis
   double spread_exp = (high[0] - low[0]) / (atr + 1e-10);
   g_teqa.te_activations[19].strength = MathMin(1.0, MathMax(0.0, spread_exp - 1.0));
   g_teqa.te_activations[19].direction = (close[0] > open_arr[0]) ? 1 : -1;
   g_teqa.te_activations[19].stress_responsive = false;
   g_teqa.te_activations[19].neural_target = false;

   // TE 20: PIF_Harbinger -- microstructure autocorrelation
   double ac = ComputeAutocorrelation(close, count, 20);
   g_teqa.te_activations[20].strength = MathAbs(ac);
   g_teqa.te_activations[20].direction = (ac > 0) ? 1 : -1;
   g_teqa.te_activations[20].stress_responsive = false;
   g_teqa.te_activations[20].neural_target = false;

   // TE 21: piggyBac -- gap analysis
   double gap = (count > 1) ? MathAbs(open_arr[0] - close[1]) / (close[1] + 1e-10) : 0;
   g_teqa.te_activations[21].strength = MathMin(1.0, gap * 100.0);
   g_teqa.te_activations[21].direction = (open_arr[0] > close[1]) ? 1 : -1;
   g_teqa.te_activations[21].stress_responsive = false;
   g_teqa.te_activations[21].neural_target = false;

   // TE 22: pogo -- session overlap
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   bool is_overlap = (dt.hour >= 13 && dt.hour <= 17);
   g_teqa.te_activations[22].strength = is_overlap ? 1.0 : 0.0;
   g_teqa.te_activations[22].direction = 0;
   g_teqa.te_activations[22].stress_responsive = false;
   g_teqa.te_activations[22].neural_target = false;

   // TE 23: Rag_like -- diversity index
   int unique_dirs = 0;
   bool has_up = false, has_down = false, has_flat = false;
   for(int i = 0; i < 30 && i+1 < count; i++)
   {
      double d = close[i] - close[i+1];
      if(d > 0) has_up = true; else if(d < 0) has_down = true; else has_flat = true;
   }
   if(has_up) unique_dirs++; if(has_down) unique_dirs++; if(has_flat) unique_dirs++;
   g_teqa.te_activations[23].strength = unique_dirs / 3.0;
   g_teqa.te_activations[23].direction = 0;
   g_teqa.te_activations[23].stress_responsive = false;
   g_teqa.te_activations[23].neural_target = false;

   // TE 24: Transib -- autocorrelation
   g_teqa.te_activations[24].strength = MathAbs(ac);
   g_teqa.te_activations[24].direction = (ac > 0.1) ? 1 : ((ac < -0.1) ? -1 : 0);
   g_teqa.te_activations[24].stress_responsive = false;
   g_teqa.te_activations[24].neural_target = false;

   // --- NEURAL TE FAMILIES (v3.0 qubits 25-32) ---

   // TE 25: L1_Neuronal -- pattern repetition
   double best_corr = 0;
   for(int offset = 10; offset < 45 && offset + 5 < count; offset++)
   {
      double corr = ComputePatternCorrelation(close, 0, offset, 5);
      if(MathAbs(corr) > MathAbs(best_corr)) best_corr = corr;
   }
   g_teqa.te_activations[25].strength = MathAbs(best_corr);
   g_teqa.te_activations[25].direction = (best_corr > 0) ? 1 : -1;
   g_teqa.te_activations[25].stress_responsive = true;
   g_teqa.te_activations[25].neural_target = false;

   // TE 26: L1_Somatic -- multi-TF variance
   double tf_signals[4];
   int tf_lookbacks[] = {5, 10, 20, 50};
   for(int t = 0; t < 4; t++)
   {
      int lb = tf_lookbacks[t];
      tf_signals[t] = (lb < count) ? (close[0] - close[lb]) / (close[lb] + 1e-10) : 0;
   }
   int tf_dirs = 0;
   for(int t = 0; t < 4; t++) tf_dirs += (tf_signals[t] > 0) ? 1 : -1;
   double tf_agreement = MathAbs(tf_dirs) / 4.0;
   double tf_diversity = 1.0 - tf_agreement;
   g_teqa.te_activations[26].strength = tf_diversity;
   g_teqa.te_activations[26].direction = 0;
   g_teqa.te_activations[26].stress_responsive = true;
   g_teqa.te_activations[26].neural_target = false;

   // TE 27: HERV_Synapse -- cross-correlation proxy (use EMA agreement)
   double herv_corr = (ema8 > ema21 && rsi > 50) ? 0.8 :
                      (ema8 < ema21 && rsi < 50) ? 0.8 : 0.3;
   g_teqa.te_activations[27].strength = herv_corr;
   g_teqa.te_activations[27].direction = (ema8 > ema21) ? 1 : -1;
   g_teqa.te_activations[27].stress_responsive = false;
   g_teqa.te_activations[27].neural_target = true;

   // TE 28: SVA_Regulatory -- compression breakout
   double cr_recent = ComputeLocalCompression(close, 0, 10);
   double cr_prior = ComputeLocalCompression(close, 10, 20);
   double breakout = cr_recent - cr_prior;
   g_teqa.te_activations[28].strength = Sigmoid(breakout * 3.0);
   g_teqa.te_activations[28].direction = (breakout > 0) ? 1 : -1;
   g_teqa.te_activations[28].stress_responsive = true;
   g_teqa.te_activations[28].neural_target = true;

   // TE 29: Alu_Exonization -- noise pattern (lag-3 autocorrelation)
   double lag3_ac = ComputeLagAutocorrelation(close, count, 3, 20);
   g_teqa.te_activations[29].strength = (MathAbs(lag3_ac) > 0.3) ? MathAbs(lag3_ac) : 0;
   g_teqa.te_activations[29].direction = (lag3_ac > 0) ? 1 : -1;
   g_teqa.te_activations[29].stress_responsive = false;
   g_teqa.te_activations[29].neural_target = true;

   // TE 30: TRIM28_Silencer -- drawdown suppressor
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd = (g_risk.balance_hwm > 0) ? (g_risk.balance_hwm - equity) / g_risk.balance_hwm : 0;
   g_teqa.te_activations[30].strength = MathMin(1.0, dd * 10.0);
   g_teqa.te_activations[30].direction = 0;
   g_teqa.te_activations[30].stress_responsive = false;
   g_teqa.te_activations[30].neural_target = false;

   // TE 31: piwiRNA_Neural -- signal/noise ratio
   double sig_power = 0, noise_power = 0;
   for(int i = 0; i < 30 && i+1 < count; i++)
   {
      double r = (close[i] - close[i+1]) / (close[i+1] + 1e-10);
      sig_power += r;
      noise_power += r * r;
   }
   sig_power = MathAbs(sig_power / 30.0);
   noise_power = MathSqrt(noise_power / 30.0) + 1e-10;
   double snr = sig_power / noise_power;
   g_teqa.te_activations[31].strength = (snr > 0.2) ? MathMin(1.0, snr * 5.0) : 0;
   g_teqa.te_activations[31].direction = 0;
   g_teqa.te_activations[31].stress_responsive = false;
   g_teqa.te_activations[31].neural_target = false;

   // TE 32: Arc_Capsid -- echo of successful patterns
   // Uses dopamine reward as proxy for "successful pattern echo"
   double dopamine_val = ComputeDopamine();
   g_teqa.te_activations[32].strength = MathAbs(dopamine_val);
   g_teqa.te_activations[32].direction = (dopamine_val > 0) ? 1 : ((dopamine_val < 0) ? -1 : 0);
   g_teqa.te_activations[32].stress_responsive = true;
   g_teqa.te_activations[32].neural_target = true;

   // --- Apply genomic shock adjustments ---
   if(g_teqa.shock_label == "EXTREME")
   {
      for(int i = 0; i < N_TE_FAMILIES; i++)
      {
         if(i != 30) g_teqa.te_activations[i].strength *= 0.1;  // TRIM28 suppresses all
         else g_teqa.te_activations[i].strength = 1.0;
      }
   }
   else if(g_teqa.shock_label == "SHOCK")
   {
      for(int i = 0; i < N_TE_FAMILIES; i++)
      {
         if(i < N_QUBITS_GENOME || g_teqa.te_activations[i].stress_responsive)
            g_teqa.te_activations[i].strength = MathMax(g_teqa.te_activations[i].strength, 0.8);
      }
   }
   else if(g_teqa.shock_label == "ELEVATED")
   {
      for(int i = 0; i < N_TE_FAMILIES; i++)
      {
         if(g_teqa.te_activations[i].stress_responsive)
            g_teqa.te_activations[i].strength = MathMin(1.0, g_teqa.te_activations[i].strength * 1.3);
      }
   }
}

//--- Classical simulation of quantum genome circuit (25 qubits) ---
double SimulateGenomeCircuit()
{
   // Simulate RY rotations + entanglement classically
   // Each TE's activation contributes a directional vote weighted by strength
   // Entanglement is modeled as correlation chains between adjacent TEs
   double bullish_weight = 0, bearish_weight = 0;

   // Class I chain (0-10): correlated retrotransposons
   double class1_signal = 0;
   for(int i = 0; i <= 10; i++)
   {
      double s = g_teqa.te_activations[i].strength * g_teqa.te_activations[i].direction;
      class1_signal += s;
   }
   class1_signal /= 11.0;

   // Class II chain (11-24): correlated DNA transposons
   double class2_signal = 0;
   for(int i = 11; i <= 24; i++)
   {
      double s = g_teqa.te_activations[i].strength * g_teqa.te_activations[i].direction;
      class2_signal += s;
   }
   class2_signal /= 14.0;

   // Cross-class bridges: LINE<->Mariner (6,16), Ty3/gypsy<->Helitron (3,13)
   double bridge_1 = g_teqa.te_activations[6].strength * g_teqa.te_activations[6].direction +
                      g_teqa.te_activations[16].strength * g_teqa.te_activations[16].direction;
   double bridge_2 = g_teqa.te_activations[3].strength * g_teqa.te_activations[3].direction +
                      g_teqa.te_activations[13].strength * g_teqa.te_activations[13].direction;

   // Genome signal: weighted fusion of classes + bridges
   double genome = class1_signal * 0.35 + class2_signal * 0.35 +
                    bridge_1 * 0.15 + bridge_2 * 0.15;

   return NormalizeSignal(genome);
}

//--- Classical simulation of quantum neural circuit (8 qubits) ---
double SimulateNeuralCircuit(double genome_signal)
{
   // Neural TEs (indices 25-32, mapped to local 0-7)
   double neural_signal = 0;

   // L1_Neuronal (0) receives genome injection
   double l1n = g_teqa.te_activations[25].strength * g_teqa.te_activations[25].direction;
   l1n += genome_signal * 0.5;  // Genome signal injection

   // L1_Somatic (1) entangled with L1_Neuronal
   double l1s = g_teqa.te_activations[26].strength * g_teqa.te_activations[26].direction;
   l1n = (l1n + l1s * 0.3);  // Entanglement

   // HERV_Synapse (2) <-> L1_Neuronal
   double herv = g_teqa.te_activations[27].strength * g_teqa.te_activations[27].direction;
   l1n = (l1n + herv * 0.2);

   // SVA_Regulatory (3) <-> Alu_Exonization (4)
   double sva = g_teqa.te_activations[28].strength * g_teqa.te_activations[28].direction;
   double alu_ex = g_teqa.te_activations[29].strength * g_teqa.te_activations[29].direction;
   double sva_alu = (sva + alu_ex) * 0.5;

   // TRIM28 (5) suppresses via CZ (phase gate = damping)
   double trim28_strength = g_teqa.te_activations[30].strength;
   double suppression = 1.0 - trim28_strength * 0.7;

   // piwiRNA (6) targets L1 (quality control)
   double piwi = g_teqa.te_activations[31].strength;
   double l1_quality = (piwi > 0.5) ? 1.0 : 0.7;  // Good SNR = trust L1 more

   // Arc_Capsid (7) -- inter-neuron signal transfer
   double arc = g_teqa.te_activations[32].strength * g_teqa.te_activations[32].direction;

   // Combine neural circuit outputs
   neural_signal = (l1n * l1_quality * 0.30 +
                    herv * 0.15 +
                    sva_alu * 0.20 +
                    arc * 0.15 +
                    l1s * 0.10 +
                    genome_signal * 0.10) * suppression;

   return NormalizeSignal(neural_signal);
}

//--- Run neural mosaic consensus voting ---
void RunMosaicConsensus(double genome_signal, double neural_signal)
{
   for(int n = 0; n < N_MOSAIC_NEURONS; n++)
   {
      // Each neuron modifies the base signals via its L1 insertions
      double neuron_signal = genome_signal * 0.6 + neural_signal * 0.4;

      for(int j = 0; j < g_teqa.neurons[n].n_insertions && j < 5; j++)
      {
         int target = g_teqa.neurons[n].insertion_targets[j];
         double mod = g_teqa.neurons[n].insertion_modifiers[j];

         if(target >= 0 && target < N_TE_FAMILIES)
         {
            double te_contribution = g_teqa.te_activations[target].strength *
                                      g_teqa.te_activations[target].direction;

            if(mod < 0)
               te_contribution = -te_contribution;  // Inversion
            else
               te_contribution *= mod;

            neuron_signal += te_contribution * 0.1;  // Each insertion adds a small perturbation
         }
      }

      g_teqa.neurons[n].confidence = MathMin(1.0, MathAbs(neuron_signal));
      if(neuron_signal > 0.05) g_teqa.neurons[n].vote = 1;
      else if(neuron_signal < -0.05) g_teqa.neurons[n].vote = -1;
      else g_teqa.neurons[n].vote = 0;
   }

   // Consensus
   int long_votes = 0, short_votes = 0;
   double weighted_dir = 0, total_conf = 0;
   for(int n = 0; n < N_MOSAIC_NEURONS; n++)
   {
      if(g_teqa.neurons[n].vote > 0) long_votes++;
      if(g_teqa.neurons[n].vote < 0) short_votes++;
      weighted_dir += g_teqa.neurons[n].vote * g_teqa.neurons[n].confidence;
      total_conf += g_teqa.neurons[n].confidence;
   }

   if(total_conf > 0) weighted_dir /= total_conf;
   g_teqa.consensus_score = MathAbs(weighted_dir);
   g_teqa.consensus_direction = (weighted_dir > 0.1) ? 1 : ((weighted_dir < -0.1) ? -1 : 0);
}

//--- Master TEQA computation ---
void ComputeTEQA(const double &close[], const double &high[],
                  const double &low[], const double &open_arr[],
                  const long &volume[], int count, double atr)
{
   if(count < 60) return;

   // Step 1: Genomic shock detection
   ComputeGenomicShock(close, high, low, volume, count);

   // Step 2: Compute all 33 TE activations
   ComputeTEActivations(close, high, low, open_arr, volume, count, atr);

   // Step 3: Simulate genome circuit (25 qubits)
   g_teqa.genome_signal = SimulateGenomeCircuit();

   // Step 4: Simulate neural circuit (8 qubits) with genome injection
   g_teqa.neural_signal = SimulateNeuralCircuit(g_teqa.genome_signal);

   // Step 5: Neural mosaic consensus
   RunMosaicConsensus(g_teqa.genome_signal, g_teqa.neural_signal);

   // Step 6: Synaptic fusion of genome + neural + consensus
   if(g_teqa.consensus_score >= NEURAL_CONSENSUS)
   {
      // Strong consensus: use consensus direction with full weight
      g_teqa.fused_signal = g_teqa.consensus_direction * g_teqa.consensus_score;
   }
   else
   {
      // Weak consensus: fall back to circuit fusion
      g_teqa.fused_signal = g_teqa.genome_signal * 0.5 + g_teqa.neural_signal * 0.5;
   }

   g_teqa.fused_signal = NormalizeSignal(g_teqa.fused_signal);
}

//+------------------------------------------------------------------+
//| LOT AND DISTANCE COMPUTATION -- ALL DERIVED FROM ATR_MULT        |
//+------------------------------------------------------------------+
void ComputeLotAndDistances(double atr)
{
   g_sl_distance = atr * ATR_MULT;
   if(g_sl_distance < _Point) g_sl_distance = _Point * 10;

   g_tp_distance = g_sl_distance * TP_MULTIPLIER;
   g_partial_dist = g_tp_distance * (PARTIAL_TP_PCT / 100.0);
   g_grid_spacing = g_sl_distance * 1.5;

   double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double min_lot    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lot_step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(tick_size <= 0 || tick_value <= 0)
   {
      g_base_lot = min_lot;
      return;
   }

   double sl_ticks = g_sl_distance / tick_size;
   if(sl_ticks * tick_value <= 0)
   {
      g_base_lot = min_lot;
      return;
   }

   g_base_lot = MAX_SL_DOLLARS / (sl_ticks * tick_value);
   g_base_lot = MathFloor(g_base_lot / lot_step) * lot_step;
   g_base_lot = MathMax(min_lot, MathMin(max_lot, g_base_lot));
}

//+------------------------------------------------------------------+
//| GRID MANAGEMENT                                                   |
//+------------------------------------------------------------------+
void OpenGridPosition(int direction, double confidence, BioSignal &bio, DMTSignal &dmt, GeneSignal &gene)
{
   double price = (direction == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                    : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   int dye = (g_etare.compression_ratio > 1.3) ? DYES_BLUE : DYES_RED;
   string dye_tag;
   if(dye == DYES_BLUE) { g_blue_count++; dye_tag = StringFormat("BLUE_%03d", g_blue_count); }
   else { g_red_count++; dye_tag = StringFormat("RED_%03d", g_red_count); }

   string comment = StringFormat("%s|%s|C:%.2f|E:%.1f|QF:%.2f|TQ:%.2f",
                                 InstanceName, dye_tag, confidence,
                                 g_etare.compression_ratio,
                                 g_qfusion.fused_signal,
                                 g_teqa.fused_signal);

   bool result = false;
   if(direction == 1)
      result = g_trade.Buy(g_base_lot, _Symbol, 0, 0, 0, comment);
   else
      result = g_trade.Sell(g_base_lot, _Symbol, 0, 0, 0, comment);

   if(result)
   {
      ulong ticket = g_trade.ResultOrder();
      ArrayResize(g_positions, g_pos_count + 1);
      g_positions[g_pos_count].ticket      = ticket;
      g_positions[g_pos_count].entry_price = price;
      g_positions[g_pos_count].direction   = direction;
      g_positions[g_pos_count].lot         = g_base_lot;
      g_positions[g_pos_count].partial_taken = false;
      g_positions[g_pos_count].dye_color   = dye;
      g_positions[g_pos_count].dye_tag     = dye_tag;
      g_positions[g_pos_count].open_time   = TimeCurrent();

      if(direction == 1)
      {
         g_positions[g_pos_count].hidden_sl  = price - g_sl_distance;
         g_positions[g_pos_count].hidden_tp  = price + g_tp_distance;
         g_positions[g_pos_count].partial_tp = price + g_partial_dist;
      }
      else
      {
         g_positions[g_pos_count].hidden_sl  = price + g_sl_distance;
         g_positions[g_pos_count].hidden_tp  = price - g_tp_distance;
         g_positions[g_pos_count].partial_tp = price - g_partial_dist;
      }

      g_pos_count++;
      Print(StringFormat("OPEN %s | %s | %s | Lot:%.4f | SL:%.5f | TP:%.5f | Bio:%.3f DMT:%.3f Gene:%.3f QF:%.3f TEQA:%.3f",
            (direction == 1) ? "BUY" : "SELL", dye_tag,
            (dye == DYES_BLUE) ? "TRENDING" : "CHOPPY",
            g_base_lot, g_sl_distance, g_tp_distance,
            bio.composite, dmt.composite, gene.composite,
            g_qfusion.fused_signal, g_teqa.fused_signal));
   }
   else
   {
      Print("ERROR opening position: ", g_trade.ResultRetcode(), " ", g_trade.ResultRetcodeDescription());
   }
}

void ManagePositions()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   for(int i = g_pos_count - 1; i >= 0; i--)
   {
      bool sl_hit = false;
      if(g_positions[i].direction == 1 && bid <= g_positions[i].hidden_sl) sl_hit = true;
      if(g_positions[i].direction == -1 && ask >= g_positions[i].hidden_sl) sl_hit = true;
      if(sl_hit) { ClosePosition(i, "HIDDEN_SL"); continue; }

      bool tp_hit = false;
      if(g_positions[i].direction == 1 && bid >= g_positions[i].hidden_tp) tp_hit = true;
      if(g_positions[i].direction == -1 && ask <= g_positions[i].hidden_tp) tp_hit = true;
      if(tp_hit) { ClosePosition(i, "HIDDEN_TP"); continue; }

      if(!g_positions[i].partial_taken)
      {
         bool partial_hit = false;
         if(g_positions[i].direction == 1 && bid >= g_positions[i].partial_tp) partial_hit = true;
         if(g_positions[i].direction == -1 && ask <= g_positions[i].partial_tp) partial_hit = true;

         if(partial_hit)
         {
            double close_lot = g_positions[i].lot * 0.5;
            double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
            close_lot = MathFloor(close_lot / lot_step) * lot_step;
            double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

            if(close_lot >= min_lot)
            {
               if(g_trade.PositionClosePartial(g_positions[i].ticket, close_lot))
               {
                  g_positions[i].partial_taken = true;
                  g_positions[i].lot -= close_lot;
                  double entry = g_positions[i].entry_price;
                  if(g_positions[i].direction == 1)
                     g_positions[i].hidden_sl = entry + g_sl_distance * (ROLLING_SL_MULT - 1.0) * 0.1;
                  else
                     g_positions[i].hidden_sl = entry - g_sl_distance * (ROLLING_SL_MULT - 1.0) * 0.1;

                  Print(StringFormat("PARTIAL TP | %s | Closed %.4f lots | SL rolled to BE+",
                        g_positions[i].dye_tag, close_lot));
               }
            }
            else
               g_positions[i].partial_taken = true;
         }
      }
   }
}

void ClosePosition(int index, string reason)
{
   if(index < 0 || index >= g_pos_count) return;
   ulong ticket = g_positions[index].ticket;
   string tag = g_positions[index].dye_tag;
   int dye = g_positions[index].dye_color;

   if(g_trade.PositionClose(ticket))
   {
      double pnl = g_trade.ResultProfit();
      Print(StringFormat("CLOSE %s | %s | %s | PnL: $%.2f | Reason: %s",
            tag, (dye == DYES_BLUE) ? "BLUE" : "RED",
            (g_positions[index].direction == 1) ? "BUY" : "SELL",
            pnl, reason));

      for(int j = index; j < g_pos_count - 1; j++)
         g_positions[j] = g_positions[j + 1];
      g_pos_count--;
      ArrayResize(g_positions, g_pos_count);
   }
   else
      Print("ERROR closing position ", ticket, ": ", g_trade.ResultRetcode(), " ", g_trade.ResultRetcodeDescription());
}

bool PriceReachedNextLevel(double bid)
{
   if(g_pos_count == 0 || g_grid_spacing < _Point) return false;
   double last_entry = g_positions[g_pos_count - 1].entry_price;
   int dir = g_positions[0].direction;
   if(dir == 1) return (bid <= last_entry - g_grid_spacing);
   else return (bid >= last_entry + g_grid_spacing);
}

void RepairHiddenLevels()
{
   for(int i = 0; i < g_pos_count; i++)
   {
      if(g_positions[i].hidden_sl == 0 || g_positions[i].hidden_tp == 0)
      {
         double entry = g_positions[i].entry_price;
         int dir = g_positions[i].direction;
         if(dir == 1)
         {
            g_positions[i].hidden_sl  = entry - g_sl_distance;
            g_positions[i].hidden_tp  = entry + g_tp_distance;
            g_positions[i].partial_tp = entry + g_partial_dist;
         }
         else
         {
            g_positions[i].hidden_sl  = entry + g_sl_distance;
            g_positions[i].hidden_tp  = entry - g_tp_distance;
            g_positions[i].partial_tp = entry - g_partial_dist;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| POSITION SYNC                                                     |
//+------------------------------------------------------------------+
void SyncPositions()
{
   g_pos_count = 0;
   ArrayResize(g_positions, 0);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      ArrayResize(g_positions, g_pos_count + 1);
      g_positions[g_pos_count].ticket      = ticket;
      g_positions[g_pos_count].entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
      g_positions[g_pos_count].lot         = PositionGetDouble(POSITION_VOLUME);
      g_positions[g_pos_count].direction   = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
      g_positions[g_pos_count].hidden_sl   = 0;
      g_positions[g_pos_count].hidden_tp   = 0;
      g_positions[g_pos_count].partial_tp  = 0;
      g_positions[g_pos_count].partial_taken = false;
      g_positions[g_pos_count].dye_color   = DYES_RED;
      g_positions[g_pos_count].dye_tag     = StringFormat("SYNC_%03d", g_pos_count + 1);
      g_positions[g_pos_count].open_time   = (datetime)PositionGetInteger(POSITION_TIME);
      g_pos_count++;
   }
}

//+------------------------------------------------------------------+
//| RISK MANAGEMENT                                                   |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
   if(TimeCurrent() >= g_risk.daily_reset_time + 86400)
   {
      g_risk.daily_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
      g_risk.daily_reset_time = GetDailyResetTime();
      g_risk.daily_dd_hit = false;
      Print("Daily risk reset. New balance: $", DoubleToString(g_risk.daily_start_balance, 2));
   }
}

void UpdateHWM()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   if(balance > g_risk.balance_hwm) g_risk.balance_hwm = balance;
   if(equity > g_risk.equity_hwm)   g_risk.equity_hwm  = equity;
}

bool IsRiskSafe()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   double daily_dd_pct = 0;
   if(g_risk.daily_start_balance > 0)
      daily_dd_pct = (g_risk.daily_start_balance - equity) / g_risk.daily_start_balance * 100.0;

   if(daily_dd_pct >= DailyDDLimit)
   {
      if(!g_risk.daily_dd_hit)
      {
         Print("DAILY DD LIMIT HIT: ", DoubleToString(daily_dd_pct, 2), "%");
         g_risk.daily_dd_hit = true;
      }
      return false;
   }

   double max_dd_pct = 0;
   if(g_risk.balance_hwm > 0)
      max_dd_pct = (g_risk.balance_hwm - equity) / g_risk.balance_hwm * 100.0;

   if(max_dd_pct >= MaxDDLimit)
   {
      if(!g_risk.max_dd_hit)
      {
         Print("MAX DD LIMIT HIT: ", DoubleToString(max_dd_pct, 2), "%");
         g_risk.max_dd_hit = true;
      }
      return false;
   }
   return true;
}

datetime GetDailyResetTime()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   return StructToTime(dt);
}

//+------------------------------------------------------------------+
//| HELPER FUNCTIONS                                                   |
//+------------------------------------------------------------------+
double NormalizeSignal(double val)
{
   return MathMax(-1.0, MathMin(1.0, val));
}

double Sigmoid(double x)
{
   x = MathMax(-50.0, MathMin(50.0, x));
   return 1.0 / (1.0 + MathExp(-x));
}

double MathTanh(double x)
{
   if(x > 20) return 1.0;
   if(x < -20) return -1.0;
   double e2x = MathExp(2.0 * x);
   return (e2x - 1.0) / (e2x + 1.0);
}

double GetATR()
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_atr_handle, 0, 0, 1, buf) < 1) return 0;
   return buf[0];
}

double GetRSI()
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_rsi_handle, 0, 0, 1, buf) < 1) return 50;
   return buf[0];
}

double GetStochastic()
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_stoch_handle, 0, 0, 1, buf) < 1) return 50;
   return buf[0];
}

double GetEMAValue(int handle)
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(handle, 0, 0, 1, buf) < 1) return 0;
   return buf[0];
}

double GetMACD_Main()
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_macd_handle, 0, 0, 1, buf) < 1) return 0;
   return buf[0];
}

double GetMACD_Signal()
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_macd_handle, 1, 0, 1, buf) < 1) return 0;
   return buf[0];
}

double GetBB_Upper()
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_bb_handle, 1, 0, 1, buf) < 1) return 0;
   return buf[0];
}

double GetBB_Lower()
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_bb_handle, 2, 0, 1, buf) < 1) return 0;
   return buf[0];
}

double GetSMA20()
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_ma_20_handle, 0, 0, 1, buf) < 1) return 0;
   return buf[0];
}

//+------------------------------------------------------------------+
//| QUANTUM FEATURE HELPERS (for QFusion adapters)                    |
//+------------------------------------------------------------------+

// Price entropy: discretize returns into bins, compute Shannon entropy
double ComputePriceEntropy(const double &close[], int count, int window)
{
   if(count < window + 1) return 2.5;  // Default moderate entropy

   int n_bins = 8;
   int bins[];
   ArrayResize(bins, n_bins);
   ArrayInitialize(bins, 0);

   for(int i = 0; i < window && i + 1 < count; i++)
   {
      double ret = (close[i] - close[i+1]) / (close[i+1] + 1e-10);
      // Map return to bin [0, n_bins-1]
      int bin_idx = (int)MathFloor((MathTanh(ret * 50.0) + 1.0) / 2.0 * (n_bins - 1));
      bin_idx = MathMax(0, MathMin(n_bins - 1, bin_idx));
      bins[bin_idx]++;
   }

   double entropy = 0;
   for(int b = 0; b < n_bins; b++)
   {
      if(bins[b] > 0)
      {
         double p = (double)bins[b] / window;
         entropy -= p * MathLog(p) / MathLog(2.0);
      }
   }
   return entropy;
}

// Dominant state probability
double ComputeDominantState(const double &close[], int count, int window)
{
   if(count < window + 1) return 0.125;
   int n_bins = 8;
   int bins[];
   ArrayResize(bins, n_bins);
   ArrayInitialize(bins, 0);

   for(int i = 0; i < window && i + 1 < count; i++)
   {
      double ret = (close[i] - close[i+1]) / (close[i+1] + 1e-10);
      int bin_idx = (int)MathFloor((MathTanh(ret * 50.0) + 1.0) / 2.0 * (n_bins - 1));
      bin_idx = MathMax(0, MathMin(n_bins - 1, bin_idx));
      bins[bin_idx]++;
   }

   int max_count = 0;
   for(int b = 0; b < n_bins; b++)
      if(bins[b] > max_count) max_count = bins[b];

   return (double)max_count / window;
}

// Superposition: fraction of significant states
double ComputeSuperposition(const double &close[], int count, int window)
{
   if(count < window + 1) return 0.5;
   int n_bins = 8;
   int bins[];
   ArrayResize(bins, n_bins);
   ArrayInitialize(bins, 0);

   for(int i = 0; i < window && i + 1 < count; i++)
   {
      double ret = (close[i] - close[i+1]) / (close[i+1] + 1e-10);
      int bin_idx = (int)MathFloor((MathTanh(ret * 50.0) + 1.0) / 2.0 * (n_bins - 1));
      bin_idx = MathMax(0, MathMin(n_bins - 1, bin_idx));
      bins[bin_idx]++;
   }

   int significant = 0;
   for(int b = 0; b < n_bins; b++)
      if((double)bins[b] / window > 0.05) significant++;

   return (double)significant / n_bins;
}

// Coherence proxy
double ComputeCoherence(const double &close[], int count, int window)
{
   if(count < window + 1) return 0.5;
   // Coherence = consistency of direction
   int same_dir = 0;
   int dir_prev = 0;
   for(int i = 0; i < window && i + 1 < count; i++)
   {
      int dir = (close[i] > close[i+1]) ? 1 : -1;
      if(i > 0 && dir == dir_prev) same_dir++;
      dir_prev = dir;
   }
   return (double)same_dir / (window - 1);
}

// Pattern correlation between two offsets
double ComputePatternCorrelation(const double &close[], int offset1, int offset2, int length)
{
   double mean1 = 0, mean2 = 0;
   for(int i = 0; i < length; i++)
   {
      mean1 += close[offset1 + i];
      mean2 += close[offset2 + i];
   }
   mean1 /= length;
   mean2 /= length;

   double cov = 0, var1 = 0, var2 = 0;
   for(int i = 0; i < length; i++)
   {
      double d1 = close[offset1 + i] - mean1;
      double d2 = close[offset2 + i] - mean2;
      cov += d1 * d2;
      var1 += d1 * d1;
      var2 += d2 * d2;
   }

   double denom = MathSqrt(var1 * var2);
   if(denom < 1e-10) return 0;
   return cov / denom;
}

// Autocorrelation of returns at lag 1
double ComputeAutocorrelation(const double &close[], int count, int window)
{
   if(count < window + 2) return 0;
   double returns[];
   ArrayResize(returns, window);
   for(int i = 0; i < window && i + 1 < count; i++)
      returns[i] = (close[i] - close[i+1]) / (close[i+1] + 1e-10);

   double mean = 0;
   for(int i = 0; i < window; i++) mean += returns[i];
   mean /= window;

   double cov = 0, var_val = 0;
   for(int i = 0; i < window - 1; i++)
   {
      cov += (returns[i] - mean) * (returns[i+1] - mean);
      var_val += (returns[i] - mean) * (returns[i] - mean);
   }
   var_val += (returns[window-1] - mean) * (returns[window-1] - mean);

   if(var_val < 1e-10) return 0;
   return cov / var_val;
}

// Lag-N autocorrelation
double ComputeLagAutocorrelation(const double &close[], int count, int lag, int window)
{
   if(count < window + lag + 1) return 0;
   double returns[];
   ArrayResize(returns, window);
   for(int i = 0; i < window && i + 1 < count; i++)
      returns[i] = (close[i] - close[i+1]) / (close[i+1] + 1e-10);

   if(window <= lag) return 0;

   double mean = 0;
   for(int i = 0; i < window; i++) mean += returns[i];
   mean /= window;

   double cov = 0, var_val = 0;
   for(int i = 0; i < window - lag; i++)
   {
      cov += (returns[i] - mean) * (returns[i+lag] - mean);
   }
   for(int i = 0; i < window; i++)
      var_val += (returns[i] - mean) * (returns[i] - mean);

   if(var_val < 1e-10) return 0;
   return cov / var_val;
}

// Local compression ratio (redundancy measure)
double ComputeLocalCompression(const double &close[], int start, int length)
{
   if(length < 3) return 1.0;

   // Measure redundancy: how predictable is the sequence?
   double sum_diff = 0, sum_abs = 0;
   for(int i = start; i < start + length - 1; i++)
   {
      double diff = close[i] - close[i+1];
      sum_diff += diff;
      sum_abs += MathAbs(diff);
   }

   if(sum_abs < 1e-10) return 2.0;  // Perfectly compressible (flat)
   // High directional consistency = high compression ratio
   return MathAbs(sum_diff) / sum_abs + 1.0;
}

// Average volume
double ComputeAvgVolume(const long &volume[], int count, int window)
{
   double sum = 0;
   for(int i = 0; i < window && i < count; i++)
      sum += (double)volume[i];
   return (window > 0) ? sum / window : 0;
}

//+------------------------------------------------------------------+
//| METHYL DYE SUMMARY                                                |
//+------------------------------------------------------------------+
void PrintMethylDyeSummary()
{
   Print("================================================================");
   Print("  METHYL DYE TRADE RECORD SUMMARY");
   Print("  Blue (Trending): ", g_blue_count, " trades");
   Print("  Red (Choppy):    ", g_red_count, " trades");
   Print("  Total:           ", g_blue_count + g_red_count, " trades");
   Print("================================================================");
}
//+------------------------------------------------------------------+
