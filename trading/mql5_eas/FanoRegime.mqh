//+------------------------------------------------------------------+
//|                                                 FanoRegime.mqh   |
//|           Fano Plane Triple Coherence + Octonion Regime Engine   |
//|                                 QuantumChildren / DooDoo 2026    |
//+------------------------------------------------------------------+
#ifndef FANO_REGIME_MQH
#define FANO_REGIME_MQH

#include <Fano\FanoDecomposition.mqh>

//+------------------------------------------------------------------+
//| 7 Fano triples (0-indexed into the lookback array)               |
//|                                                                  |
//| Each triple (a,b,c) maps to three SMA lookback periods.          |
//| The Fano plane encodes the octonion multiplication structure:     |
//|   e_i * e_j = e_k  for each ordered triple (i,j,k).             |
//|                                                                  |
//| Triple coherence = signal[a] * signal[b] * signal[c]            |
//| When all three signals in a triple agree in sign, coherence is   |
//| strong and directional. Disagreement collapses coherence.        |
//+------------------------------------------------------------------+
const int FANO_TRIPLES[7][3] =
{
   {0, 1, 3},  // T1: (e1,e2,e4) -> lookbacks (5,8,21)
   {1, 2, 4},  // T2: (e2,e3,e5) -> lookbacks (8,13,34)
   {2, 3, 5},  // T3: (e3,e4,e6) -> lookbacks (13,21,55)
   {3, 4, 6},  // T4: (e4,e5,e7) -> lookbacks (21,34,89)
   {4, 5, 0},  // T5: (e5,e6,e1) -> lookbacks (34,55,5)
   {5, 6, 1},  // T6: (e6,e7,e2) -> lookbacks (55,89,8)
   {6, 0, 2}   // T7: (e7,e1,e3) -> lookbacks (89,5,13)
};

//--- Default Fibonacci lookback periods mapped to octonion basis e1..e7
const int FANO_LOOKBACKS_DEFAULT[7] = {5, 8, 13, 21, 34, 55, 89};

//+------------------------------------------------------------------+
//| FanoRegime — full regime detection via Fano triple coherence      |
//|                                                                  |
//| Pipeline:                                                        |
//|   1. 7 SMA signals normalized by ATR through tanh               |
//|   2. Signal deltas (momentum) from bar-to-bar change            |
//|   3. Octonion decomposition: Jordan (consensus) + Commutator     |
//|   4. 7 triple coherences from Fano plane structure               |
//|   5. Active triple = strongest |coherence|                       |
//|   6. Regime confidence = Jordan-triple agreement score           |
//|   7. Final direction with threshold + optional inversion         |
//+------------------------------------------------------------------+
struct FanoRegime
{
   //--- Signal state
   double   signals[7];           // current tanh-normalized SMA signals
   double   prev_signals[7];      // previous bar's signals
   double   dsignals[7];          // signals - prev_signals (momentum)

   //--- Triple coherence state
   double   coherence[7];         // triple coherence values
   int      active_triple;        // index of strongest |coherence|
   double   active_coherence;     // coherence of the active triple
   int      triple_direction;     // sign(active_coherence): +1=BUY, -1=SELL

   //--- Regime output
   double   regime_confidence;    // combined Jordan + triple confidence [-1, +1]
   int      final_direction;      // +1=BUY, -1=SELL, 0=FLAT (after threshold + inversion)

   //--- Octonion decomposition
   FanoDecomp decomp;

   //--- Indicator handles
   int      sma_handles[7];       // handles for the 7 SMA indicators
   int      atr_handle;           // ATR(14) handle

   //--- Configuration
   int      lookbacks[7];         // actual lookback periods (customizable)
   string   m_symbol;             // symbol this regime is bound to
   ENUM_TIMEFRAMES m_timeframe;   // timeframe this regime is bound to
   bool     initialized;          // whether Init() succeeded

   //+------------------------------------------------------------------+
   //| Init — create indicator handles and prepare state                |
   //|                                                                  |
   //| Parameters:                                                      |
   //|   symbol — chart symbol (e.g. "BTCUSD")                        |
   //|   tf     — timeframe (e.g. PERIOD_M1)                           |
   //|   lb[]   — custom lookback periods (7 elements), or empty to    |
   //|            use FANO_LOOKBACKS_DEFAULT                            |
   //|                                                                  |
   //| Returns: true if all handles created successfully               |
   //+------------------------------------------------------------------+
   bool Init(string symbol, ENUM_TIMEFRAMES tf, const int &lb[])
   {
      m_symbol    = symbol;
      m_timeframe = tf;
      initialized = false;

      //--- Set lookback periods: custom or default
      if(ArraySize(lb) >= 7)
      {
         for(int i = 0; i < 7; i++)
            lookbacks[i] = lb[i];
      }
      else
      {
         for(int i = 0; i < 7; i++)
            lookbacks[i] = FANO_LOOKBACKS_DEFAULT[i];
      }

      //--- Create 7 SMA indicator handles
      for(int i = 0; i < 7; i++)
      {
         sma_handles[i] = iMA(m_symbol, m_timeframe, lookbacks[i], 0, MODE_SMA, PRICE_CLOSE);
         if(sma_handles[i] == INVALID_HANDLE)
         {
            PrintFormat("[FanoRegime] FAILED to create SMA(%d) handle for %s %s",
                        lookbacks[i], m_symbol, EnumToString(m_timeframe));
            Deinit();
            return false;
         }
      }

      //--- Create ATR(14) handle
      atr_handle = iATR(m_symbol, m_timeframe, 14);
      if(atr_handle == INVALID_HANDLE)
      {
         PrintFormat("[FanoRegime] FAILED to create ATR(14) handle for %s %s",
                     m_symbol, EnumToString(m_timeframe));
         Deinit();
         return false;
      }

      //--- Zero out signal arrays
      ArrayInitialize(signals, 0.0);
      ArrayInitialize(prev_signals, 0.0);
      ArrayInitialize(dsignals, 0.0);
      ArrayInitialize(coherence, 0.0);

      //--- Zero out state
      active_triple     = 0;
      active_coherence  = 0.0;
      triple_direction  = 0;
      regime_confidence = 0.0;
      final_direction   = 0;

      initialized = true;

      PrintFormat("[FanoRegime] Initialized: %s %s | Lookbacks: %d,%d,%d,%d,%d,%d,%d",
                  m_symbol, EnumToString(m_timeframe),
                  lookbacks[0], lookbacks[1], lookbacks[2], lookbacks[3],
                  lookbacks[4], lookbacks[5], lookbacks[6]);

      return true;
   }

   //+------------------------------------------------------------------+
   //| Deinit — release all indicator handles                           |
   //+------------------------------------------------------------------+
   void Deinit()
   {
      for(int i = 0; i < 7; i++)
      {
         if(sma_handles[i] != INVALID_HANDLE && sma_handles[i] != 0)
         {
            IndicatorRelease(sma_handles[i]);
            sma_handles[i] = INVALID_HANDLE;
         }
      }

      if(atr_handle != INVALID_HANDLE && atr_handle != 0)
      {
         IndicatorRelease(atr_handle);
         atr_handle = INVALID_HANDLE;
      }

      initialized = false;
   }

   //+------------------------------------------------------------------+
   //| Update — run the full signal pipeline                            |
   //|                                                                  |
   //| Parameters:                                                      |
   //|   confidence_threshold — minimum |regime_confidence| to produce |
   //|                          a directional signal (typical: 0.15)    |
   //|   inverted_mode        — if true, flip the regime direction      |
   //|                          (for mean-reversion or fade logic)      |
   //|                                                                  |
   //| Pipeline:                                                        |
   //|   1. Shift signals to prev_signals                              |
   //|   2. Get current close price                                    |
   //|   3. Get ATR(14) for normalization                              |
   //|   4. Compute 7 SMA-based signals: tanh((close - SMA) / ATR)    |
   //|   5. Compute momentum deltas                                    |
   //|   6. Run octonion decomposition (Jordan/Commutator/Associator)  |
   //|   7. Compute 7 Fano triple coherences                          |
   //|   8. Find active triple (highest |coherence|)                   |
   //|   9. Combine Jordan direction + triple coherence into regime    |
   //|  10. Apply inversion mode if enabled                            |
   //|  11. Threshold to final direction: BUY / SELL / FLAT            |
   //+------------------------------------------------------------------+
   void Update(double confidence_threshold, bool inverted_mode)
   {
      if(!initialized)
         return;

      //--- 1. Copy current signals to previous
      for(int i = 0; i < 7; i++)
         prev_signals[i] = signals[i];

      //--- 2. Get current close price
      double close_arr[];
      ArraySetAsSeries(close_arr, true);
      if(CopyClose(m_symbol, m_timeframe, 0, 2, close_arr) < 2)
         return;
      double close_now = close_arr[0];

      //--- 3. Get ATR(14) value for normalization
      double atr_buf[];
      ArraySetAsSeries(atr_buf, true);
      if(CopyBuffer(atr_handle, 0, 0, 1, atr_buf) < 1)
         return;
      double atr_val = atr_buf[0];

      //--- Protect against zero/near-zero ATR (illiquid or no data)
      if(atr_val < 1e-10)
         return;

      //--- 4. Get 7 SMA values and compute tanh-normalized signals
      for(int i = 0; i < 7; i++)
      {
         double sma_buf[];
         ArraySetAsSeries(sma_buf, true);
         if(CopyBuffer(sma_handles[i], 0, 0, 1, sma_buf) < 1)
            return;

         double raw = (close_now - sma_buf[0]) / atr_val;
         signals[i] = MathTanh(raw);
      }

      //--- 5. Compute signal deltas (momentum)
      for(int i = 0; i < 7; i++)
         dsignals[i] = signals[i] - prev_signals[i];

      //--- 6. Run octonion decomposition (Jordan + Commutator + Associator)
      decomp.Compute(signals, dsignals);

      //--- 7. Compute 7 Fano triple coherences
      //    Each coherence = product of three signals from the Fano triple.
      //    Strong coherence means the three timeframe signals agree directionally.
      for(int k = 0; k < 7; k++)
      {
         int a = FANO_TRIPLES[k][0];
         int b = FANO_TRIPLES[k][1];
         int c = FANO_TRIPLES[k][2];
         coherence[k] = signals[a] * signals[b] * signals[c];
      }

      //--- 8. Find active triple (highest absolute coherence)
      active_triple = 0;
      double max_abs = MathAbs(coherence[0]);
      for(int k = 1; k < 7; k++)
      {
         double abs_c = MathAbs(coherence[k]);
         if(abs_c > max_abs)
         {
            max_abs       = abs_c;
            active_triple = k;
         }
      }
      active_coherence  = coherence[active_triple];
      triple_direction  = (active_coherence > 0) ? 1 : -1;

      //--- 9. Compute regime confidence
      //    Jordan ratio: how much of the decomposition energy is consensus vs. conflict.
      //    Agreement: +1 if Jordan and triple point same direction, -1 if they disagree.
      //    Final confidence = ratio * |coherence| * agreement, bounded roughly in [-1, +1].
      double denom        = decomp.jordan_strength + decomp.commutator_strength + 1e-10;
      double jordan_ratio = decomp.jordan_strength / denom;
      double agreement    = (decomp.jordan_direction == triple_direction) ? 1.0 : -1.0;
      regime_confidence   = jordan_ratio * MathAbs(active_coherence) * agreement;

      //--- 10. Apply inversion if needed (mean-reversion / fade mode)
      if(inverted_mode)
         regime_confidence = -regime_confidence;

      //--- 11. Final direction decision with threshold
      if(regime_confidence >= confidence_threshold)
         final_direction = inverted_mode ? -triple_direction : triple_direction;
      else if(regime_confidence <= -confidence_threshold)
         final_direction = inverted_mode ? triple_direction : -triple_direction;
      else
         final_direction = 0;  // FLAT — no trade
   }
};

#endif // FANO_REGIME_MQH
