//+------------------------------------------------------------------+
//|                                              FanoBayesian.mqh    |
//|        Bayesian Pattern Association Voting for Fano Grid EA      |
//|                                 QuantumChildren / DooDoo 2026    |
//+------------------------------------------------------------------+
//|                                                                  |
//| Each of the 7 Fano lookback points maintains a table of 32       |
//| binary patterns (5-bit encoding of up/down sequences). For each  |
//| pattern, we track how often the next bar moved UP vs DOWN.       |
//|                                                                  |
//| Voting uses Laplace-smoothed Bayesian probability:               |
//|   P(up | pattern) = (count_up + 1) / (total + 2)                |
//|   Vote = P(up) - 0.5   (range: [-0.5, +0.5])                    |
//|                                                                  |
//| A triple vote averages the non-abstaining votes from the 3       |
//| lookback points in the active Fano triple, giving the EA a       |
//| pattern-based directional bias that respects the Fano geometry.  |
//+------------------------------------------------------------------+
#ifndef FANO_BAYESIAN_MQH
#define FANO_BAYESIAN_MQH

#include <Fano\FanoRegime.mqh>

//--- Pattern encoding constants
#define FANO_PATTERN_LEN    5    // 5-bit binary patterns (5 consecutive bars)
#define FANO_PATTERN_COUNT  32   // 2^5 = 32 possible patterns

//+------------------------------------------------------------------+
//| PatternCounter — per-lookback hit table for all 32 patterns      |
//|                                                                  |
//| count_up[p]   = times next bar was UP after pattern p appeared   |
//| count_down[p] = times next bar was DOWN after pattern p appeared |
//+------------------------------------------------------------------+
struct PatternCounter
{
   int count_up[32];
   int count_down[32];

   void Zero()
   {
      ArrayInitialize(count_up, 0);
      ArrayInitialize(count_down, 0);
   }
};

//+------------------------------------------------------------------+
//| FanoBayesian — Bayesian pattern association voting engine         |
//|                                                                  |
//| Usage:                                                           |
//|   1. Call Init() once at EA start                                |
//|   2. Optionally call LoadFromFile() to restore learned counters  |
//|   3. Call UpdateCounters() each bar to learn from new data       |
//|   4. Call GetTripleVote() to get the Bayesian vote for the       |
//|      active Fano triple                                          |
//|   5. Call SaveToFile() periodically or at EA shutdown             |
//|                                                                  |
//| All close[] arrays MUST be ArraySetAsSeries(close, true) —       |
//| index 0 is the most recent bar.                                  |
//+------------------------------------------------------------------+
struct FanoBayesian
{
   //--- State
   PatternCounter counters[7];   // one counter table per Fano lookback point
   int            min_samples;   // minimum observations before a pattern vote counts

   //+------------------------------------------------------------------+
   //| Init — zero all counters and set minimum sample threshold        |
   //|                                                                  |
   //| min_samp: patterns with fewer than this many observations will   |
   //|           abstain (return 0.0) instead of voting. Default 20.    |
   //+------------------------------------------------------------------+
   void Init(int min_samp = 20)
   {
      for(int i = 0; i < 7; i++)
         counters[i].Zero();

      min_samples = min_samp;
   }

   //+------------------------------------------------------------------+
   //| EncodePattern — encode 5 consecutive bars as a 5-bit integer     |
   //|                                                                  |
   //| Reads close[start] through close[start + FANO_PATTERN_LEN].     |
   //| Since close[] is series (0 = newest), start + i is older than    |
   //| start + i + 1 is... wait, no: in series mode, higher index =    |
   //| older bar. So close[start+i] is older than close[start+i-1].    |
   //|                                                                  |
   //| Bit i is SET if close[start+i] > close[start+i+1], meaning      |
   //| the bar at offset (start+i) closed higher than the bar before    |
   //| it (the older bar at start+i+1). This encodes an UP move.       |
   //|                                                                  |
   //| Returns: integer in [0, 31]                                      |
   //+------------------------------------------------------------------+
   int EncodePattern(const double &close[], int start)
   {
      int pattern = 0;
      for(int i = 0; i < FANO_PATTERN_LEN; i++)
      {
         if(close[start + i] > close[start + i + 1])
            pattern |= (1 << i);
      }
      return pattern;
   }

   //+------------------------------------------------------------------+
   //| UpdateCounters — learn from the current bar's historical data    |
   //|                                                                  |
   //| For each of the 7 lookback points, looks at the 5-bar pattern   |
   //| that existed 'lookback' bars ago and records whether the next    |
   //| bar (one bar after the pattern) went UP or DOWN.                |
   //|                                                                  |
   //| close[] must be series-ordered with enough history.              |
   //| bars_available = total bars in the close[] array.                |
   //+------------------------------------------------------------------+
   void UpdateCounters(const double &close[], int bars_available)
   {
      for(int lb_idx = 0; lb_idx < 7; lb_idx++)
      {
         int lb = FANO_LOOKBACKS_DEFAULT[lb_idx];

         //--- Need: close[lb + FANO_PATTERN_LEN] accessible, plus the
         //--- outcome bar at close[lb - 1] and close[lb].
         //--- Minimum bars: lb + FANO_PATTERN_LEN + 2
         if(bars_available < lb + FANO_PATTERN_LEN + 2)
            continue;

         //--- Pattern that existed 'lb' bars ago
         int pat = EncodePattern(close, lb);

         //--- Outcome: did the bar immediately after the pattern go up?
         //--- In series mode: index lb-1 is one bar NEWER than index lb.
         //--- So if close[lb-1] > close[lb], the bar after the pattern went up.
         if(close[lb - 1] > close[lb])
            counters[lb_idx].count_up[pat]++;
         else
            counters[lb_idx].count_down[pat]++;
      }
   }

   //+------------------------------------------------------------------+
   //| GetVote — Bayesian vote for current pattern at one lookback      |
   //|                                                                  |
   //| lookback_idx: index into the 7-element arrays (0-6)              |
   //| close[]: series-ordered close prices (index 0 = newest)          |
   //|                                                                  |
   //| Returns:                                                         |
   //|   [-0.5, +0.5] — directional bias from learned pattern stats    |
   //|   0.0          — abstain (not enough samples for this pattern)   |
   //+------------------------------------------------------------------+
   double GetVote(int lookback_idx, const double &close[])
   {
      //--- Encode the current 5-bar pattern starting at bar 0
      int pat  = EncodePattern(close, 0);

      int up   = counters[lookback_idx].count_up[pat];
      int dn   = counters[lookback_idx].count_down[pat];
      int total = up + dn;

      //--- Not enough data for this pattern — abstain
      if(total < min_samples)
         return 0.0;

      //--- Laplace-smoothed Bayesian probability
      double p_up = (up + 1.0) / (total + 2.0);

      //--- Center around zero: positive = bullish bias, negative = bearish
      return p_up - 0.5;
   }

   //+------------------------------------------------------------------+
   //| GetTripleVote — average Bayesian vote across a Fano triple       |
   //|                                                                  |
   //| triple_idx: which of the 7 Fano triples (0-6) to vote on        |
   //| close[]: series-ordered close prices                             |
   //|                                                                  |
   //| Averages the non-abstaining votes from the 3 lookback points    |
   //| in the specified triple. If all 3 abstain, returns 0.0.         |
   //|                                                                  |
   //| Returns: [-0.5, +0.5] directional bias, or 0.0 if no votes     |
   //+------------------------------------------------------------------+
   double GetTripleVote(int triple_idx, const double &close[])
   {
      double sum   = 0.0;
      int    valid = 0;

      for(int i = 0; i < 3; i++)
      {
         int lb_idx = FANO_TRIPLES[triple_idx][i];
         double v   = GetVote(lb_idx, close);

         //--- Only count non-abstaining votes
         if(v != 0.0)
         {
            sum += v;
            valid++;
         }
      }

      if(valid == 0)
         return 0.0;

      return sum / valid;
   }

   //+------------------------------------------------------------------+
   //| SaveToFile — persist all pattern counters to binary file         |
   //|                                                                  |
   //| Writes 7 lookbacks * 32 patterns * 2 counters (up/down) = 448   |
   //| integers to disk. File is in MQL5\Files\ directory.             |
   //|                                                                  |
   //| Returns: true if write succeeded                                |
   //+------------------------------------------------------------------+
   bool SaveToFile(string filename)
   {
      int handle = FileOpen(filename, FILE_WRITE | FILE_BIN);
      if(handle == INVALID_HANDLE)
      {
         PrintFormat("[FanoBayesian] SaveToFile FAILED: cannot open %s (error %d)",
                     filename, GetLastError());
         return false;
      }

      for(int i = 0; i < 7; i++)
      {
         for(int j = 0; j < FANO_PATTERN_COUNT; j++)
         {
            FileWriteInteger(handle, counters[i].count_up[j]);
            FileWriteInteger(handle, counters[i].count_down[j]);
         }
      }

      FileClose(handle);
      return true;
   }

   //+------------------------------------------------------------------+
   //| LoadFromFile — restore pattern counters from binary file         |
   //|                                                                  |
   //| If the file does not exist, returns false and counters remain    |
   //| zeroed (cold start — the system will learn from scratch).       |
   //|                                                                  |
   //| Returns: true if load succeeded, false if file missing/error    |
   //+------------------------------------------------------------------+
   bool LoadFromFile(string filename)
   {
      if(!FileIsExist(filename))
         return false;

      int handle = FileOpen(filename, FILE_READ | FILE_BIN);
      if(handle == INVALID_HANDLE)
      {
         PrintFormat("[FanoBayesian] LoadFromFile FAILED: cannot open %s (error %d)",
                     filename, GetLastError());
         return false;
      }

      for(int i = 0; i < 7; i++)
      {
         for(int j = 0; j < FANO_PATTERN_COUNT; j++)
         {
            counters[i].count_up[j]   = FileReadInteger(handle);
            counters[i].count_down[j] = FileReadInteger(handle);
         }
      }

      FileClose(handle);

      PrintFormat("[FanoBayesian] Loaded counters from %s", filename);
      return true;
   }
};

#endif // FANO_BAYESIAN_MQH
