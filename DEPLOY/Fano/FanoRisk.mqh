//+------------------------------------------------------------------+
//|                                                  FanoRisk.mqh   |
//|                  Risk & Drawdown Protection for Grid Trading     |
//|                                    QuantumChildren / DooDoo 2026 |
//+------------------------------------------------------------------+
#ifndef FANO_RISK_MQH
#define FANO_RISK_MQH

//+------------------------------------------------------------------+
//| FanoRisk — daily & total drawdown guard                          |
//|                                                                  |
//| Encapsulates the risk state that BG_AtlasGrid carried as loose  |
//| globals. Drop into any grid EA:                                  |
//|                                                                  |
//|   FanoRisk risk;                                                 |
//|   risk.Init();                  // in OnInit                     |
//|   risk.CheckDailyReset();       // top of OnTick                 |
//|   risk.UpdateHWM();             // after balance changes         |
//|   if(!risk.IsSafe(4.5, 9.0))    // before opening trades         |
//|      return;                                                     |
//+------------------------------------------------------------------+
struct FanoRisk
{
   //--- state
   double   start_balance;       // captured once on Init()
   double   high_water_mark;     // peak equity seen
   double   daily_start_balance; // reset each trading day
   datetime last_day_reset;      // when daily balance was last reset
   bool     blocked;             // true if a risk limit was hit
   string   block_reason;        // human-readable reason

   //+---------------------------------------------------------------+
   //| Init — call once from OnInit                                  |
   //+---------------------------------------------------------------+
   void Init()
   {
      start_balance       = AccountInfoDouble(ACCOUNT_BALANCE);
      high_water_mark     = AccountInfoDouble(ACCOUNT_EQUITY);
      daily_start_balance = start_balance;
      last_day_reset      = TimeCurrent();
      blocked             = false;
      block_reason        = "";

      Print("[FanoRisk] Init | balance=$", DoubleToString(start_balance, 2),
            " | equity=$", DoubleToString(high_water_mark, 2));
   }

   //+---------------------------------------------------------------+
   //| CheckDailyReset — call each tick / check cycle                |
   //|  Detects server-day rollover and resets daily baseline.       |
   //+---------------------------------------------------------------+
   void CheckDailyReset()
   {
      MqlDateTime now, last;
      TimeToStruct(TimeCurrent(), now);
      TimeToStruct(last_day_reset, last);

      if(now.day != last.day || now.mon != last.mon || now.year != last.year)
      {
         daily_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
         last_day_reset      = TimeCurrent();
         blocked             = false;
         block_reason        = "";

         Print("[FanoRisk] Daily reset | new baseline=$",
               DoubleToString(daily_start_balance, 2));
      }
   }

   //+---------------------------------------------------------------+
   //| DailyDD — current daily drawdown as a percentage              |
   //|  Positive = losing money relative to day start.               |
   //+---------------------------------------------------------------+
   double DailyDD()
   {
      if(daily_start_balance <= 0.0)
         return 0.0;

      double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      return ((daily_start_balance - current_equity) / daily_start_balance) * 100.0;
   }

   //+---------------------------------------------------------------+
   //| TotalDD — drawdown from high-water mark as a percentage       |
   //|  Also updates HWM if equity made a new high.                  |
   //+---------------------------------------------------------------+
   double TotalDD()
   {
      double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);

      if(current_equity > high_water_mark)
         high_water_mark = current_equity;

      if(high_water_mark <= 0.0)
         return 0.0;

      return ((high_water_mark - current_equity) / high_water_mark) * 100.0;
   }

   //+---------------------------------------------------------------+
   //| IsSafe — returns true if both DD limits are respected         |
   //|                                                               |
   //|  daily_dd_limit  — max allowed daily DD %  (e.g. 4.5)        |
   //|  max_dd_limit    — max allowed total DD %  (e.g. 9.0)        |
   //|                                                               |
   //| Side effects: sets blocked + block_reason + prints warning   |
   //| if a limit is breached.                                       |
   //+---------------------------------------------------------------+
   bool IsSafe(double daily_dd_limit, double max_dd_limit)
   {
      double dd_daily = DailyDD();
      if(dd_daily >= daily_dd_limit)
      {
         blocked      = true;
         block_reason = StringFormat("Daily DD %.2f%% >= limit %.2f%%",
                                     dd_daily, daily_dd_limit);
         Print("[FanoRisk] BLOCKED: ", block_reason);
         return false;
      }

      double dd_total = TotalDD();
      if(dd_total >= max_dd_limit)
      {
         blocked      = true;
         block_reason = StringFormat("Total DD %.2f%% >= limit %.2f%%",
                                     dd_total, max_dd_limit);
         Print("[FanoRisk] BLOCKED: ", block_reason);
         return false;
      }

      return true;
   }

   //+---------------------------------------------------------------+
   //| UpdateHWM — standalone high-water mark refresh                |
   //|  Call after profitable closes or whenever convenient.          |
   //|  TotalDD() also updates HWM internally, so this is           |
   //|  only needed if you want to update without computing DD.      |
   //+---------------------------------------------------------------+
   void UpdateHWM()
   {
      double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(current_equity > high_water_mark)
         high_water_mark = current_equity;
   }
};

#endif // FANO_RISK_MQH
