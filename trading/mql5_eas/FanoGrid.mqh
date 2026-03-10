//+------------------------------------------------------------------+
//|                                                    FanoGrid.mqh  |
//|                        Fano Superposition Grid - Position Mgmt   |
//|                              Grid position management module     |
//+------------------------------------------------------------------+
#ifndef FANO_GRID_MQH
#define FANO_GRID_MQH

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| GridPosition — single position in the grid                       |
//+------------------------------------------------------------------+
struct GridPosition
{
   ulong    ticket;          // position ticket
   double   entry_price;     // entry fill price
   double   hidden_sl;       // internal SL price (not sent to broker)
   double   hidden_tp;       // internal TP price (not sent to broker)
   double   partial_tp;      // partial TP price (DYNTP%)
   bool     partial_closed;  // whether 50% already taken
   double   lot;             // position lot size
   int      level;           // grid level (0 = first entry)
   int      direction;       // +1=BUY, -1=SELL
   datetime open_time;       // when opened
};

//+------------------------------------------------------------------+
//| FanoGrid — grid position manager                                 |
//+------------------------------------------------------------------+
struct FanoGrid
{
   //--- State
   GridPosition  positions[20];       // fixed array, max 20 grid levels
   int           count;               // current number of open grid positions
   int           magic;               // magic number for this EA
   string        symbol;              // trading symbol
   double        base_lot;            // computed lot size ($1 risk)
   double        sl_distance;         // SL distance in price units
   double        tp_distance;         // TP distance in price units
   double        partial_tp_distance; // partial TP distance in price units
   double        grid_spacing;        // distance between grid levels
   double        breakeven_trigger;   // distance to trigger SL-to-breakeven

   //--- Methods
   void   Init(string sym, int mag);
   void   ComputeLotAndDistances(double atr, double chaos_level, double sl_dollars,
                                 double tp_dollars, double dyntp_pct,
                                 double atr_mult, double rollsl_mult);
   bool   CanOpenNew(int max_positions, int direction);
   bool   OpenPosition(int direction, CTrade &trade);
   void   ManagePositions(CTrade &trade);
   void   RemovePosition(int idx);
   void   SyncWithBroker(void);
   double LastEntryPrice(void);
   bool   PriceReachedNextLevel(double current_price, int direction);
   double TotalPnL(void);
};

//+------------------------------------------------------------------+
//| Init — initialize grid for a symbol and magic number             |
//+------------------------------------------------------------------+
void FanoGrid::Init(string sym, int mag)
{
   symbol  = sym;
   magic   = mag;
   count   = 0;

   base_lot            = 0.0;
   sl_distance         = 0.0;
   tp_distance         = 0.0;
   partial_tp_distance = 0.0;
   grid_spacing        = 0.0;
   breakeven_trigger   = 0.0;

   ZeroMemory(positions);
}

//+------------------------------------------------------------------+
//| ComputeLotAndDistances — THE CORE SCALING FORMULA                |
//|   Derives lot size, SL/TP distances, grid spacing from ATR and   |
//|   chaos level. Risk is pinned to $sl_dollars per trade.          |
//+------------------------------------------------------------------+
void FanoGrid::ComputeLotAndDistances(double atr, double chaos_level,
                                      double sl_dollars, double tp_dollars,
                                      double dyntp_pct, double atr_mult,
                                      double rollsl_mult)
{
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double lot_step   = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double min_lot    = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_lot    = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);

   //--- SL distance in price
   sl_distance = atr * atr_mult;

   //--- SL distance in ticks
   double sl_ticks = sl_distance / tick_size;

   //--- Base lot: risk exactly $sl_dollars per trade
   //    sl_dollars = base_lot * sl_ticks * tick_value
   //    => base_lot = sl_dollars / (sl_ticks * tick_value)
   if(sl_ticks * tick_value > 0)
      base_lot = sl_dollars / (sl_ticks * tick_value);
   else
      base_lot = min_lot;

   //--- Normalize to broker constraints
   base_lot = MathFloor(base_lot / lot_step) * lot_step;
   base_lot = MathMax(base_lot, min_lot);
   base_lot = MathMin(base_lot, max_lot);

   //--- TP distance in price
   //    tp_dollars = base_lot * tp_ticks * tick_value
   //    tp_distance = tp_dollars / (base_lot * tick_value / tick_size)
   if(base_lot * tick_value > 0)
      tp_distance = (tp_dollars * tick_size) / (base_lot * tick_value);
   else
      tp_distance = sl_distance * 3.0;

   //--- Partial TP distance
   partial_tp_distance = tp_distance * (dyntp_pct / 100.0);

   //--- Grid spacing: ATR * ROLLSLMULT, widened by chaos
   double chaos_norm  = MathMin(chaos_level / 5.0, 10.0);
   double spacing_mult = 1.0 + 0.2 * chaos_norm;
   grid_spacing = atr * rollsl_mult * spacing_mult;

   //--- Breakeven trigger
   breakeven_trigger = sl_distance * rollsl_mult;
}

//+------------------------------------------------------------------+
//| CanOpenNew — check if a new grid level can be opened             |
//+------------------------------------------------------------------+
bool FanoGrid::CanOpenNew(int max_positions, int direction)
{
   if(count >= max_positions)
      return false;
   if(count > 0 && positions[0].direction != direction)
      return false;   // don't flip mid-grid
   return true;
}

//+------------------------------------------------------------------+
//| OpenPosition — send order and register in grid array             |
//+------------------------------------------------------------------+
bool FanoGrid::OpenPosition(int direction, CTrade &trade)
{
   double price = (direction == 1)
                  ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                  : SymbolInfoDouble(symbol, SYMBOL_BID);

   bool ok = false;
   if(direction == 1)
      ok = trade.Buy(base_lot, symbol, price, 0, 0,
                     StringFormat("FANO L%d", count));
   else
      ok = trade.Sell(base_lot, symbol, price, 0, 0,
                      StringFormat("FANO L%d", count));

   if(ok)
   {
      ulong ticket = trade.ResultOrder();
      if(ticket > 0)
      {
         int idx = count;
         positions[idx].ticket        = ticket;
         positions[idx].entry_price   = trade.ResultPrice();
         positions[idx].lot           = base_lot;
         positions[idx].level         = count;
         positions[idx].direction     = direction;
         positions[idx].open_time     = TimeCurrent();
         positions[idx].partial_closed = false;

         //--- Set hidden SL/TP
         if(direction == 1)
         {
            positions[idx].hidden_sl  = positions[idx].entry_price - sl_distance;
            positions[idx].hidden_tp  = positions[idx].entry_price + tp_distance;
            positions[idx].partial_tp = positions[idx].entry_price + partial_tp_distance;
         }
         else
         {
            positions[idx].hidden_sl  = positions[idx].entry_price + sl_distance;
            positions[idx].hidden_tp  = positions[idx].entry_price - tp_distance;
            positions[idx].partial_tp = positions[idx].entry_price - partial_tp_distance;
         }

         count++;
         Print(StringFormat("[FanoGrid] Opened L%d %s %.4f @ %.5f | SL=%.5f TP=%.5f",
               idx, direction == 1 ? "BUY" : "SELL", base_lot,
               positions[idx].entry_price,
               positions[idx].hidden_sl, positions[idx].hidden_tp));
         return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| ManagePositions — called EVERY TICK                              |
//|   Checks hidden SL, hidden TP, partial TP, breakeven for all     |
//|   grid positions. Reverse iterates for safe removal.             |
//+------------------------------------------------------------------+
void FanoGrid::ManagePositions(CTrade &trade)
{
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);

   for(int i = count - 1; i >= 0; i--)   // reverse iterate for safe removal
   {
      //--- Verify position still exists with broker
      if(!PositionSelectByTicket(positions[i].ticket))
      {
         RemovePosition(i);
         continue;
      }

      double entry = positions[i].entry_price;
      int    dir   = positions[i].direction;

      //--- Hidden SL check
      if(dir == 1 && bid <= positions[i].hidden_sl)
      {
         trade.PositionClose(positions[i].ticket);
         Print(StringFormat("[FanoGrid] SL HIT L%d @ %.5f", i, bid));
         RemovePosition(i);
         continue;
      }
      if(dir == -1 && ask >= positions[i].hidden_sl)
      {
         trade.PositionClose(positions[i].ticket);
         Print(StringFormat("[FanoGrid] SL HIT L%d @ %.5f", i, ask));
         RemovePosition(i);
         continue;
      }

      //--- Hidden TP check (full close)
      if(dir == 1 && bid >= positions[i].hidden_tp)
      {
         trade.PositionClose(positions[i].ticket);
         Print(StringFormat("[FanoGrid] TP HIT L%d @ %.5f (+$3)", i, bid));
         RemovePosition(i);
         continue;
      }
      if(dir == -1 && ask <= positions[i].hidden_tp)
      {
         trade.PositionClose(positions[i].ticket);
         Print(StringFormat("[FanoGrid] TP HIT L%d @ %.5f (+$3)", i, ask));
         RemovePosition(i);
         continue;
      }

      //--- Partial TP check (50% close)
      if(!positions[i].partial_closed)
      {
         bool partial_hit = false;
         if(dir == 1  && bid >= positions[i].partial_tp) partial_hit = true;
         if(dir == -1 && ask <= positions[i].partial_tp) partial_hit = true;

         if(partial_hit)
         {
            double vol_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
            double vol_min  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
            double half_lot = MathFloor((positions[i].lot / 2.0) / vol_step) * vol_step;

            if(half_lot >= vol_min)
            {
               trade.PositionClosePartial(positions[i].ticket, half_lot);
               positions[i].partial_closed = true;
               positions[i].hidden_sl      = entry;   // move SL to breakeven
               positions[i].lot           -= half_lot;
               Print(StringFormat("[FanoGrid] Partial TP L%d, closed %.4f, SL->BE", i, half_lot));
            }
         }
      }

      //--- Breakeven trigger (for positions that haven't had partial close)
      if(!positions[i].partial_closed)
      {
         if(dir == 1  && bid >= entry + breakeven_trigger)
            positions[i].hidden_sl = entry;
         if(dir == -1 && ask <= entry - breakeven_trigger)
            positions[i].hidden_sl = entry;
      }
   }
}

//+------------------------------------------------------------------+
//| RemovePosition — shift remaining positions down after removal    |
//+------------------------------------------------------------------+
void FanoGrid::RemovePosition(int idx)
{
   for(int j = idx; j < count - 1; j++)
      positions[j] = positions[j + 1];
   count--;
}

//+------------------------------------------------------------------+
//| SyncWithBroker — rebuild grid from actual open positions on init |
//+------------------------------------------------------------------+
void FanoGrid::SyncWithBroker(void)
{
   count = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != magic) continue;

      if(count >= 20) break;

      positions[count].ticket        = ticket;
      positions[count].entry_price   = PositionGetDouble(POSITION_PRICE_OPEN);
      positions[count].lot           = PositionGetDouble(POSITION_VOLUME);
      positions[count].direction     = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
      positions[count].open_time     = (datetime)PositionGetInteger(POSITION_TIME);
      positions[count].level         = count;
      positions[count].partial_closed = false;
      //--- Set default hidden SL/TP (will be recalculated on next ComputeLotAndDistances call)
      positions[count].hidden_sl     = 0;
      positions[count].hidden_tp     = 0;
      positions[count].partial_tp    = 0;
      count++;
   }

   Print(StringFormat("[FanoGrid] Synced %d positions from broker", count));
}

//+------------------------------------------------------------------+
//| LastEntryPrice — return entry price of most recent position      |
//+------------------------------------------------------------------+
double FanoGrid::LastEntryPrice(void)
{
   if(count <= 0) return 0.0;
   return positions[count - 1].entry_price;
}

//+------------------------------------------------------------------+
//| PriceReachedNextLevel — has price moved grid_spacing AGAINST us? |
//+------------------------------------------------------------------+
bool FanoGrid::PriceReachedNextLevel(double current_price, int direction)
{
   if(count == 0) return false;

   double last = LastEntryPrice();

   if(direction == 1)
      return (current_price <= last - grid_spacing);   // price dropped
   else
      return (current_price >= last + grid_spacing);   // price rose
}

//+------------------------------------------------------------------+
//| TotalPnL — sum floating P&L of all grid positions                |
//+------------------------------------------------------------------+
double FanoGrid::TotalPnL(void)
{
   double total = 0.0;

   for(int i = 0; i < count; i++)
   {
      if(PositionSelectByTicket(positions[i].ticket))
         total += PositionGetDouble(POSITION_PROFIT);
   }

   return total;
}

#endif // FANO_GRID_MQH
