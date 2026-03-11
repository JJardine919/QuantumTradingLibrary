import MetaTrader5 as mt5
mt5.initialize()
for name in ['BTCUSD', 'XAUUSD', 'EURUSD', 'GBPUSD', 'US30', 'NAS100']:
    info = mt5.symbol_info(name)
    if info:
        tv = info.trade_tick_value
        ts = info.trade_tick_size
        pt = info.point
        cs = info.trade_contract_size
        sp = info.spread
        # cost of spread for 0.01 lot
        sp_cost = sp * tv * 0.01
        print(f"{name}: contract={cs} tick_val={tv} tick_size={ts} spread={sp} spread_cost_01lot=${sp_cost:.4f}")
mt5.shutdown()
