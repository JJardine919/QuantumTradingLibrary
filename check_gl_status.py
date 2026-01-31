import MetaTrader5 as mt5

# Try to connect
if not mt5.initialize(path=r'C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe'):
    print(f'FAILED to connect: {mt5.last_error()}')
else:
    info = mt5.terminal_info()
    acc = mt5.account_info()
    print(f'Terminal: {info.name}')
    print(f'Connected: {info.connected}')
    if acc:
        print(f'Account: {acc.login}')
        print(f'Balance: ${acc.balance:.2f}')
        print(f'Equity: ${acc.equity:.2f}')
        print(f'Trade Allowed: {acc.trade_allowed}')

    # Check positions
    positions = mt5.positions_get()
    if positions:
        print(f'\nOpen Positions ({len(positions)}):')
        for p in positions:
            ptype = "BUY" if p.type==0 else "SELL"
            print(f'  {p.symbol} {ptype} {p.volume} @ {p.price_open:.2f} | Magic: {p.magic} | P/L: ${p.profit:.2f}')
    else:
        print('\nNo open positions')

    # Check pending orders
    orders = mt5.orders_get()
    if orders:
        print(f'\nPending Orders ({len(orders)}):')
        for o in orders:
            print(f'  {o.symbol} | Magic: {o.magic}')
    else:
        print('No pending orders')

    # Check if BTCUSD is available
    sym = mt5.symbol_info("BTCUSD")
    if sym:
        print(f'\nBTCUSD Status:')
        print(f'  Visible: {sym.visible}')
        print(f'  Trade Mode: {sym.trade_mode}')
        print(f'  Volume Min: {sym.volume_min}')
    else:
        print('\nBTCUSD NOT FOUND!')

    mt5.shutdown()
