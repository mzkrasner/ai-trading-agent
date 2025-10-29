"""Entry-point script that wires together the trading agent, data feeds, and API."""

import sys
import argparse
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src.agent.decision_maker import TradingAgent
from src.indicators.hyperliquid_indicators import HyperliquidIndicators
from src.trading.hyperliquid_api import HyperliquidAPI
from src.sentiment.x_sentiment import XSentimentAnalyzer
import asyncio
import logging
from collections import deque, OrderedDict
from datetime import datetime, timezone
import math  # For Sharpe
from dotenv import load_dotenv
import os
import json
from aiohttp import web
from src.utils.formatting import format_number as fmt, format_size as fmt_sz
from src.utils.prompt_utils import json_default, round_or_none, round_series

load_dotenv()

# Force reconfigure logging (clears any existing configuration)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)


def clear_terminal():
    """Clear the terminal screen on Windows or POSIX systems."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_interval_seconds(interval_str):
    """Convert interval strings like '5m' or '1h' to seconds."""
    if interval_str.endswith('m'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 3600
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 86400
    else:
        raise ValueError(f"Unsupported interval: {interval_str}")

def main():
    """Parse CLI args, bootstrap dependencies, and launch the trading loop."""
    clear_terminal()
    parser = argparse.ArgumentParser(description="LLM-based Trading Agent on Hyperliquid")
    parser.add_argument("--assets", type=str, nargs="+", required=False, help="Assets to trade, e.g., BTC ETH")
    parser.add_argument("--interval", type=str, required=False, help="Interval period, e.g., 1h")
    args = parser.parse_args()

    # Allow assets/interval via .env (CONFIG) if CLI not provided
    from src.config_loader import CONFIG
    assets_env = CONFIG.get("assets")
    interval_env = CONFIG.get("interval")
    if (not args.assets or len(args.assets) == 0) and assets_env:
        # Support space or comma separated
        if "," in assets_env:
            args.assets = [a.strip() for a in assets_env.split(",") if a.strip()]
        else:
            args.assets = [a.strip() for a in assets_env.split(" ") if a.strip()]
    if not args.interval and interval_env:
        args.interval = interval_env

    if not args.assets or not args.interval:
        parser.error("Please provide --assets and --interval, or set ASSETS and INTERVAL in .env")

    hyperliquid = HyperliquidAPI()
    indicators = HyperliquidIndicators(hyperliquid)
    agent = TradingAgent()
    sentiment = XSentimentAnalyzer()


    start_time = datetime.now(timezone.utc)
    invocation_count = 0
    trade_log = []  # For Sharpe: list of returns
    active_trades = []  # {'asset','is_long','amount','entry_price','tp_oid','sl_oid','exit_plan'}
    recent_events = deque(maxlen=200)
    diary_path = "diary.jsonl"
    initial_account_value = None
    # Perp mid-price history sampled each loop (authoritative, avoids spot/perp basis mismatch)
    price_history = {}

    print(f"Starting trading agent for assets: {args.assets} at interval: {args.interval}")

    def add_event(msg: str):
        """Log an informational event and push it into the recent events deque."""
        logging.info(msg)

    async def run_loop():
        """Main trading loop that gathers data, calls the agent, and executes trades."""
        nonlocal invocation_count, initial_account_value
        while True:
            invocation_count += 1
            minutes_since_start = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

            # Global account state
            state = await hyperliquid.get_user_state()
            total_value = state.get('total_value') or state['balance'] + sum(p.get('pnl', 0) for p in state['positions'])
            sharpe = calculate_sharpe(trade_log)
            
            # Fetch user fee rates
            user_fees = await hyperliquid.get_user_fees()

            account_value = total_value
            if initial_account_value is None:
                initial_account_value = account_value
            total_return_pct = ((account_value - initial_account_value) / initial_account_value * 100.0) if initial_account_value else 0.0

            positions = []
            for pos_wrap in state['positions']:
                pos = pos_wrap
                coin = pos.get('coin')
                current_px = await hyperliquid.get_current_price(coin) if coin else None
                positions.append({
                    "symbol": coin,
                    "quantity": round_or_none(pos.get('szi'), 6),
                    "entry_price": round_or_none(pos.get('entryPx'), 2),
                    "current_price": round_or_none(current_px, 2),
                    "liquidation_price": round_or_none(pos.get('liquidationPx') or pos.get('liqPx'), 2),
                    "unrealized_pnl": round_or_none(pos.get('pnl'), 4),
                    "leverage": pos.get('leverage')
                })

            recent_diary = []
            try:
                with open(diary_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        entry = json.loads(line)
                        recent_diary.append(entry)
            except Exception:
                pass

            open_orders_struct = []
            try:
                open_orders = await hyperliquid.get_open_orders()
                for o in open_orders[:50]:
                    open_orders_struct.append({
                        "coin": o.get('coin'),
                        "oid": o.get('oid'),
                        "is_buy": o.get('isBuy'),
                        "size": round_or_none(o.get('sz'), 6),
                        "price": round_or_none(o.get('px'), 2),
                        "trigger_price": round_or_none(o.get('triggerPx'), 2),
                        "order_type": o.get('orderType')
                    })
            except Exception:
                open_orders = []

            # Reconcile active trades and clean up orphaned orders
            try:
                assets_with_positions = set()
                position_sizes = {}
                for pos in state['positions']:
                    try:
                        size = abs(float(pos.get('szi') or 0))
                        if size > 0:
                            coin = pos.get('coin')
                            assets_with_positions.add(coin)
                            position_sizes[coin] = size
                    except Exception:
                        continue
                
                # Clean up orphaned TP/SL orders (no matching position)
                tp_sl_orders_by_asset = {}
                for o in (open_orders or []):
                    order_type = o.get('orderType', '')
                    if isinstance(order_type, str) and ('Take Profit' in order_type or 'Stop Market' in order_type):
                        coin = o.get('coin')
                        if coin not in tp_sl_orders_by_asset:
                            tp_sl_orders_by_asset[coin] = []
                        tp_sl_orders_by_asset[coin].append(o)
                
                # Cancel orphaned TP/SL orders
                for coin, orders in tp_sl_orders_by_asset.items():
                    if coin not in assets_with_positions:
                        add_event(f"Found {len(orders)} orphaned TP/SL orders for {coin} (no position) - cancelling")
                        for order in orders:
                            try:
                                await hyperliquid.cancel_order(coin, order.get('oid'))
                                add_event(f"Cancelled orphaned {order.get('orderType')} for {coin}")
                            except Exception as e:
                                logging.error(f"Failed to cancel orphaned order {order.get('oid')}: {e}")
                    elif len(orders) > 2:
                        # If more than 2 TP/SL orders (should only have 1 TP + 1 SL), cancel duplicates
                        # Keep the most recent ones based on size matching position
                        position_size = position_sizes.get(coin, 0)
                        matching_orders = []
                        non_matching_orders = []
                        
                        for order in orders:
                            order_size = abs(float(order.get('sz', 0)))
                            if abs(order_size - position_size) < 0.001:  # Close enough match
                                matching_orders.append(order)
                            else:
                                non_matching_orders.append(order)
                        
                        # Cancel non-matching orders
                        for order in non_matching_orders:
                            try:
                                await hyperliquid.cancel_order(coin, order.get('oid'))
                                add_event(f"Cancelled mismatched {order.get('orderType')} for {coin} (size {order.get('sz')} vs position {position_size})")
                            except Exception as e:
                                logging.error(f"Failed to cancel mismatched order {order.get('oid')}: {e}")
                        
                        # If we have more than 2 matching orders, keep only 2 (1 TP, 1 SL)
                        if len(matching_orders) > 2:
                            tp_orders = [o for o in matching_orders if 'Take Profit' in o.get('orderType', '')]
                            sl_orders = [o for o in matching_orders if 'Stop' in o.get('orderType', '')]
                            
                            # Cancel extra TP orders (keep first)
                            for order in tp_orders[1:]:
                                try:
                                    await hyperliquid.cancel_order(coin, order.get('oid'))
                                    add_event(f"Cancelled duplicate TP for {coin}")
                                except Exception as e:
                                    logging.error(f"Failed to cancel duplicate TP {order.get('oid')}: {e}")
                            
                            # Cancel extra SL orders (keep first)
                            for order in sl_orders[1:]:
                                try:
                                    await hyperliquid.cancel_order(coin, order.get('oid'))
                                    add_event(f"Cancelled duplicate SL for {coin}")
                                except Exception as e:
                                    logging.error(f"Failed to cancel duplicate SL {order.get('oid')}: {e}")
                
                # Reconcile active_trades in-memory list
                assets_with_orders = {o.get('coin') for o in (open_orders or []) if o.get('coin')}
                for tr in active_trades[:]:
                    asset = tr.get('asset')
                    if asset not in assets_with_positions and asset not in assets_with_orders:
                        add_event(f"Reconciling stale active trade for {asset} (no position, no orders)")
                        active_trades.remove(tr)
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": "reconcile_close",
                                "reason": "no_position_no_orders",
                                "opened_at": tr.get('opened_at')
                            }) + "\n")
            except Exception as e:
                logging.error(f"Reconciliation error: {e}")
                pass

            recent_fills_struct = []
            try:
                fills = await hyperliquid.get_recent_fills(limit=50)
                for f_entry in fills[-20:]:
                    try:
                        t_raw = f_entry.get('time') or f_entry.get('timestamp')
                        timestamp = None
                        if t_raw is not None:
                            try:
                                t_int = int(t_raw)
                                if t_int > 1e12:
                                    timestamp = datetime.fromtimestamp(t_int / 1000, tz=timezone.utc).isoformat()
                                else:
                                    timestamp = datetime.fromtimestamp(t_int, tz=timezone.utc).isoformat()
                            except Exception:
                                timestamp = str(t_raw)
                        recent_fills_struct.append({
                            "timestamp": timestamp,
                            "coin": f_entry.get('coin') or f_entry.get('asset'),
                            "is_buy": f_entry.get('isBuy'),
                            "size": round_or_none(f_entry.get('sz') or f_entry.get('size'), 6),
                            "price": round_or_none(f_entry.get('px') or f_entry.get('price'), 2)
                        })
                    except Exception:
                        continue
            except Exception:
                pass

            dashboard = {
                "total_return_pct": round(total_return_pct, 2),
                "balance": round_or_none(state['balance'], 2),
                "account_value": round_or_none(account_value, 2),
                "sharpe_ratio": round_or_none(sharpe, 3),
                "positions": positions,
                "active_trades": [
                    {
                        "asset": tr.get('asset'),
                        "is_long": tr.get('is_long'),
                        "amount": round_or_none(tr.get('amount'), 6),
                        "entry_price": round_or_none(tr.get('entry_price'), 2),
                        "tp_oid": tr.get('tp_oid'),
                        "sl_oid": tr.get('sl_oid'),
                        "exit_plan": tr.get('exit_plan'),
                        "opened_at": tr.get('opened_at')
                    }
                    for tr in active_trades
                ],
                "open_orders": open_orders_struct,
                "recent_diary": recent_diary,
                "recent_fills": recent_fills_struct,
                "fee_structure": user_fees,  # Pure data: user's current fee rates
            }

            # Gather data for ALL assets first
            market_sections = []
            asset_prices = {}
            for asset in args.assets:
                try:
                    current_price = await hyperliquid.get_current_price(asset)
                    asset_prices[asset] = current_price
                    if asset not in price_history:
                        price_history[asset] = deque(maxlen=60)
                    price_history[asset].append({"t": datetime.now(timezone.utc).isoformat(), "mid": round_or_none(current_price, 2)})
                    oi = await hyperliquid.get_open_interest(asset)
                    funding = await hyperliquid.get_funding_rate(asset)

                    # Fetch 5m indicators from Hyperliquid
                    intraday_tf = "5m"
                    intraday_candles = await indicators.get_candles(asset, intraday_tf, num_candles=100)
                    # Pass candles to avoid duplicate API call
                    intraday_data = await indicators.get_indicators_from_candles(asset, intraday_tf, intraday_candles)
                    
                    # Get series for intraday indicators
                    ema_series = indicators.get_series(intraday_candles, "ema", 20)
                    macd_series = indicators.get_series(intraday_candles, "macd")
                    rsi7_series = indicators.get_series(intraday_candles, "rsi", 7)
                    rsi14_series = indicators.get_series(intraday_candles, "rsi", 14)
                    
                    # Calculate market metrics for 5m timeframe
                    intraday_market_metrics = indicators.calculate_market_metrics(intraday_candles, intraday_data)
                    
                    # Fetch 4h indicators from Hyperliquid
                    lt_candles = await indicators.get_candles(asset, "4h", num_candles=100)
                    # Pass candles to avoid duplicate API call
                    lt_data = await indicators.get_indicators_from_candles(asset, "4h", lt_candles)
                    
                    # Calculate market metrics for 4h timeframe
                    lt_market_metrics = indicators.calculate_market_metrics(lt_candles, lt_data)
                    
                    # Extract long-term values
                    lt_ema20 = lt_data.get("ema20")
                    lt_ema50 = lt_data.get("ema50")
                    lt_atr3 = None  # Calculate if needed
                    lt_atr14 = lt_data.get("atr14")
                    lt_macd_series = indicators.get_series(lt_candles, "macd")
                    lt_rsi_series = indicators.get_series(lt_candles, "rsi", 14)

                    recent_mids = [entry["mid"] for entry in list(price_history.get(asset, []))[-10:]]
                    funding_annualized = round(funding * 24 * 365 * 100, 2) if funding else None

                    market_sections.append({
                        "asset": asset,
                        "current_price": round_or_none(current_price, 2),
                        "intraday": {
                            "ema20": round_or_none(ema_series[-1], 2) if ema_series else None,
                            "macd": round_or_none(macd_series[-1], 2) if macd_series else None,
                            "rsi7": round_or_none(rsi7_series[-1], 2) if rsi7_series else None,
                            "rsi14": round_or_none(rsi14_series[-1], 2) if rsi14_series else None,
                            "series": {
                                "ema20": round_series(ema_series, 2),
                                "macd": round_series(macd_series, 2),
                                "rsi7": round_series(rsi7_series, 2),
                                "rsi14": round_series(rsi14_series, 2)
                            },
                            "market_metrics": intraday_market_metrics  # NEW: 5m market metrics
                        },
                        "long_term": {
                            "ema20": round_or_none(lt_ema20, 2),
                            "ema50": round_or_none(lt_ema50, 2),
                            "atr3": round_or_none(lt_atr3, 2),
                            "atr14": round_or_none(lt_atr14, 2),
                            "macd_series": round_series(lt_macd_series, 2),
                            "rsi_series": round_series(lt_rsi_series, 2),
                            "market_metrics": lt_market_metrics  # NEW: 4h market metrics
                        },
                        "open_interest": round_or_none(oi, 2),
                        "funding_rate": round_or_none(funding, 8),
                        "funding_annualized_pct": funding_annualized,
                        "recent_mid_prices": recent_mids
                    })
                except Exception as e:
                    add_event(f"Data gather error {asset}: {e}")
                    continue

            # Fetch X/Twitter sentiment observations (non-prescriptive)
            x_observations = {}
            market_context = {}
            try:
                # Fetch both asset-specific sentiment and general market context in parallel
                x_observations, market_context = await asyncio.gather(
                    sentiment.get_sentiment_data(args.assets),
                    sentiment.get_market_context(),
                    return_exceptions=True
                )
                # Handle exceptions from gather
                if isinstance(x_observations, Exception):
                    logging.debug(f"X sentiment error: {x_observations}")
                    x_observations = {}
                if isinstance(market_context, Exception):
                    logging.debug(f"Market context error: {market_context}")
                    market_context = {}
                    
                if x_observations or market_context:
                    logging.info(f"Fetched X sentiment and market context for analysis")
            except Exception as e:
                logging.debug(f"X sentiment unavailable (non-critical): {e}")
                # Continue without sentiment - it's supplementary data
            
            # Single LLM call with all assets
            context_payload = OrderedDict([
                ("invocation", {
                    "minutes_since_start": round(minutes_since_start, 2),
                    "current_time": datetime.now(timezone.utc).isoformat(),
                    "invocation_count": invocation_count
                }),
                ("account", dashboard),
                ("market_data", market_sections),
                ("x_observations", x_observations) if x_observations else ("x_observations", {"note": "No X data available"}),
                ("market_context", market_context) if market_context else ("market_context", {"note": "No market context available"}),
                ("instructions", {
                    "assets": args.assets,
                    "requirement": "Decide actions for all assets and return a strict JSON array matching the schema."
                })
            ])
            context = json.dumps(context_payload, default=json_default)
            add_event(f"Combined prompt length: {len(context)} chars for {len(args.assets)} assets")
            with open("prompts.log", "a") as f:
                f.write(f"\n\n--- {datetime.now()} - ALL ASSETS ---\n{json.dumps(context_payload, indent=2, default=json_default)}\n")

            def _is_failed_outputs(outs):
                """Return True when outputs are missing or clearly invalid."""
                if not isinstance(outs, dict):
                    return True
                decisions = outs.get("trade_decisions")
                if not isinstance(decisions, list) or not decisions:
                    return True
                try:
                    return all(
                        isinstance(o, dict)
                        and (o.get('action') == 'hold')
                        and ('parse error' in (o.get('rationale', '').lower()))
                        for o in decisions
                    )
                except Exception:
                    return True

            try:
                outputs = agent.decide_trade(args.assets, context)
                if not isinstance(outputs, dict):
                    add_event(f"Invalid output format (expected dict): {outputs}")
                    outputs = {}
            except Exception as e:
                import traceback
                add_event(f"Agent error: {e}")
                add_event(f"Traceback: {traceback.format_exc()}")
                outputs = {}

            # Retry once on failure/parse error with a stricter instruction prefix
            if _is_failed_outputs(outputs):
                add_event("Retrying LLM once due to invalid/parse-error output")
                context_retry_payload = OrderedDict([
                    ("retry_instruction", "Return ONLY the JSON array per schema with no prose."),
                    ("original_context", context_payload)
                ])
                context_retry = json.dumps(context_retry_payload, default=json_default)
                try:
                    outputs = agent.decide_trade(args.assets, context_retry)
                    if not isinstance(outputs, dict):
                        add_event(f"Retry invalid format: {outputs}")
                        outputs = {}
                except Exception as e:
                    import traceback
                    add_event(f"Retry agent error: {e}")
                    add_event(f"Retry traceback: {traceback.format_exc()}")
                    outputs = {}

            reasoning_text = outputs.get("reasoning", "") if isinstance(outputs, dict) else ""
            if reasoning_text:
                add_event(f"LLM reasoning summary: {reasoning_text}")

            # Execute trades for each asset
            for output in outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []:
                try:
                    asset = output.get("asset")
                    if not asset or asset not in args.assets:
                        continue
                    action = output.get("action")
                    current_price = asset_prices.get(asset, 0)
                    action = output["action"]
                    rationale = output.get("rationale", "")
                    if rationale:
                        add_event(f"Decision rationale for {asset}: {rationale}")
                    if action in ("buy", "sell"):
                        is_buy = action == "buy"
                        alloc_usd = float(output.get("allocation_usd", 0.0))
                        if alloc_usd <= 0:
                            add_event(f"Holding {asset}: zero/negative allocation")
                            continue
                        if alloc_usd < 10:
                            add_event(f"WARNING: {asset} allocation ${alloc_usd:.2f} below minimum $10 - skipping trade")
                            continue
                        amount = alloc_usd / current_price

                        order = await hyperliquid.place_buy_order(asset, amount) if is_buy else await hyperliquid.place_sell_order(asset, amount)
                        # Check if the order filled by examining the order response directly
                        filled = hyperliquid.check_order_filled(order)
                        trade_log.append({"type": action, "price": current_price, "amount": amount, "exit_plan": output["exit_plan"], "filled": filled})
                        tp_oid = None
                        sl_oid = None
                        
                        # Only place TP/SL if the main order filled successfully
                        if filled:
                            # PRE-TRADE CLEANUP: Cancel all existing TP/SL orders for this asset before placing new ones
                            try:
                                existing_tp_sl = [o for o in open_orders if o.get('coin') == asset and 
                                                 isinstance(o.get('orderType', ''), str) and 
                                                 ('Take Profit' in o.get('orderType', '') or 'Stop Market' in o.get('orderType', ''))]
                                if existing_tp_sl:
                                    add_event(f"Cancelling {len(existing_tp_sl)} existing TP/SL orders for {asset} before placing new ones")
                                    for old_order in existing_tp_sl:
                                        try:
                                            await hyperliquid.cancel_order(asset, old_order.get('oid'))
                                            add_event(f"Cancelled old {old_order.get('orderType')} for {asset}")
                                        except Exception as e:
                                            logging.error(f"Failed to cancel old order {old_order.get('oid')}: {e}")
                            except Exception as e:
                                logging.error(f"Pre-trade cleanup error for {asset}: {e}")
                            
                            # Now place new TP/SL orders
                            if output["tp_price"]:
                                tp_order = await hyperliquid.place_take_profit(asset, is_buy, amount, output["tp_price"])
                                tp_oids = hyperliquid.extract_oids(tp_order)
                                tp_oid = tp_oids[0] if tp_oids else None
                                add_event(f"TP placed {asset} at {output['tp_price']}")
                            if output["sl_price"]:
                                sl_order = await hyperliquid.place_stop_loss(asset, is_buy, amount, output["sl_price"])
                                sl_oids = hyperliquid.extract_oids(sl_order)
                                sl_oid = sl_oids[0] if sl_oids else None
                                add_event(f"SL placed {asset} at {output['sl_price']}")
                        else:
                            add_event(f"WARNING: {asset} main order did not fill - skipping TP/SL orders")
                        # Only add to active_trades if the order filled
                        if filled:
                            # Reconcile: if opposite-side position exists or TP/SL just filled, clear stale active_trades for this asset
                            for existing in active_trades[:]:
                                if existing.get('asset') == asset:
                                    try:
                                        active_trades.remove(existing)
                                    except ValueError:
                                        pass
                            active_trades.append({
                                "asset": asset,
                                "is_long": is_buy,
                                "amount": amount,
                                "entry_price": current_price,
                                "tp_oid": tp_oid,
                                "sl_oid": sl_oid,
                                "exit_plan": output["exit_plan"],
                                "opened_at": datetime.now().isoformat()
                            })
                        add_event(f"{action.upper()} {asset} amount {amount:.4f} at ~{current_price}")
                        if rationale:
                            add_event(f"Post-trade rationale for {asset}: {rationale}")
                        # Write to diary after confirming fills status
                        with open(diary_path, "a") as f:
                            diary_entry = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": action,
                                "allocation_usd": alloc_usd,
                                "amount": amount,
                                "entry_price": current_price,
                                "tp_price": output.get("tp_price"),
                                "tp_oid": tp_oid,
                                "sl_price": output.get("sl_price"),
                                "sl_oid": sl_oid,
                                "exit_plan": output.get("exit_plan", ""),
                                "rationale": output.get("rationale", ""),
                                "order_result": str(order),
                                "opened_at": datetime.now(timezone.utc).isoformat(),
                                "filled": filled
                            }
                            f.write(json.dumps(diary_entry) + "\n")
                    elif action == "close":
                        # CLOSE action - exit the entire position at market
                        add_event(f"Close {asset}: {output.get('rationale', '')}")
                        
                        # Check if there's an existing position to close
                        position = None
                        for pos in state['positions']:
                            if pos.get('coin') == asset and abs(float(pos.get('szi', 0))) > 0:
                                position = pos
                                break
                        
                        if position:
                            try:
                                # Close the position at market
                                close_result = await hyperliquid.close_position(asset)
                                filled = hyperliquid.check_order_filled(close_result)
                                
                                if filled:
                                    add_event(f"Successfully closed {asset} position")
                                    
                                    # Cancel any existing TP/SL orders for this asset
                                    try:
                                        existing_tp_sl = [o for o in open_orders if o.get('coin') == asset and 
                                                         isinstance(o.get('orderType', ''), str) and 
                                                         ('Take Profit' in o.get('orderType', '') or 'Stop Market' in o.get('orderType', ''))]
                                        for old_order in existing_tp_sl:
                                            try:
                                                await hyperliquid.cancel_order(asset, old_order.get('oid'))
                                                add_event(f"Cancelled {old_order.get('orderType')} for closed {asset} position")
                                            except Exception as e:
                                                logging.error(f"Failed to cancel order {old_order.get('oid')}: {e}")
                                    except Exception as e:
                                        logging.error(f"Error cancelling TP/SL after close for {asset}: {e}")
                                    
                                    # Remove from active_trades
                                    for tr in active_trades[:]:
                                        if tr.get('asset') == asset:
                                            try:
                                                active_trades.remove(tr)
                                            except ValueError:
                                                pass
                                    
                                    # Write to diary
                                    with open(diary_path, "a") as f:
                                        diary_entry = {
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                            "asset": asset,
                                            "action": "close",
                                            "rationale": output.get("rationale", ""),
                                            "close_result": str(close_result),
                                            "filled": filled
                                        }
                                        f.write(json.dumps(diary_entry) + "\n")
                                else:
                                    add_event(f"WARNING: {asset} close order did not fill")
                            except Exception as e:
                                logging.error(f"Failed to close {asset} position: {e}")
                                add_event(f"ERROR: Failed to close {asset}: {e}")
                        else:
                            add_event(f"No {asset} position to close")
                            # Still write to diary
                            with open(diary_path, "a") as f:
                                diary_entry = {
                                    "timestamp": datetime.now().isoformat(),
                                    "asset": asset,
                                    "action": "close",
                                    "rationale": output.get("rationale", ""),
                                    "result": "no_position_to_close"
                                }
                                f.write(json.dumps(diary_entry) + "\n")
                    else:
                        # HOLD action - but check if LLM wants to update TP/SL orders
                        add_event(f"Hold {asset}: {output.get('rationale', '')}")
                        
                        # Check if LLM provided new TP/SL prices for existing position
                        new_tp = output.get("tp_price")
                        new_sl = output.get("sl_price")
                        
                        if new_tp or new_sl:
                            # Find existing position for this asset
                            position = None
                            for pos in state['positions']:
                                if pos.get('coin') == asset and abs(float(pos.get('szi', 0))) > 0:
                                    position = pos
                                    break
                            
                            if position:
                                # We have a position - update TP/SL orders
                                position_size = abs(float(position.get('szi', 0)))
                                is_long = float(position.get('szi', 0)) > 0
                                
                                # Cancel existing TP/SL orders for this asset
                                try:
                                    existing_tp_sl = [o for o in open_orders if o.get('coin') == asset and 
                                                     isinstance(o.get('orderType', ''), str) and 
                                                     ('Take Profit' in o.get('orderType', '') or 'Stop Market' in o.get('orderType', ''))]
                                    if existing_tp_sl:
                                        add_event(f"Updating TP/SL for {asset} - cancelling {len(existing_tp_sl)} existing orders")
                                        for old_order in existing_tp_sl:
                                            try:
                                                await hyperliquid.cancel_order(asset, old_order.get('oid'))
                                            except Exception as e:
                                                logging.error(f"Failed to cancel old order {old_order.get('oid')}: {e}")
                                except Exception as e:
                                    logging.error(f"Error cancelling old TP/SL for {asset}: {e}")
                                
                                # Place new TP/SL orders
                                tp_oid = None
                                sl_oid = None
                                if new_tp:
                                    try:
                                        tp_order = await hyperliquid.place_take_profit(asset, is_long, position_size, new_tp)
                                        tp_oids = hyperliquid.extract_oids(tp_order)
                                        tp_oid = tp_oids[0] if tp_oids else None
                                        add_event(f"Updated TP for {asset} at {new_tp}")
                                    except Exception as e:
                                        logging.error(f"Failed to place new TP for {asset}: {e}")
                                
                                if new_sl:
                                    try:
                                        sl_order = await hyperliquid.place_stop_loss(asset, is_long, position_size, new_sl)
                                        sl_oids = hyperliquid.extract_oids(sl_order)
                                        sl_oid = sl_oids[0] if sl_oids else None
                                        add_event(f"Updated SL for {asset} at {new_sl}")
                                    except Exception as e:
                                        logging.error(f"Failed to place new SL for {asset}: {e}")
                                
                                # Update active_trades with new TP/SL oids
                                for tr in active_trades:
                                    if tr.get('asset') == asset:
                                        if tp_oid:
                                            tr['tp_oid'] = tp_oid
                                        if sl_oid:
                                            tr['sl_oid'] = sl_oid
                                        break
                        
                        # Write hold to diary
                        with open(diary_path, "a") as f:
                            diary_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "asset": asset,
                                "action": "hold",
                                "rationale": output.get("rationale", ""),
                                "tp_price": output.get("tp_price"),
                                "sl_price": output.get("sl_price"),
                                "updated_orders": bool(new_tp or new_sl)
                            }
                            f.write(json.dumps(diary_entry) + "\n")
                except Exception as e:
                    import traceback
                    add_event(f"Execution error {asset}: {e}")

            # Show countdown timer between runs
            interval_seconds = get_interval_seconds(args.interval)
            print(f"\n{'='*60}")
            print(f"Next run in {args.interval}. Countdown starting...")
            print(f"{'='*60}")
            
            # Countdown with progress bar
            for remaining in range(interval_seconds, 0, -1):
                # Calculate progress
                progress = (interval_seconds - remaining) / interval_seconds
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                # Format time remaining
                if remaining >= 3600:
                    hours = remaining // 3600
                    minutes = (remaining % 3600) // 60
                    seconds = remaining % 60
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                elif remaining >= 60:
                    minutes = remaining // 60
                    seconds = remaining % 60
                    time_str = f"{minutes:02d}:{seconds:02d}"
                else:
                    time_str = f"00:{remaining:02d}"
                
                # Print countdown with carriage return to update same line
                print(f"\r⏱️  [{bar}] {time_str} remaining", end="", flush=True)
                await asyncio.sleep(1)
            
            # Clear the countdown line and show starting message
            print(f"\r{'  ' * 30}", end="")  # Clear the line
            print(f"\r🚀 Starting next trading cycle...\n")

    async def handle_diary(request):
        """Return diary entries as JSON or newline-delimited text."""
        try:
            raw = request.query.get('raw')
            download = request.query.get('download')
            if raw or download:
                if not os.path.exists(diary_path):
                    return web.Response(text="", content_type="text/plain")
                with open(diary_path, "r") as f:
                    data = f.read()
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename=diary.jsonl"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(request.query.get('limit', '200'))
            with open(diary_path, "r") as f:
                lines = f.readlines()
            start = max(0, len(lines) - limit)
            entries = [json.loads(l) for l in lines[start:]]
            return web.json_response({"entries": entries})
        except FileNotFoundError:
            return web.json_response({"entries": []})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_logs(request):
        """Stream log files with optional download or tailing behaviour."""
        try:
            path = request.query.get('path', 'llm_requests.log')
            download = request.query.get('download')
            limit_param = request.query.get('limit')
            if not os.path.exists(path):
                return web.Response(text="", content_type="text/plain")
            with open(path, "r") as f:
                data = f.read()
            if download or (limit_param and (limit_param.lower() == 'all' or limit_param == '-1')):
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename={os.path.basename(path)}"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(limit_param) if limit_param else 2000
            return web.Response(text=data[-limit:], content_type="text/plain")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def start_api(app):
        """Register HTTP endpoints for observing diary entries and logs."""
        app.router.add_get('/diary', handle_diary)
        app.router.add_get('/logs', handle_logs)

    async def main_async():
        """Start the aiohttp server and kick off the trading loop."""
        app = web.Application()
        await start_api(app)
        from src.config_loader import CONFIG as CFG
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, CFG.get("api_host"), int(CFG.get("api_port")))
        await site.start()
        await run_loop()

    def calculate_total_return(state, trade_log):
        """Compute percent return relative to an assumed initial balance."""
        initial = 10000
        current = state['balance'] + sum(p.get('pnl', 0) for p in state.get('positions', []))
        return ((current - initial) / initial) * 100 if initial else 0

    def calculate_sharpe(returns):
        """Compute a naive Sharpe-like ratio from the trade log."""
        if not returns:
            return 0
        vals = [r.get('pnl', 0) if 'pnl' in r else 0 for r in returns]
        if not vals:
            return 0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 0
        return mean / std if std > 0 else 0

    async def check_exit_condition(trade, indicators, hyperliquid):
        """Evaluate whether a given trade's exit plan triggers a close."""
        plan = (trade.get("exit_plan") or "").lower()
        if not plan:
            return False
        try:
            if "macd" in plan and "below" in plan:
                data = await indicators.get_indicators(trade["asset"], "4h")
                macd = data.get("macd", 0)
                threshold = float(plan.split("below")[-1].strip())
                return macd < threshold
            if "close above ema50" in plan:
                data = await indicators.get_indicators(trade["asset"], "4h")
                ema50 = data.get("ema50", 0)
                current = await hyperliquid.get_current_price(trade["asset"])
                return current > ema50
        except Exception:
            return False
        return False

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
