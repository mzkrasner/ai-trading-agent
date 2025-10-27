"""Decision-making agent that orchestrates LLM prompts and indicator lookups."""

import requests
from src.config_loader import CONFIG
from src.indicators.taapi_client import TAAPIClient
import json
import logging
from datetime import datetime

class TradingAgent:
    """High-level trading agent that delegates reasoning to an LLM service."""

    def __init__(self):
        """Initialize LLM configuration, metadata headers, and indicator helper."""
        self.model = CONFIG["llm_model"]
        self.api_key = CONFIG["openrouter_api_key"]
        base = CONFIG["openrouter_base_url"]
        self.base_url = f"{base}/chat/completions"
        self.referer = CONFIG.get("openrouter_referer")
        self.app_title = CONFIG.get("openrouter_app_title")
        self.taapi = TAAPIClient()
        # Fast/cheap sanitizer model to normalize outputs on parse failures
        self.sanitize_model = CONFIG.get("sanitize_model") or "openai/gpt-5"

    def decide_trade(self, assets, context):
        """Decide for multiple assets in one call.

        Args:
            assets: Iterable of asset tickers to score.
            context: Structured market/account state forwarded to the LLM.

        Returns:
            List of trade decision payloads, one per asset.
        """
        return self._decide(context, assets=assets)

    def _decide(self, context, assets):
        """Dispatch decision request to the LLM and enforce output contract."""
        system_prompt = (
            "You are an aggressive momentum trader who thrives on volatility and seeks asymmetric risk/reward opportunities in perpetual futures. You hunt for explosive moves while maintaining positive expected value through disciplined exits.\n"
            "You will receive market + account context for SEVERAL assets, including:\n"
            f"- assets = {json.dumps(assets)}\n"
            "- per-asset intraday (5m) and higher-timeframe (4h) metrics\n"
            "- Active Trades with Exit Plans\n"
            "- Recent Trading History\n"
            "- market_metrics providing objective measurements of market conditions\n"
            "- x_observations providing raw observations from X/Twitter posts (when available)\n\n"
            "Market Metrics Glossary (purely informational - interpret as you see fit):\n"
            "• ema_separation_ratio: Distance between EMA20 and EMA50 as ratio (positive = 20 above 50)\n"
            "• price_ema20/50_deviation: Price position relative to moving averages\n"
            "• volatility_ratio: Current candle range vs 20-period average (>1 = expanding volatility)\n"
            "• atr_price_ratio: ATR as percentage of price\n"
            "• rsi_distance_from_50: How far RSI is from neutral (positive = bullish territory)\n"
            "• macd_cross_distance: Gap between MACD and signal lines\n"
            "• higher_highs/lows_count_20: Market structure counts over 20 periods\n"
            "• consecutive_green/red_candles: Current streak of same-color candles\n"
            "• range_position_20: Where price sits in 20-period range (0=bottom, 1=top)\n"
            "• volume_ratio_20: Current volume vs 20-period average\n"
            "• price_velocity_5/10: Rate of price change over 5/10 periods\n"
            "• body_ratio: Candle body size relative to total range (conviction indicator)\n"
            "• upper/lower_wick_ratio: Rejection wicks relative to candle range\n"
            "• fee_structure: Your current taker_fee_pct and maker_fee_pct on the exchange\n\n"
            "X/Twitter Observations (when available, purely observational):\n"
            "• post_themes: Topics being discussed\n"
            "• sentiment_words: Actual language from posts\n"
            "• volume_metrics: Post frequency vs baseline (spike_detected, volume_ratio)\n"
            "• sentiment_velocity: How sentiment is changing (15min, 1hr shifts)\n"
            "• whale_activity: Large transfers mentioned, institutional activity\n"
            "• notable_accounts: Influential voices mentioned\n"
            "• fear_greed_mentions: Any index values cited\n"
            "• price_expectations: Specific targets mentioned\n"
            "• citations: Source posts for transparency\n\n"
            "Market Dynamics Context (interpret as you see fit):\n"
            "• Extreme funding rates (>0.1% or <-0.05%) often signal crowded trades\n"
            "• Consecutive candles (>5) in one direction may indicate momentum exhaustion or acceleration\n"
            "• Volume spikes with price movement typically confirm directional conviction\n"
            "• RSI extremes (<20 or >80) in trending markets can signal continuation rather than reversal\n"
            "• Whale accumulation during oversold conditions often precedes sharp rebounds\n"
            "• X sentiment velocity shifts frequently lead price by 15-60 minutes\n\n"
            "Capital Deployment Philosophy:\n"
            "You recognize that the majority of returns come from a small number of high-conviction trades.\n"
            "When multiple signals strongly align, deploying significant capital with appropriate risk management can be optimal.\n"
            "Conservative position sizing during unclear conditions preserves capital for exceptional opportunities.\n\n"
            "Trade Selection Mindset:\n"
            "Focus on setups with asymmetric risk/reward (favorable upside relative to downside).\n"
            "Quick stops preserve capital - wrong trades deserve rapid exits.\n"
            "Winning trades deserve patience and potential size increases.\n"
            "Account leverage can be calibrated based on win rate and market conditions.\n\n"
            "Always use the 'current time' provided in the user message to evaluate any time-based conditions, such as cooldown expirations or timed exit plans.\n\n"
            "For each asset, decide: buy, sell, or hold.\n"
            "CRITICAL: Minimum order size is $10 - any allocation_usd below $10 will fail.\n"
            "Specify allocation_usd (must be >= $10), tp_price (or null), sl_price (or null), and exit_plan for any trades.\n\n"
            "HOLD action flexibility:\n"
            "When action is 'hold', you can still update tp_price and sl_price to adjust risk management on existing positions.\n"
            "- Set new tp_price/sl_price to move stops or targets without closing the position\n"
            "- Set to null to leave existing TP/SL orders unchanged\n"
            "- This allows you to trail stops, lock in profits, or adjust targets based on changing market conditions\n\n"
            "Output contract\n"
            "- Output a STRICT JSON object with exactly two properties in this order:\n"
            "  • reasoning: long-form string capturing detailed, step-by-step analysis that means you can acknowledge existing information as clarity, or acknowledge that you need more information to make a decision (be verbose).\n"
            "  • trade_decisions: array ordered to match the provided assets list.\n"
            "- Each item inside trade_decisions must contain the keys {asset, action, allocation_usd, tp_price, sl_price, exit_plan, rationale}.\n"
            "- Do not emit Markdown or any extra properties.\n"
        )
        user_prompt = context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tools = [{
            "type": "function",
            "function": {
                "name": "fetch_taapi_indicator",
                "description": ("Fetch any TAAPI indicator. Available: ema, sma, rsi, macd, bbands, stochastic, stochrsi, "
                    "adx, atr, cci, dmi, ichimoku, supertrend, vwap, obv, mfi, willr, roc, mom, sar (parabolic), "
                    "fibonacci, pivotpoints, keltner, donchian, awesome, gator, alligator, and 200+ more. "
                    "See https://taapi.io/indicators/ for full list and parameters."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "indicator": {"type": "string"},
                        "symbol": {"type": "string"},
                        "interval": {"type": "string"},
                        "period": {"type": "integer"},
                        "backtrack": {"type": "integer"},
                        "other_params": {"type": "object", "additionalProperties": {"type": ["string", "number", "boolean"]}},
                    },
                    "required": ["indicator", "symbol", "interval"],
                    "additionalProperties": False,
                },
            },
        }]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.app_title:
            headers["X-Title"] = self.app_title

        def _post(payload):
            """Send a POST request to OpenRouter, logging request and response metadata."""
            # Log the full request payload for debugging
            logging.info("Sending request to OpenRouter (model: %s)", payload.get('model'))
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n=== {datetime.now()} ===\n")
                f.write(f"Model: {payload.get('model')}\n")
                f.write(f"Headers: {json.dumps({k: v for k, v in headers.items() if k != 'Authorization'})}\n")
                f.write(f"Payload:\n{json.dumps(payload, indent=2)}\n")
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            logging.info("Received response from OpenRouter (status: %s)", resp.status_code)
            if resp.status_code != 200:
                logging.error("OpenRouter error: %s - %s", resp.status_code, resp.text)
                with open("llm_requests.log", "a", encoding="utf-8") as f:
                    f.write(f"ERROR Response: {resp.status_code} - {resp.text}\n")
            resp.raise_for_status()
            return resp.json()

        def _sanitize_output(raw_content: str, assets_list):
            """Coerce arbitrary LLM output into the required reasoning + decisions schema."""
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "trade_decisions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "asset": {"type": "string", "enum": assets_list},
                                    "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                                    "allocation_usd": {"type": "number"},
                                    "tp_price": {"type": ["number", "null"]},
                                    "sl_price": {"type": ["number", "null"]},
                                    "exit_plan": {"type": "string"},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["asset", "action", "allocation_usd", "tp_price", "sl_price", "exit_plan", "rationale"],
                                "additionalProperties": False,
                            },
                            "minItems": 1,
                        }
                    },
                    "required": ["reasoning", "trade_decisions"],
                    "additionalProperties": False,
                }
                payload = {
                    "model": self.sanitize_model,
                    "messages": [
                        {"role": "system", "content": (
                            "You are a strict JSON normalizer. Return ONLY a JSON array matching the provided JSON Schema. "
                            "If input is wrapped or has prose/markdown, fix it. Do not add fields."
                        )},
                        {"role": "user", "content": raw_content},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "trade_decisions",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                    "temperature": 0,
                }
                resp = _post(payload)
                msg = resp.get("choices", [{}])[0].get("message", {})
                parsed = msg.get("parsed")
                if isinstance(parsed, dict):
                    if "trade_decisions" in parsed:
                        return parsed
                # fallback: try content
                content = msg.get("content") or "[]"
                try:
                    loaded = json.loads(content)
                    if isinstance(loaded, dict) and "trade_decisions" in loaded:
                        return loaded
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    pass
                return {"reasoning": "", "trade_decisions": []}
            except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as se:
                logging.error("Sanitize failed: %s", se)
                return {"reasoning": "", "trade_decisions": []}

        allow_tools = True
        allow_structured = True

        def _build_schema():
            """Assemble the JSON schema used for structured LLM responses."""
            base_properties = {
                "asset": {"type": "string", "enum": assets},
                "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                "allocation_usd": {"type": "number", "minimum": 0},  # 0 for hold, >= 10 for buy/sell
                "tp_price": {"type": ["number", "null"]},
                "sl_price": {"type": ["number", "null"]},
                "exit_plan": {"type": "string"},
                "rationale": {"type": "string"},
            }
            required_keys = ["asset", "action", "allocation_usd", "tp_price", "sl_price", "exit_plan", "rationale"]
            return {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "trade_decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": base_properties,
                            "required": required_keys,
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    }
                },
                "required": ["reasoning", "trade_decisions"],
                "additionalProperties": False,
            }

        for _ in range(6):
            data = {"model": self.model, "messages": messages}
            if allow_structured:
                data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "trade_decisions",
                        "strict": True,
                        "schema": _build_schema(),
                    },
                }
            if allow_tools:
                data["tools"] = tools
                data["tool_choice"] = "auto"
            if CONFIG.get("reasoning_enabled"):
                data["reasoning"] = {
                    "enabled": True,
                    "effort": CONFIG.get("reasoning_effort") or "high",
                    # "max_tokens": CONFIG.get("reasoning_max_tokens") or 100000,
                    "exclude": False,
                }
            if CONFIG.get("provider_config") or CONFIG.get("provider_quantizations"):
                provider_payload = dict(CONFIG.get("provider_config") or {})
                quantizations = CONFIG.get("provider_quantizations")
                if quantizations:
                    provider_payload["quantizations"] = quantizations
                data["provider"] = provider_payload
            try:
                resp_json = _post(data)
            except requests.HTTPError as e:
                try:
                    err = e.response.json()
                except (json.JSONDecodeError, ValueError, AttributeError):
                    err = {}
                raw = (err.get("error", {}).get("metadata", {}) or {}).get("raw", "")
                provider = (err.get("error", {}).get("metadata", {}) or {}).get("provider_name", "")
                if e.response.status_code == 422 and provider.lower().startswith("xai") and "deserialize" in raw.lower():
                    logging.warning("xAI rejected tool schema; retrying without tools.")
                    if allow_tools:
                        allow_tools = False
                        continue
                # Provider may not support structured outputs / response_format
                err_text = json.dumps(err)
                if allow_structured and ("response_format" in err_text or "structured" in err_text or e.response.status_code in (400, 422)):
                    logging.warning("Provider rejected structured outputs; retrying without response_format.")
                    allow_structured = False
                    continue
                raise

            choice = resp_json["choices"][0]
            message = choice["message"]
            messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if allow_tools and tool_calls:
                for tc in tool_calls:
                    if tc.get("type") == "function" and tc.get("function", {}).get("name") == "fetch_taapi_indicator":
                        args = json.loads(tc["function"].get("arguments") or "{}")
                        try:
                            params = {
                                "secret": self.taapi.api_key,
                                "exchange": "binance",
                                "symbol": args["symbol"],
                                "interval": args["interval"],
                            }
                            if args.get("period") is not None:
                                params["period"] = args["period"]
                            if args.get("backtrack") is not None:
                                params["backtrack"] = args["backtrack"]
                            if isinstance(args.get("other_params"), dict):
                                params.update(args["other_params"])
                            ind_resp = requests.get(f"{self.taapi.base_url}{args['indicator']}", params=params, timeout=30).json()
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id"),
                                "name": "fetch_taapi_indicator",
                                "content": json.dumps(ind_resp),
                            })
                        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as ex:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id"),
                                "name": "fetch_taapi_indicator",
                                "content": f"Error: {str(ex)}",
                            })
                continue

            try:
                # Prefer parsed field from structured outputs if present
                if isinstance(message.get("parsed"), dict):
                    parsed = message.get("parsed")
                else:
                    content = message.get("content") or "{}"
                    parsed = json.loads(content)

                if not isinstance(parsed, dict):
                    logging.error("Expected dict payload, got: %s; attempting sanitize", type(parsed))
                    sanitized = _sanitize_output(content if 'content' in locals() else json.dumps(parsed), assets)
                    if sanitized.get("trade_decisions"):
                        return sanitized
                    return {"reasoning": "", "trade_decisions": []}

                reasoning_text = parsed.get("reasoning", "") or ""
                decisions = parsed.get("trade_decisions")

                if isinstance(decisions, list):
                    normalized = []
                    for item in decisions:
                        if isinstance(item, dict):
                            item.setdefault("allocation_usd", 0.0)
                            item.setdefault("tp_price", None)
                            item.setdefault("sl_price", None)
                            item.setdefault("exit_plan", "")
                            item.setdefault("rationale", "")
                            normalized.append(item)
                        elif isinstance(item, list) and len(item) >= 7:
                            normalized.append({
                                "asset": item[0],
                                "action": item[1],
                                "allocation_usd": float(item[2]) if item[2] else 0.0,
                                "tp_price": float(item[3]) if item[3] and item[3] != "null" else None,
                                "sl_price": float(item[4]) if item[4] and item[4] != "null" else None,
                                "exit_plan": item[5] if len(item) > 5 else "",
                                "rationale": item[6] if len(item) > 6 else ""
                            })
                    return {"reasoning": reasoning_text, "trade_decisions": normalized}

                logging.error("trade_decisions missing or invalid; attempting sanitize")
                sanitized = _sanitize_output(content if 'content' in locals() else json.dumps(parsed), assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {"reasoning": reasoning_text, "trade_decisions": []}
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logging.error("JSON parse error: %s, content: %s", e, content[:200])
                # Try sanitizer as last resort
                sanitized = _sanitize_output(content, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {
                    "reasoning": "Parse error",
                    "trade_decisions": [{
                        "asset": a,
                        "action": "hold",
                        "allocation_usd": 0.0,
                        "tp_price": None,
                        "sl_price": None,
                        "exit_plan": "",
                        "rationale": "Parse error"
                    } for a in assets]
                }

        return {
            "reasoning": "tool loop cap",
            "trade_decisions": [{
                "asset": a,
                "action": "hold",
                "allocation_usd": 0.0,
                "tp_price": None,
                "sl_price": None,
                "exit_plan": "",
                "rationale": "tool loop cap"
            } for a in assets]
        }
