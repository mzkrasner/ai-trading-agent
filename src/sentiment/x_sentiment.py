"""X/Twitter sentiment analysis via Grok API's Live Search."""

import os
import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
from src.config_loader import CONFIG

class XSentimentAnalyzer:
    """Fetches real-time sentiment from X/Twitter using Grok's Live Search API."""
    
    def __init__(self):
        """Initialize the Grok API client for sentiment analysis."""
        self.api_key = CONFIG.get("grok_api_key")
        self.enabled = bool(self.api_key)
        
        if not self.enabled:
            logging.info("Grok API key not found - X sentiment analysis disabled")
        else:
            logging.info("X sentiment analyzer initialized with Grok API")
            
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self.model = "grok-4-fast"  # Optimized for speed/cost with good quality
        
    async def get_sentiment_data(self, assets: List[str]) -> Dict:
        """Fetch raw sentiment data from X for the given assets.
        
        Returns non-prescriptive data - just observations from X posts.
        The LLM decides how to interpret and use this information.
        
        Args:
            assets: List of crypto symbols (e.g., ['BTC', 'ETH'])
            
        Returns:
            Dictionary with raw sentiment observations per asset
        """
        if not self.enabled:
            return {}
            
        try:
            # Create prompt that asks for observations, not recommendations
            assets_str = ", ".join(assets)
            prompt = (
                f"Analyze recent X/Twitter posts about these cryptocurrencies: {assets_str}. "
                "Focus on multiple time windows (last 15 minutes, last hour, last 4 hours). "
                "Return a JSON object with this exact format:\n"
                "{\n"
                "  \"observations\": {\n"
                "    \"ASSET\": {\n"
                "      \"post_themes\": [list of main topics being discussed],\n"
                "      \"sentiment_words\": [actual words/phrases from posts],\n"
                "      \"volume_metrics\": {\n"
                "        \"current_posts_per_hour\": number,\n"
                "        \"avg_posts_per_hour_baseline\": number,\n"
                "        \"volume_ratio\": number (current/baseline),\n"
                "        \"spike_detected\": boolean,\n"
                "        \"unusual_activity_description\": \"any abnormal patterns\"\n"
                "      },\n"
                "      \"sentiment_velocity\": {\n"
                "        \"15min_sentiment_shift\": \"getting more bullish/bearish/stable\",\n"
                "        \"1hr_sentiment_shift\": \"description of change\",\n"
                "        \"momentum_description\": \"accelerating/decelerating/stable\",\n"
                "        \"notable_shift_events\": [any sudden changes noticed]\n"
                "      },\n"
                "      \"whale_activity\": {\n"
                "        \"large_transfers_mentioned\": [any whale alerts or large moves],\n"
                "        \"whale_sentiment\": \"what whales/large holders are saying\",\n"
                "        \"institutional_mentions\": [any institutional activity],\n"
                "        \"smart_money_signals\": [any smart money indicators mentioned]\n"
                "      },\n"
                "      \"notable_accounts\": [influential accounts posting],\n"
                "      \"fear_greed_mentions\": \"any specific index values\",\n"
                "      \"price_expectations\": [specific targets mentioned]\n"
                "    }\n"
                "  },\n"
                "  \"search_metadata\": {\n"
                "    \"posts_analyzed\": number,\n"
                "    \"time_windows_checked\": \"15min, 1hr, 4hr\",\n"
                "    \"data_freshness\": \"how recent the newest posts are\"\n"
                "  }\n"
                "}\n"
                "Only report what you observe in posts, don't interpret or recommend. "
                "For volume metrics, compare recent activity to typical baseline. "
                "For sentiment velocity, describe how sentiment is changing over time."
            )
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "search_parameters": {
                    "mode": "on",
                    "sources": [{"type": "x"}],
                    "return_citations": True,  # Get source posts for transparency
                    "limit": 50  # More posts to detect patterns across time windows
                },
                "temperature": 0.3  # Lower temperature for more factual reporting
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract the content and citations
                        content = result['choices'][0]['message']['content']
                        citations = result.get('citations', [])
                        
                        # Try to parse as JSON
                        try:
                            sentiment_data = json.loads(content)
                            # Add citations for transparency
                            sentiment_data['citations'] = citations[:10]  # Top 10 sources
                            return sentiment_data
                        except json.JSONDecodeError:
                            # If not JSON, return as raw observation
                            return {
                                "observations": {
                                    "raw_summary": content,
                                    "note": "Unstructured response from X analysis"
                                },
                                "citations": citations[:10]
                            }
                    else:
                        error_text = await response.text()
                        logging.error(f"Grok API error {response.status}: {error_text}")
                        return {}
                        
        except asyncio.TimeoutError:
            logging.warning("X sentiment request timed out")
            return {}
        except Exception as e:
            logging.error(f"Error fetching X sentiment: {e}")
            return {}
    
    async def get_market_context(self) -> Dict:
        """Get general crypto market sentiment and macro events.
        
        Returns:
            Dictionary with market-wide context (regulatory, institutional, macro events)
        """
        if not self.enabled:
            return {}
            
        try:
            prompt = (
                "What major crypto market news, regulatory developments, or macro events "
                "are being discussed on X right now? Focus on market-wide factors that could "
                "affect all cryptocurrencies, not individual coins. Include:\n"
                "- Regulatory news (SEC, government actions, legal developments)\n"
                "- Exchange developments (outages, hacks, institutional moves)\n"
                "- Institutional activity (ETF flows, corporate treasury moves)\n"
                "- Macro economic factors (Fed policy, global markets)\n"
                "- General market sentiment shifts (fear/greed, risk-on/risk-off)\n\n"
                "Return a JSON object:\n"
                "{\n"
                "  \"regulatory_events\": [list of regulatory news],\n"
                "  \"institutional_activity\": [institutional developments],\n"
                "  \"macro_factors\": [broader economic context],\n"
                "  \"market_sentiment\": \"overall mood description\",\n"
                "  \"notable_events\": [any major events affecting crypto broadly],\n"
                "  \"risk_factors\": [any systemic risks mentioned]\n"
                "}\n"
                "Just report what you observe, don't interpret or recommend."
            )
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "search_parameters": {
                    "mode": "on",
                    "sources": [{"type": "x"}],
                    "return_citations": True,
                    "limit": 30
                },
                "temperature": 0.3
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        citations = result.get('citations', [])
                        
                        try:
                            market_data = json.loads(content)
                            market_data['citations'] = citations[:10]
                            return market_data
                        except json.JSONDecodeError:
                            return {
                                "market_context": content,
                                "citations": citations[:10]
                            }
                    return {}
                    
        except asyncio.TimeoutError:
            logging.warning("Market context request timed out")
            return {}
        except Exception as e:
            logging.error(f"Error fetching market context: {e}")
            return {}
