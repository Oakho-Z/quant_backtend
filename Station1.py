from __future__ import annotations
import logging
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np
import ssl
import urllib3
import matplotlib.pyplot as plt

# Handle SSL certificate verification issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ======================================================
# NLTK Setup - Download VADER lexicon if needed
# ======================================================
import nltk

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Try to instantiate to check if lexicon is available
    _test = SentimentIntensityAnalyzer()
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER is correctly loaded
try:
    _VADER_TEST = SentimentIntensityAnalyzer()
    print("✓ VADER lexicon loaded successfully")
except Exception as e:
    print(f"❌ Error loading VADER: {e}")
    import sys

    sys.exit(1)

# ======================================================
# Logging Configuration
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ======================================================
# Directory Management
# ======================================================
def _ensure_dir(root: Path, sub: str | Path) -> Path:
    """Ensure the directory root/sub exists; create it if not."""
    path = root / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_dirs(base: str | Path, mode: str) -> dict[str, Path]:
    """
    Build standardized output directory structure:
        results/<mode>/data/
    Returns:
        dict with "data_dir" pointing to the created path.
    """
    root = Path(base).expanduser().resolve()
    results_root = _ensure_dir(root, "../../QuantEye/results")
    mode_root = _ensure_dir(results_root, mode)
    return {
        "data_dir": _ensure_dir(mode_root, "data")
    }

# ======================================================
# Stage 1 – News (Unstructured) ETL
# ======================================================
NEWS_URL = "https://data-api.coindesk.com/news/v1/article/list"


def fetch_news_range(
        api_key: str | None,
        start_dt: datetime,
        end_dt: datetime,
        lang: str = "EN"
) -> pd.DataFrame:
    """Fetch CoinDesk news articles between start_dt and end_dt."""
    out: list[pd.DataFrame] = []

    while end_dt > start_dt:
        query_ts = int(end_dt.timestamp())
        query_day = end_dt.strftime("%Y-%m-%d")
        logging.info(f"Requesting articles up to {query_day} …")

        resp = requests.get(f"{NEWS_URL}?lang={lang}&to_ts={query_ts}")
        if not resp.ok:
            logging.error(f"Request failed with status {resp.status_code}")
            break

        data = resp.json().get("Data", [])
        d = pd.DataFrame(data)
        if d.empty:
            logging.info(f"No data returned for {query_day}, stopping.")
            break

        d["date"] = pd.to_datetime(d["PUBLISHED_ON"], unit="s")
        out.append(d[d["date"] >= start_dt])

        # Move one second before the earliest article to avoid duplication
        end_dt = datetime.utcfromtimestamp(d["PUBLISHED_ON"].min() - 1)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def stage1_load_news(
        api_key: str | None,
        start_dt: datetime,
        end_dt: datetime,
        data_dir: Path
) -> pd.DataFrame:
    """Stage 1 – Fetch and store raw news data."""
    tic = time.time()
    logging.info("Stage 1 (NEWS) – Downloading CoinDesk news …")

    df = fetch_news_range(api_key, start_dt, end_dt)

    drop_cols = [
        "GUID", "PUBLISHED_ON_NS", "IMAGE_URL", "SUBTITLE", "AUTHORS",
        "URL", "UPVOTES", "DOWNVOTES", "SCORE", "CREATED_ON", "UPDATED_ON",
        "SOURCE_DATA", "CATEGORY_DATA", "STATUS", "SOURCE_ID", "TYPE", "LANG", "PUBLISHED_ON"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df.columns = df.columns.str.lower()
    other_cols = [c for c in df.columns if c not in ["date", "id"]]

    # remove empty headlines or body
    df = df.dropna(subset=["title", "body"])

    # remove duplicate articles (same title + body)
    df = df.drop_duplicates(subset=["title", "body"])

    # Standardize all "empty" forms
    df["keywords"] = df["keywords"].astype(str).replace(["", "[]", "{}", "null"], np.nan)
    # Then uniformly fill MARKET_WIDE
    df["keywords"] = df["keywords"].fillna("MARKET_WIDE")

    df = df[["date", "id"] + other_cols]
    if "sentiment" in df.columns:
        df["positive"] = np.where(df["sentiment"].str.upper() == "POSITIVE", 1, 0)
        df = df.drop(columns="sentiment")
    else:
        df["positive"] = np.nan

    out = data_dir / "stage_1_news_raw.csv"
    df.to_csv(out, index=False)
    logging.info(f"Saved news Stage 1 -> {out} ({time.time() - tic:.2f}s)")
    return df

# ======================================================
# Stage 1 – Crypto (Structured) ETL
# ======================================================
BASE_URL = "https://data-api.coindesk.com"


def _headers(api_key: str) -> dict[str, str]:
    return {"authorization": f"Apikey {api_key}"}


def get_top_coins(
        api_key: str,
        pages: List[int],
        limit: int = 100,
        sort_by: str = "CIRCULATING_MKT_CAP_USD"
) -> List[str]:
    """Return top coin symbols based on market cap."""
    coins = []

    for page in pages:
        url = (
            f"{BASE_URL}/asset/v1/top/list?"
            f"page={page}&page_size={limit}&sort_by={sort_by}"
            f"&sort_direction=DESC&groups=ID,BASIC,MKT_CAP"
        )
        resp = requests.get(url, headers=_headers(api_key), timeout=30)
        data = resp.json()

        if "Data" not in data or "LIST" not in data["Data"]:
            logging.warning(f"Page {page} returned no data: {data.get('Message')}")
            continue

        for coin in data["Data"]["LIST"]:
            coins.append(coin["SYMBOL"])

        logging.info(f"Collected {len(data['Data']['LIST'])} symbols from page {page}")

    if not coins:
        raise RuntimeError("No symbols retrieved. Check API key or parameters.")
    return coins


def get_daily_ohlcv(
        symbol: str,
        api_key: str,
        limit: int = 2000,
        currency: str = "USD",
        max_retries: int = 3,
        verbose: bool = False  # ✅ New parameter: whether to print detailed debug logs
) -> pd.DataFrame | None:
    """
    Download historical OHLCV data for a coin with retry mechanism + optional verbose diagnostics.
    """

    url = (
        f"{BASE_URL}/index/cc/v1/historical/days"
        f"?market=cadli&instrument={symbol}-{currency}"
        f"&limit={limit}&aggregate=1&fill=true&apply_mapping=true"
    )

    for attempt in range(1, max_retries + 1):
        try:
            # === 1️⃣ Print basic info before the request (hide authorization)
            safe_headers = {k: ("***" if k.lower() == "authorization" else v)
                            for k, v in _headers(api_key).items()}
            logging.info(f"[{symbol}] Attempt {attempt}/{max_retries} → GET {url}")
            logging.debug(f"[{symbol}] Req-Headers: {safe_headers}")

            # === 2️⃣ Send the request
            resp = requests.get(url, headers=_headers(api_key), timeout=30)

            # === 3️⃣ If verbose is enabled, print more detailed HTTP debug info
            if verbose:
                req_safe_hdr = {k: ("***" if k.lower() == "authorization" else v)
                                for k, v in resp.request.headers.items()}
                logging.info(
                    "[%s] HTTP %s | req-hdrs=%s | rate-limit-remaining=%s",
                    symbol,
                    resp.status_code,
                    req_safe_hdr,
                    resp.headers.get("x-ratelimit-remaining")
                )
                # Print first 500 characters to avoid overly long logs
                logging.debug("[%s] raw-json=%s", symbol, resp.text[:500])

            # === 4️⃣ Parse JSON
            data = resp.json()
            if data.get("Response") == "Error" or "Data" not in data:
                logging.warning(f"[{symbol}] No data: {data.get('Message')}")
                return None

            df = pd.DataFrame(data["Data"])
            df["date"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
            df = df.rename(columns={
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "VOLUME": "btc_volume",
                "QUOTE_VOLUME": "usd_volume"
            })
            df = df[["date", "open", "high", "low", "close", "usd_volume", "btc_volume"]].copy()
            df["usd_volume_mil"] = df["usd_volume"] / 1e6
            df["symbol"] = symbol
            df.set_index(["symbol", "date"], inplace=True)

            # === 5️⃣ Basic data validation
            df = df[df["low"] <= df["open"]]
            df = df[df["low"] <= df["close"]]
            df = df[df["close"] <= df["high"]]
            df = df[df["open"] <= df["high"]]
            df = df[df["usd_volume"] >= 0]
            df = df[df["btc_volume"] >= 0]
            df = df[~((df["open"] == 0) & (df["high"] == 0) & (df["low"] == 0) & (df["close"] == 0))]
            df = df.dropna()

            return df

        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            logging.warning(
                f"[{symbol}] SSL/Connection error on attempt {attempt}/{max_retries} → {e}"
            )
            if attempt < max_retries:
                # Exponential backoff to avoid frequent requests
                time.sleep(2 * attempt)
            else:
                logging.error(f"[{symbol}] Failed after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            logging.error(f"[{symbol}] Unexpected error: {e}")
            return None


def stage1_etl_crypto(
        api_key: str,
        pages: List[int],
        top_limit: int = 100,
        history_limit: int = 2000,
        currency: str = "USD",
        sleep_sec: float = 0.5,
        data_dir: Path | None = None
) -> pd.DataFrame:
    """Stage 1 – Download crypto OHLCV data and save as CSV."""
    logging.info("Fetching list of top coins …")
    symbols = get_top_coins(api_key, pages, top_limit)
    logging.info(f"Total symbols collected: {len(symbols)}")

    all_frames = []
    failed_symbols = []

    for i, sym in enumerate(symbols):
        logging.info(f"Downloading history for {sym} ({i+1}/{len(symbols)})")
        df = get_daily_ohlcv(sym, api_key, history_limit, currency)
        if df is not None:
            all_frames.append(df)
        else:
            failed_symbols.append(sym)
        time.sleep(sleep_sec)

    if failed_symbols:
        logging.warning(f"Failed to download data for {len(failed_symbols)} symbols: {failed_symbols[:10]}...")

    if not all_frames:
        raise RuntimeError("No historical data retrieved.")

    logging.info(f"Successfully downloaded data for {len(all_frames)} symbols")
    data = pd.concat(all_frames).sort_index()

    if data_dir is not None:
        out_path = data_dir / "stage_1_crypto_data.csv"
        data.to_csv(out_path)
        logging.info(f"Stage 1 Crypto CSV -> {out_path}")

    return data

# ======================================================
# Unified Stage 1 Entry Point
# ======================================================
def run_stage1(
        mode: str,
        api_key: str,
        base_dir: str = ".",
        **kwargs
):
    """
    Unified entry point for Stage 1:
        mode = "news" → fetch CoinDesk news
        mode = "crypto" → fetch OHLCV for top coins
    """
    dirs = build_dirs(base_dir, mode)

    if mode == "news":
        return stage1_load_news(
            api_key=api_key,
            start_dt=kwargs["start_dt"],
            end_dt=kwargs["end_dt"],
            data_dir=dirs["data_dir"]
        )

    elif mode == "crypto":
        return stage1_etl_crypto(
            api_key=api_key,
            pages=kwargs["pages"],
            top_limit=kwargs.get("top_limit", 100),
            history_limit=kwargs.get("history_limit", 2000),
            currency=kwargs.get("currency", "USD"),
            sleep_sec=kwargs.get("sleep_sec", 0.5),
            data_dir=dirs["data_dir"]
        )

    else:
        raise ValueError("mode must be 'news' or 'crypto'")


# ======================================================
# Station 2 – Feature Engineering (Market + News)
# ======================================================
# === Initialize VADER ===
_VADER = SentimentIntensityAnalyzer()
# === Update VADER with crypto+finance lexicon
custom_words = {
    # ─── General market tone ─────────────────────────────────────────────
    "bullish": 3.2,
    "bearish": -3.2,
    "rally": 2.4,
    "selloff": -2.6,
    "soar": 3.0,
    "plummet": -3.0,
    "skyrocket": 3.5,
    "tank": -2.8,
    "breakout": 2.6,
    "breakdown": -2.6,
    "recovery": 2.3,
    "capitulation": -3.3,
    "moon": 3.8,
    "moonshot": 3.5,
    "dip": -0.7,
    "buy_the_dip": 2.5,
    "crash": -3.2,
    "correction": -1.2,
    "bubble": -2.4,
    "dead_cat_bounce": -2.5,
    "all_time_high": 3.6,
    "all_time_low": -3.6,
    "bull_run": 3.4,
    "market_meltdown": -3.8,
    "flash_crash": -3.5,
    "volatility_spike": -1.5,
    "risk_on": 1.4,
    "risk_off": -1.4,
    "safe_haven": 1.1,
    # ─── Trader slang / emotions ────────────────────────────────────────
    "fomo": -0.8,
    "bagholder": -2.0,
    "whale": 1.0,
    "hodl": 2.1,
    "hodling": 2.0,
    "fear": -2.1,
    "greed": 1.8,
    "panic_sell": -3.0,
    "short_squeeze": 2.0,
    "pump": 1.8,
    "dump": -2.6,
    "pump_and_dump": -3.6,
    "rugpull": -3.5,
    # ─── Corporate / earnings language ───────────────────────────────────
    "earnings_beat": 2.4,
    "earnings_miss": -2.4,
    "guidance_raise": 2.2,
    "guidance_cut": -2.2,
    "profit_take": 1.5,
    "profit_warning": -2.6,
    "dividend_hike": 2.3,
    "dividend_cut": -2.3,
    "share_buyback": 1.8,
    "share_dilution": -1.8,
    "upgrade": 2.1,
    "downgrade": -2.1,
    "rating_boost": 1.8,
    "rating_cut": -1.8,
    # ─── Macro & policy terms ───────────────────────────────────────────
    "quantitative_easing": 0.5,
    "quantitative_tightening": -1.2,
    "interest_rate_hike": -1.3,
    "interest_rate_cut": 1.3,
    "inflation_surge": -2.4,
    "deflation": -1.0,
    "stagflation": -3.0,
    "recession": -3.4,
    "depression": -3.8,
    "soft_landing": 1.5,
    "hard_landing": -2.5,
    "economic_expansion": 2.4,
    "stimulus": 1.2,
    "credit_crunch": -3.1,
    "yield_curve_inversion": -2.7,
    # ─── Balance-sheet / distress ───────────────────────────────────────
    "default": -3.5,
    "bankruptcy": -4.0,
    "chapter_11": -3.6,
    "insolvency": -3.6,
    "liquidation": -3.0,
    "margin_call": -2.8,
    "asset_write_down": -2.5,
    "leverage_buyout": 0.2,
    # ─── Crypto-specific positive ───────────────────────────────────────
    "halving": 1.4,
    "hashrate_record": 2.3,
    "mainnet_launch": 2.5,
    "whitepaper_release": 1.6,
    "etf_approval": 2.8,
    "token_listed": 2.0,
    "layer2_scaling": 1.1,
    "gas_fee_drop": 1.7,
    "airdrop": 1.3,
    "alt_season": 2.0,
    "nft_boom": 1.9,
    "metaverse_boost": 2.0,
    "token_burn": 1.5,
    "yield_farming": 1.2,
    "liquidity_mining": 1.0,
    "stablecoin_recovery": 2.1,
    # ─── Crypto-specific negative ───────────────────────────────────────
    "crypto_winter": -3.3,
    "depeg": -3.1,
    "hack": -3.4,
    "exploit": -2.8,
    "bridge_exploit": -3.2,
    "smart_contract_bug": -2.5,
    "exit_scam": -4.0,
    "treasury_drain": -2.7,
    "gas_fee_spike": -1.7,
    "testnet_delay": -1.4,
    "token_delisted": -2.0,
    "pow_ban": -2.3,
    "regulatory_crackdown": -2.2,
    "etf_rejection": -2.8,
    "cease_and_desist": -2.8,
    "fraudulent": -4.0,
    "securities_violation": -3.0,
    "class_action": -2.9,
    "hard_fork": -0.2,
    "fork": 0.0,
    "soft_fork": 0.2,
    "ordinals_collapse": -1.5,
    "nft_crash": -1.9,
    "impermanent_loss": -1.6,
    "validator_slash": -2.0,
    # ─── Neutral / mild, but useful context ─────────────────────────────
    "contango": 0.1,
    "backwardation": -0.1,
    "open_interest_surge": 0.9,
    "funding_rate_flip": 0.3,
    "market_depth": 0.4,
    "thin_liquidity": -1.1,
    "governance_vote": 0.5,
    "license_grant": 2.2,
    "settlement_reached": 1.0,
    "bailout": -0.8,
    "flight_to_quality": 0.7,
    "esg_compliance": 1.0,
    # ── market / price action ──
    "onboarding": -0.4,         # onboarding paused → mildly negative :contentReference[oaicite:0]{index=0}
    "offboarded": -2.0,         # lost banking access → negative :contentReference[oaicite:1]{index=1}
    "listing": 1.5,             # exchange listing → positive :contentReference[oaicite:2]{index=2}
    "delisting": -1.5,
    "gainer": 2.2,              # top daily gainer :contentReference[oaicite:3]{index=3}
    "resistance": -0.8,         # price stuck below resistance :contentReference[oaicite:4]{index=4}
    "support_level": 0.8,
    "leg_up": 1.3,
    "momentum": 1.4,            # “building momentum” :contentReference[oaicite:5]{index=5}
    "turbulence": -1.5,
    "liquidations": -1.9,       # large liquidations :contentReference[oaicite:6]{index=6}
    "retrace": -0.5,
    "retest": 0.6,
    "parabolic": 3.0,
    "melt_up": 2.8,
    "freefall": -3.0,
    "oversold": 1.0,
    "overbought": -1.1,
    "grind_higher": 1.3,
    "slump": -2.0,
    "bounce": 1.3,
    "headwinds": -1.2,
    "tailwinds": 1.2,
    # ── trader / crypto slang ──
    "ape_in": 1.6,
    "diamond_hands": 2.4,
    "paper_hands": -1.8,
    "rekt": -3.2,
    "degen": -1.0,
    "flippening": 1.7,
    # ── risk & attack vectors ──
    "front_run": -2.2,
    "sandwich_attack": -2.6,
    "oracle_failure": -2.9,
    "slashing": -2.0,
    "unstake": -0.6,
    # ── DeFi / staking ──
    "staking_reward": 1.7,
    "overleveraged": -2.3,
    "hashwar": -2.5,
    "ghost_chain": -2.4,
    # ── corporate / deal flow ──
    "pivot": 0.7,
    "windfall": 3.0,
    "oversubscribed": 2.0,
    "shortfall": -2.1,
    # ── macro / policy ──
    "hawkish": -0.9,
    "dovish": 0.9,
    "taper": -0.7,
    "fiscal_cliff": -2.5,
    # ── compliance / regulation ──
    "whitelist": 1.2,
    "blacklist": -1.2,
    # ── gas & fees ──
    "gasless": 1.5,
    "peg_restore": 2.0
}
_VADER.lexicon.update(custom_words)
logging.info(f"✅ Custom VADER lexicon injected, total terms added: {len(custom_words)}")
# Sample check a few words to see if they are effective
sample_check_words = [
    "profit", "bankruptcy", "rug pull",
    "moon", "bearish", "bloodbath",
    "airdrop", "token unlock", "halving", "etf approval", "exploit", "hack"
]
for w in sample_check_words:
    logging.info(f"Word '{w}' sentiment score → {_VADER.lexicon.get(w.lower(), 'NOT FOUND')}")


# Coin name mapping table
COIN_NAME_MAPPING = {
    # Major coins
    "BITCOIN": "BTC",
    "BITCOIN (BTC)": "BTC",
    "BTC": "BTC",
    "BTCUSD": "BTC",
    "BTCUSDT": "BTC",
    "BTCEUR": "BTC",
    "BTCGBP": "BTC",
    "BTCUSDC": "BTC",

    "ETHEREUM": "ETH",
    "ETHEREUM (ETH)": "ETH",
    "ETH": "ETH",
    "ETHUSD": "ETH",
    "ETHBTC": "ETH",
    "ETHER": "ETH",

    "RIPPLE": "XRP",
    "XRP": "XRP",

    "BINANCE COIN": "BNB",
    "BNB": "BNB",
    "BNBBTC": "BNB",

    "SOLANA": "SOL",
    "SOL": "SOL",

    "CHAINLINK": "LINK",
    "LINK": "LINK",

    "SHIBA INU": "SHIB",
    "SHIB": "SHIB",

    "CARDANO": "ADA",
    "ADA": "ADA",

    "POLKADOT": "DOT",
    "DOT": "DOT",

    "AVALANCHE": "AVAX",
    "AVAX": "AVAX",

    "POLYGON": "MATIC",
    "MATIC": "MATIC",
    "POL": "MATIC",  # Polygon rebrand

    "TRON": "TRX",
    "TRX": "TRX",

    "DOGECOIN": "DOGE",
    "DOGE": "DOGE",

    "LITECOIN": "LTC",
    "LTC": "LTC",

    # More coins
    "UNISWAP": "UNI",
    "UNI": "UNI",

    "COSMOS": "ATOM",
    "ATOM": "ATOM",

    "NEAR PROTOCOL": "NEAR",
    "NEAR": "NEAR",

    "APTOS": "APT",
    "APT": "APT",

    "ARBITRUM": "ARB",
    "ARB": "ARB",

    "OPTIMISM": "OP",
    "OP": "OP",

    "FILECOIN": "FIL",
    "FIL": "FIL",

    "INTERNET COMPUTER": "ICP",
    "ICP": "ICP",

    "THE GRAPH": "GRT",
    "GRT": "GRT",

    "CURVE": "CRV",
    "CRV": "CRV",

    "FANTOM": "FTM",
    "FTM": "FTM",

    "STELLAR": "XLM",
    "XLM": "XLM",

    "VECHAIN": "VET",
    "VET": "VET",

    "SANDBOX": "SAND",
    "SAND": "SAND",

    "AAVE": "AAVE",

    "ENJIN": "ENJ",
    "ENJ": "ENJ",

    "MAKER": "MKR",
    "MKR": "MKR",

    # Layer 2
    "IMMUTABLE": "IMX",
    "IMX": "IMX",

    # Memecoins
    "PEPE": "PEPE",
    "FLOKI": "FLOKI",
    "BONK": "BONK",
    "WIF": "WIF",

    # Stablecoins
    "TETHER": "USDT",
    "TETHER (USDT)": "USDT",
    "USDT": "USDT",

    "USD COIN": "USDC",
    "USDC": "USDC",

    "DAI": "DAI",
}

# -------------------------------
# Pre-aggregation news processing
# -------------------------------
def stage2_clean_text_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple cleaning of news body + title, concatenated for VADER usage.
    """
    logging.info("Stage 2 (NEWS) – basic cleaning (title+body) …")
    df = news_df.copy()

    # Concatenate title + body to avoid missing title sentiment
    df["reviewText"] = (
            df["title"].fillna("").astype(str) + ". " + df["body"].fillna("").astype(str)
    )
    return df


def stage2_sentiment_scores(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run VADER on news text → neg, neu, pos, compound + binary sentiment label
    """
    logging.info("Stage 2 (NEWS) – Running VADER sentiment …")
    df = news_df.copy()
    df[["neg", "neu", "pos", "compound"]] = df["reviewText"].apply(
        lambda txt: pd.Series(_VADER.polarity_scores(txt))
    )
    df["sentiment"] = (df["compound"] > 0.05).astype(int)
    return df


def debug_keywords_symbols(news_df: pd.DataFrame, crypto_symbols: set) -> None:
    """
    Debug function: analyze matching between keywords and symbols.
    """
    logging.info("=== Debugging Keywords vs Symbols ===")

    # Analyze keywords - note using | as separator
    all_keywords = []
    for kw in news_df["keywords"]:
        if isinstance(kw, str) and kw != "MARKET_WIDE":
            # Use | separator
            all_keywords.extend([k.strip().upper() for k in kw.split("|")])

    unique_keywords = set(all_keywords)
    logging.info(f"Unique keywords found: {len(unique_keywords)}")
    logging.info(f"Sample keywords: {list(unique_keywords)[:20]}")

    # Analyze symbols
    logging.info(f"Crypto symbols count: {len(crypto_symbols)}")
    logging.info(f"Sample symbols: {list(crypto_symbols)[:20]}")

    # Find matches
    matched = unique_keywords.intersection(crypto_symbols)
    logging.info(f"Matched symbols: {len(matched)}")
    logging.info(f"Matched examples: {list(matched)[:20]}")

    # Find unmatched keywords
    unmatched_keywords = unique_keywords - crypto_symbols
    logging.info(f"Unmatched keywords: {len(unmatched_keywords)}")
    logging.info(f"Unmatched examples: {list(unmatched_keywords)[:20]}")


def stage2_aggregate_sentiment(news_df: pd.DataFrame, t_lag: int = 1) -> pd.DataFrame:
    """
    Aggregate news sentiment (symbol, date) and apply T+1 lag to avoid forward-looking bias.

    T+1 principle: articles published on day t only affect sentiment features of day t+1
    - First apply T+1 lag at daily level
    - Then aggregate to weekly (W-WED)
    - Ensure weekly timestamps align with market data
    """
    logging.info("Stage 2 (NEWS) – Aggregating weekly sentiment with T+1 principle…")

    # Process keywords field - use | separator, apply coin mapping
    def parse_keywords(x):
        if pd.isna(x) or x == "MARKET_WIDE":
            return ["MARKET_WIDE"]

        if isinstance(x, str):
            # Use | separator and clean
            parts = [k.strip().upper() for k in x.split("|") if k.strip()]

            # Extract coin-related keywords
            coins = []
            for part in parts:
                # Directly match mapping table
                if part in COIN_NAME_MAPPING:
                    coins.append(COIN_NAME_MAPPING[part])
                # Check if already a symbol (2-5 letters) and exclude common non-coin terms
                elif 2 <= len(part) <= 5 and part.isalpha() and not part in ["NEWS", "DEFI", "AI", "SEC", "USD", "EUR",
                                                                             "CEO", "IPO", "ETF", "API", "DAO", "NFT",
                                                                             "DXJ", "M&A", "RWA", "WEB3", "TECH",
                                                                             "CORE", "FLOW"]:
                    coins.append(part)

            # If no coins found, return MARKET_WIDE
            return coins if coins else ["MARKET_WIDE"]

        return ["MARKET_WIDE"]

    news_df["keywords"] = news_df["keywords"].apply(parse_keywords)

    # explode keywords → each article can correspond to multiple symbols
    df_expanded = news_df.explode("keywords").copy()
    df_expanded.rename(columns={"keywords": "symbol"}, inplace=True)
    df_expanded["date"] = pd.to_datetime(df_expanded["date"])

    # Log symbol distribution
    symbol_counts = df_expanded["symbol"].value_counts()
    logging.info(f"Top symbols in news: \n{symbol_counts.head(20)}")

    # First aggregate to daily
    daily_sent = (
        df_expanded.groupby(["symbol", pd.Grouper(key="date", freq="D")])
            .agg(
            avg_compound=("compound", "mean"),
            pos_ratio=("sentiment", "mean"),
            news_count=("id", "count")
        )
            .reset_index()
    )

    # Then aggregate to weekly (week ending Wednesday, consistent with market data)
    weekly_sent = (
        daily_sent.set_index("date")
        .groupby("symbol")
        .resample("W-WED", closed='left', label='left')
        .agg({
            "avg_compound": "mean",
            "pos_ratio": "mean",
            "news_count": "sum"
        })
        .reset_index()
    )

    # Shift weekly forward by one week (implement T+1 weekly principle)
    weekly_sent["date"] = weekly_sent["date"] + pd.Timedelta(weeks=1)

    logging.info(f"Stage 2 (NEWS) – Weekly aggregated sentiment shape={weekly_sent.shape}")
    logging.info(f"Date range after T+1: {weekly_sent['date'].min()} to {weekly_sent['date'].max()}")

    return weekly_sent

# -------------------------------
# Market factors
# -------------------------------
# ==========================
# ✅ Plot
# ==========================
def plot_crypto_trend_factors(dfw: pd.DataFrame, symbol: str = "ETH"):

    df_symbol = dfw[dfw["symbol"] == symbol].copy().sort_values("date")

    if df_symbol.empty:
        logging.warning(f"No data available for {symbol}, cannot plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df_symbol["date"], df_symbol["momentum_28"], label="Momentum (4-week)")
    plt.plot(df_symbol["date"], df_symbol["rolling_mean_30d"], label="Rolling Mean (30d)")
    plt.plot(df_symbol["date"], df_symbol["return"], label="Weekly Return")

    plt.title(f"{symbol} Trend-Based Factors")
    plt.xlabel("Date")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save picture
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"results/plots/{symbol}_trend_factors.png", dpi=200)
    plt.show()

def stage2_feature_engineering_market(
        tidy_prices: pd.DataFrame | None = None,
        csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate market factors: volume shocks, momentum, volatility, short-term reversal, weekly return
    """
    if tidy_prices is None:
        if csv_path is None:
            raise ValueError("Provide either tidy_prices or csv_path.")
        logging.info("Reading Stage 1 CSV from %s", csv_path)
        tidy_prices = pd.read_csv(
            csv_path, index_col=["symbol", "date"], parse_dates=["date"]
        )

    df = tidy_prices.reset_index().sort_values(["symbol", "date"]).copy()

    # Volume shocks
    # Before volume shock calculation, protect against 0 trading volume
    df["usd_volume_safe"] = df["usd_volume"].replace(0, 1e-8)

    for m in [7, 14, 21, 28, 42]:
        rolling_mean = (
            df.groupby("symbol")["usd_volume_safe"]
            .shift(1)
            .rolling(m, min_periods=m)
            .mean()
        )
        df[f"v_{m}d"] = np.log(df["usd_volume_safe"]) - np.log(rolling_mean)

    # Daily log returns
    df["log_return"] = np.log1p(df.groupby("symbol")["close"].pct_change())
    df["log_return"] = np.where(df["log_return"] > 2, 2, df["log_return"])
    df = df.replace([-np.inf, np.inf], np.nan)

    # Momentum & Volatility
    for m in [14, 21, 28, 42, 90]:
        shifted = df.groupby("symbol")["log_return"].shift(7)
        df[f"momentum_{m}"] = np.exp(shifted.rolling(m, min_periods=m).sum()) - 1.0
        df[f"volatility_{m}"] = (
                                    df.groupby("symbol")["log_return"]
                                        .rolling(m, min_periods=m)
                                        .std()
                                        .reset_index(level=0, drop=True)
                                ) * np.sqrt(365.0)


    # === Required factors ===
    # Rolling mean return
    df['rolling_mean_7d'] = df.groupby('symbol')['log_return'] \
        .rolling(7, min_periods=3).mean().reset_index(level=0, drop=True)
    df['rolling_mean_30d'] = df.groupby('symbol')['log_return'] \
        .rolling(30, min_periods=10).mean().reset_index(level=0, drop=True)

    # Value-at-Risk (VaR 95%) - 5% quantile of returns over the past 30 days
    df['VaR_30d_95'] = df.groupby('symbol')['log_return'] \
        .rolling(30, min_periods=10).quantile(0.05).reset_index(level=0, drop=True)

    # Maximum Drawdown (MDD 30d)
    def rolling_mdd(x):
        """Calculate maximum drawdown within a 30-day window"""
        return (x / x.cummax() - 1).min()

    df['mdd_30d'] = df.groupby('symbol')['close'] \
        .rolling(30, min_periods=10).apply(rolling_mdd, raw=False).reset_index(level=0, drop=True)

    # === Product enhancement priority 1: liquidity proxy factor ===
    # Liquidity score (short-term vs long-term)
    df["rolling_volume_30d"] = df.groupby("symbol")["usd_volume"].rolling(30, min_periods=10).mean().reset_index(
        level=0, drop=True)
    df["rolling_volume_7d"] = df.groupby("symbol")["usd_volume"].rolling(7, min_periods=3).mean().reset_index(level=0,
                                                                                                              drop=True)
    df["liquidity_score"] = df["rolling_volume_7d"] / df["rolling_volume_30d"]

    # ✅ Remove intermediate variables, keep only liquidity_score
    df.drop(columns=["rolling_volume_7d", "rolling_volume_30d"], inplace=True)
    # Drop log_return (only an intermediate variable)
    df.drop(columns=["log_return"], inplace=True)
    # Weekly resample (align with news data)
    dfw = (
        df.set_index("date")
        .groupby("symbol")
        .resample(
            "W-WED",
            closed="left",
            label="left",
            include_groups=False
        )
        .apply(lambda g: g.ffill().iloc[-1])  # ✅ Forward fill within week and take the last valid value
        .reset_index()  # ✅ Restore symbol/date as columns
    )

    dfw["return"] = dfw.groupby("symbol")["close"].pct_change()
    dfw["return"] = np.where(dfw["return"] > 2, 2, dfw["return"])
    dfw["strev_weekly"] = dfw["return"]
    dfw = dfw.reset_index()

    # Remove stablecoins / wrapped tokens
    stable_tickers = ["USD", "USDT", "USDC", "TUSD", "BUSD", "DAI", "SUSD", "FRAX", "USDD", "UST", "USTC", "EUR",
                      "EURT", "EURS", "EUROC", "AEUR", "AGEUR", "PYUSD"]
    wrapped_tickers = ["WBTC", "WETH", "WBNB", "WSTETH", "WUSDC", "WUSDT", "WCRO", "WFTM", "WTRX", "WCELO", "WFIL",
                       "WGLMR", "WXRP", "WLTC", "WSOL", "WADA"]
    tickers_to_drop = {t.upper() for t in stable_tickers + wrapped_tickers}
    is_exact_drop = dfw["symbol"].str.upper().isin(tickers_to_drop)
    has_usd_substr = dfw["symbol"].str.upper().str.contains("USD", na=False)
    dfw = dfw[~(is_exact_drop | has_usd_substr)].copy()

    # Clean abnormal values
    dfw = dfw[dfw["return"] > -1.0]
    dfw = dfw.replace([-np.inf, np.inf], np.nan)

    logging.info("✅ Station 2 (MARKET) – Technical factors done, shape=%s", dfw.shape)

    #
    try:
        plot_crypto_trend_factors(dfw, symbol="ETH")
    except Exception as e:
        logging.warning(f"Plot failed: {e}")

    return dfw




# -------------------------------
# Merge market factors + news factors
# -------------------------------
def stage2_merge_market_news(market_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge market factors and news sentiment factors (align on symbol, date)
    Improved version: adds debug info and MARKET_WIDE as fallback
    """
    logging.info("Station 2 – Merging market & news sentiment features …")

    # Ensure symbol columns are uppercase
    market_df["symbol"] = market_df["symbol"].str.upper()
    sentiment_df["symbol"] = sentiment_df["symbol"].str.upper()

    # Check symbol matching
    market_symbols = set(market_df["symbol"].unique())
    sentiment_symbols = set(sentiment_df["symbol"].unique())

    logging.info(f"Market symbols: {len(market_symbols)}")
    logging.info(f"Sentiment symbols: {len(sentiment_symbols)}")

    # Check date ranges
    logging.info(f"Market date range: {market_df['date'].min()} to {market_df['date'].max()}")
    logging.info(f"Sentiment date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")

    matched_symbols = market_symbols.intersection(sentiment_symbols)
    logging.info(f"Matched symbols for merge: {len(matched_symbols)}")
    logging.info(f"Examples of matched: {list(matched_symbols)[:10]}")

    # Perform merge
    merged = market_df.merge(sentiment_df, on=["symbol", "date"], how="left")

    # Merge result statistics
    has_sentiment = merged["avg_compound"].notna().sum()
    total_rows = len(merged)
    logging.info(f"Rows with sentiment data: {has_sentiment}/{total_rows} ({has_sentiment/total_rows*100:.1f}%)")

    # Handle MARKET_WIDE fallback
    if "MARKET_WIDE" in sentiment_symbols:
        market_wide_sent = sentiment_df[sentiment_df["symbol"] == "MARKET_WIDE"].copy()

        # Drop symbol column but keep date
        market_wide_sent = market_wide_sent.drop(columns=["symbol"])

        # Add suffix for each column, exclude date
        cols_to_rename = {col: f"{col}_market" for col in market_wide_sent.columns if col != "date"}
        market_wide_sent = market_wide_sent.rename(columns=cols_to_rename)

        # Merge market-wide sentiment into all records
        merged = merged.merge(market_wide_sent, on="date", how="left")

        # For rows without individual sentiment, use market-wide sentiment
        for col in ["avg_compound", "pos_ratio", "news_count"]:
            if f"{col}_market" in merged.columns:
                merged[col] = merged[col].fillna(merged[f"{col}_market"])
                merged = merged.drop(columns=[f"{col}_market"])
    logging.info("Station 2 (MARKET) – Technical factors done, shape=%s", market_df)
    # Final statistics
    final_has_sentiment = merged["avg_compound"].notna().sum()
    logging.info(
        f"After MARKET_WIDE fallback: {final_has_sentiment}/{total_rows} rows have sentiment ({final_has_sentiment/total_rows*100:.1f}%)")

    logging.info("Station 2 – Final feature matrix shape=%s", merged.shape)
    return merged

def compute_market_sentiment_index(sentiment_df: pd.DataFrame, smooth_window: int = 3) -> pd.DataFrame:
    """
    Compute weekly market-wide sentiment index:
      - weighted by news_count (so big weeks with more news have more impact)
      - smoothed by rolling mean to reduce noise
    Returns:
      df with [date, market_sentiment_index]
    """
    logging.info("Computing market-wide sentiment index …")

    df = sentiment_df.copy()

    # 1️⃣ Weighted sentiment by news_count
    df["weighted_sent"] = df["avg_compound"] * df["news_count"]

    # 2️⃣ Aggregate weekly: sum of weighted_sent / sum of news_count
    market_sentiment = (
        df.groupby("date")
          .agg(
              total_weighted_sent=("weighted_sent", "sum"),
              total_news=("news_count", "sum")
          )
          .reset_index()
    )

    # 3️⃣ Avoid division by zero
    market_sentiment["market_sentiment_index"] = (
        market_sentiment["total_weighted_sent"] /
        market_sentiment["total_news"].replace(0, np.nan)
    )

    # 4️⃣ Smooth the series to reduce noise (3-week rolling mean by default)
    market_sentiment["market_sentiment_index"] = (
        market_sentiment["market_sentiment_index"]
        .rolling(smooth_window, min_periods=1)
        .mean()
    )

    logging.info(
        "Market sentiment index date range: %s → %s",
        market_sentiment["date"].min(), market_sentiment["date"].max()
    )
    return market_sentiment[["date", "market_sentiment_index"]]




# -------------------------------
# Station 2 master controller
# -------------------------------
def run_stage2(
        market_csv: Path,
        news_csv: Path,
        out_dir: Path,
        output_filename: str = "station2_feature_matrix.csv"
) -> pd.DataFrame:
    """
    Station 2 full pipeline:
      1) Market factors - technical indicators (momentum, volatility, volume shocks, etc.)
      2) News VADER sentiment + weekly aggregation (T+1 principle)
      3) Merge (symbol,date) - MARKET_WIDE as fallback
      4) NEW: Compute market-wide sentiment index (macro sentiment)

    Time alignment explanation:
    - Both market data and news are resampled weekly (W-WED)
    - News uses T+1 principle: news of day t affects sentiment features of day t+1
    - MARKET_WIDE sentiment is broadcast to all assets without individual news
    - Macro market_sentiment_index is a weekly aggregate for regime signals
    """
    tic = time.time()
    logging.info("Station 2 – Starting full feature engineering …")

    # === 1) Market factors ===
    df_market_feat = stage2_feature_engineering_market(csv_path=market_csv)

    # === 2) News sentiment factors ===
    df_news_raw = pd.read_csv(news_csv, parse_dates=["date"])
    df_news_clean = stage2_clean_text_news(df_news_raw)
    df_news_sent = stage2_sentiment_scores(df_news_clean)

    # Debug: check keywords vs symbols matching
    market_symbols = set(df_market_feat["symbol"].str.upper().unique())
    debug_keywords_symbols(df_news_sent, market_symbols)

    # Weekly aggregation of sentiment (T+1 lag)
    df_news_weekly = stage2_aggregate_sentiment(df_news_sent, t_lag=1)

    # === 3) Merge micro-level (symbol-level) sentiment with market data ===
    df_final = stage2_merge_market_news(df_market_feat, df_news_weekly)

    # === ✅ 4) Compute MACRO market sentiment index ===
    df_market_sent_idx = compute_market_sentiment_index(df_news_weekly, smooth_window=3)

    # ✅ Merge macro sentiment into the feature matrix → every row same macro sentiment for that week
    df_final = df_final.merge(df_market_sent_idx, on="date", how="left")

    # 假设你想保存到 ~/Desktop/QuantEye/results
    from pathlib import Path

    # Step 1: 定义路径（如果你在 Mac 上）
    output_path = Path("~/Desktop/QuantEye/results/macro_sentiment_index.csv").expanduser()

    # Step 2: 保存 DataFrame 到 CSV
    df_market_sent_idx.to_csv(output_path, index=False)

    # Step 3: 打印确认
    logging.info(f"✅ 宏观情绪指数已保存到 {output_path}")
    # === Save output ===
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename
    df_final.to_csv(out_path, index=False)

    logging.info(
        "Station 2 – Feature matrix saved -> %s | shape=%s | runtime=%.2fs",
        out_path, df_final.shape, time.time() - tic
    )
    return df_final

