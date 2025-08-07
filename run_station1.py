"""
Execute Stage 1 for BOTH:
    1. Crypto OHLCV pipeline (structured data)
    2. CoinDesk News pipeline (unstructured data)
Then run Stage 2:
    1. Compute market technical factors
    2. Compute news sentiment (VADER + T+1 aggregation)
    3. Merge into final feature matrix

Outputs:
  ./results/crypto/data
  ./results/news/data
  ./results/station2/station2_feature_matrix.csv
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import argparse

from Station1 import run_stage1   # Stage 1 pipeline
from Station1 import run_stage2   # Stage 2 feature engineering

# ======================================================
# Default parameters
# ======================================================

# ---- Crypto inputs ----
API_KEY_CRYPTO   = ""          # Your Crypto API key (if required)
PAGES            = [1, 2]         # Pages of top coins to fetch
TOP_LIMIT        = 100        # Coins per page
HISTORY_LIMIT    = 600         # Days of history
CURRENCY         = "USD"       # Quote currency

# ---- News inputs ----
API_KEY_NEWS     = None        # CoinDesk API still allows public calls
END_DT = datetime.today()
START_DT = END_DT - timedelta(days=180)

# ---- Pipeline control ----
SKIP_CRYPTO      = False       # If crypto data already downloaded, skip
SKIP_NEWS        = False       # If news data already downloaded, skip
SKIP_STAGE2      = False       # If only Stage1 needed, skip Stage2
# ======================================================


def check_existing_data(base_dir: Path):
    """Check for existing data files under base_dir"""
    crypto_file = base_dir / "crypto" / "data" / "stage_1_crypto_data.csv"
    news_file = base_dir / "news" / "data" / "stage_1_news_raw.csv"
    stage2_file = base_dir / "station2" / "station2_feature_matrix.csv"

    print("\n=== Checking existing data files ===")
    print(f"✓ Found crypto data: {crypto_file}" if crypto_file.exists() else "✗ No crypto data")
    print(f"✓ Found news data: {news_file}" if news_file.exists() else "✗ No news data")
    print(f"✓ Found Stage2 matrix: {stage2_file}" if stage2_file.exists() else "✗ No Stage2 feature matrix")
    print()

    return crypto_file.exists(), news_file.exists(), stage2_file.exists()


def main():
    parser = argparse.ArgumentParser(description="Combined Pipeline (Crypto + News + Stage2)")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Optional base directory for outputs (default=./results)")
    args = parser.parse_args()

    # Detect project root (where this script is located)
    PROJECT_ROOT = Path(__file__).resolve().parent

    # Determine base_dir (either user-specified or default to ./results)
    BASE_DIR = Path(args.base_dir).resolve() if args.base_dir else PROJECT_ROOT / "results"
    BASE_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Running COMBINED Pipeline, output base: {BASE_DIR}")
    print("=" * 60)

    # Check existing files
    has_crypto, has_news, has_stage2 = check_existing_data(BASE_DIR)

    # Ask user whether to use existing data
    if has_crypto and not SKIP_CRYPTO:
        response = input("Crypto data already exists. Skip download? (y/n): ")
        SKIP_CRYPTO_LOCAL = (response.lower() == 'y')
    else:
        SKIP_CRYPTO_LOCAL = SKIP_CRYPTO

    # -----------------------
    # 1. Crypto Stage 1
    # -----------------------
    if not SKIP_CRYPTO_LOCAL:
        print("\n>>> Starting Crypto Stage 1 ETL …")
        try:
            run_stage1(
                mode="crypto",
                api_key=API_KEY_CRYPTO,
                base_dir=BASE_DIR,
                pages=PAGES,
                top_limit=TOP_LIMIT,
                history_limit=HISTORY_LIMIT,
                currency=CURRENCY
            )
            print(f"✓ Crypto data saved under {BASE_DIR / 'crypto' / 'data'}\n")
        except Exception as e:
            print(f"❌ Crypto Stage 1 failed: {e}")
            if not has_crypto:
                print("No existing crypto data available, cannot continue.")
                sys.exit(1)
            else:
                print("Using existing crypto data to continue.")
    else:
        print("\n>>> Skipping Crypto Stage 1 (using existing data)")

    # -----------------------
    # 2. News Stage 1
    # -----------------------
    if not SKIP_NEWS:
        print("\n>>> Starting News Stage 1 ETL …")
        try:
            run_stage1(
                mode="news",
                api_key=API_KEY_NEWS,
                base_dir=BASE_DIR,
                start_dt=START_DT,
                end_dt=END_DT
            )
            print(f"✓ News data saved under {BASE_DIR / 'news' / 'data'}\n")
        except Exception as e:
            print(f"❌ News Stage 1 failed: {e}")
            if not has_news:
                print("No existing news data available, cannot continue.")
                sys.exit(1)
            else:
                print("Using existing news data to continue.")
    else:
        print("\n>>> Skipping News Stage 1 (using existing data)")

    print("=" * 60)
    print("✅ Stage 1 pipelines completed!")
    print("=" * 60)

    # =================================================
    # Now run Station 2 (merge market + news features)
    # =================================================
    if not SKIP_STAGE2:
        print("\n>>> Starting Station 2 Feature Engineering …")

        # Stage 1 output files
        market_csv = BASE_DIR / "crypto" / "data" / "stage_1_crypto_data.csv"
        news_csv   = BASE_DIR / "news" / "data" / "stage_1_news_raw.csv"
        out_dir    = BASE_DIR / "station2"

        # Ensure both input files exist
        if not market_csv.exists():
            print(f"❌ Market data file not found: {market_csv}")
            sys.exit(1)

        if not news_csv.exists():
            print(f"❌ News data file not found: {news_csv}")
            sys.exit(1)

        try:
            run_stage2(
                market_csv=market_csv,
                news_csv=news_csv,
                out_dir=out_dir
            )
            print(f"\n✓ Station 2 feature matrix saved under {out_dir}")
        except Exception as e:
            print(f"❌ Stage 2 failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n>>> Skipping Stage 2")

    print("=" * 60)
    print("✅ All pipelines completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()