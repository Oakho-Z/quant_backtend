"""
Sentiment Exploratory Analysis Runner
只做情绪 vs 收益相关性分析，不影响正式策略
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# ===== 币种情绪 vs 自己收益 =====
def analyze_coin_sentiment_vs_return(df_stage2):
    df_clean = df_stage2[['symbol', 'avg_compound', 'return']].dropna()
    overall_corr = df_clean['avg_compound'].corr(df_clean['return'])
    print(f"\n📊 Overall token sentiment vs return correlation: {overall_corr:.2f}")

    plt.figure(figsize=(8,6))
    sns.regplot(data=df_clean, x='avg_compound', y='return', scatter_kws={'alpha':0.3})
    plt.title(f"Coin-level Sentiment vs Weekly Return\nCorrelation={overall_corr:.2f}")
    plt.xlabel("Sentiment (avg_compound)")
    plt.ylabel("Weekly Return")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    scatter_path = Path("results/sentiment_analysis/coin_sentiment_vs_return_scatter.png")
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(scatter_path, dpi=300)
    plt.show()

    # 单个代表币种，比如 ETH
    coin = "ETH"
    df_coin = df_clean[df_clean['symbol'] == coin]
    if not df_coin.empty:
        coin_corr = df_coin['avg_compound'].corr(df_coin['return'])
        print(f"📊 {coin} sentiment vs return correlation: {coin_corr:.2f}")
        plt.figure(figsize=(8,6))
        sns.regplot(data=df_coin, x='avg_compound', y='return', scatter_kws={'alpha':0.5, 'color': 'blue'})
        plt.title(f"{coin} Sentiment vs Weekly Return\nCorrelation={coin_corr:.2f}")
        plt.xlabel("Sentiment (avg_compound)")
        plt.ylabel("Weekly Return")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        coin_path = Path(f"results/sentiment_analysis/{coin}_sentiment_vs_return.png")
        coin_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(coin_path, dpi=300)
        plt.show()



if __name__ == "__main__":
    # 1️⃣ 读取 Stage2 特征矩阵（币种因子+情绪）
    stage2_file = "results/station2/station2_feature_matrix.csv"
    df_stage2 = pd.read_csv(stage2_file, parse_dates=['date'])

    # 2️⃣ 读取 Stage3 回测结果（组合收益）
    backtest_file = "results/station3/backtest_results.csv"  # 你 Station3 Runner 输出的回测结果
    backtest_df = pd.read_csv(backtest_file, parse_dates=['date'])

    print("🔍 Running sentiment exploratory analysis...")

    # 币种情绪 vs 自己收益
    analyze_coin_sentiment_vs_return(df_stage2)



    print("\n✅ Sentiment exploratory analysis completed! Charts saved in 'charts/'")
    print("Backtest_df sample:")
    print(backtest_df.head())

    print("\nStage2 market sentiment sample:")
    print(df_stage2[['date', 'market_sentiment_index']].drop_duplicates().head(10))

    merged = backtest_df.merge(
        df_stage2[['date', 'market_sentiment_index']].drop_duplicates(),
        on='date', how='left'
    )

    print("\nMerged sample:")
    print(merged.head())
    print("Macro sentiment NaN count:", merged['market_sentiment_index'].isna().sum())
