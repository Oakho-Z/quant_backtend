"""
Station 3 Runner: Institutional Portfolio Optimization Analysis
Output:
  ✅ 5 optimized core charts (institutional style)
  ✅ Core metrics, backtest results, latest recommended portfolios CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import timedelta
from pathlib import Path

from station3 import PortfolioOptimizer, PerformanceAnalyzer, Config

warnings.filterwarnings('ignore')

# ======== Unified matplotlib style ========
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")


# ============================
# Institutional Visualization Class
# ============================
class SeparatedVisualization:
    """Institutional version: with smart label positioning and clear backgrounds"""

    def __init__(self, save_dir="charts"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Keep default font, avoid SimHei issues
        plt.rcParams['axes.unicode_minus'] = False  # ensure minus sign shows

        # Institutional color palette
        self.colors = {
            'portfolio': '#1f77b4',  # Blue
            'benchmark': '#7f7f7f',  # Grey
            'positive': '#2ca02c',   # Green
            'negative': '#ff7f0e',   # Orange
        }

    def create_institutional_charts(self, backtest_df, recommendations, metrics):
        """Generate core 5 charts for institutional decision-making"""
        if backtest_df.empty:
            print("⚠️ No backtest data available")
            return

        print("🎨 Creating institutional decision charts...")

        self.plot_cumulative_returns(backtest_df, metrics)
        self.plot_drawdown_analysis(backtest_df)
        self.plot_return_distribution(backtest_df, metrics)
        self.plot_key_metrics_summary(metrics)
        self.plot_asset_frequency_and_weights(recommendations)
        self.plot_macro_sentiment_gauge()  # Macro sentiment gauge

        print(f"✅ Core institutional charts saved to '{self.save_dir}'")

    # === 1️⃣ Cumulative returns vs BTC ===
    def plot_cumulative_returns(self, backtest_df, metrics):
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(backtest_df['date'], backtest_df['cum_portfolio'],
                color=self.colors['portfolio'], linewidth=3,
                label=f'Portfolio (CAGR {metrics["portfolio_cagr"]:.1f}%)')
        ax.plot(backtest_df['date'], backtest_df['cum_btc'],
                color=self.colors['benchmark'], linewidth=2,
                label=f'BTC Benchmark (CAGR {metrics["btc_cagr"]:.1f}%)')

        # Highlight the point where portfolio first surpasses BTC
        diff = backtest_df['cum_portfolio'] - backtest_df['cum_btc']
        surpass_idx = diff[diff > 0].first_valid_index()
        if surpass_idx:
            date_surpass = backtest_df['date'][surpass_idx]
            y_val = backtest_df['cum_portfolio'].iloc[surpass_idx]

            # Vertical line marker
            ax.axvline(date_surpass, color='green', linestyle='--', alpha=0.6)

            # Shift label slightly right and upward
            shifted_date = date_surpass + pd.Timedelta(days=14)
            ax.text(
                shifted_date, y_val + max(2, y_val * 0.05),  # Upward offset
                'Surpass BTC',
                color='green', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )

        # Light fill under portfolio curve
        ax.fill_between(backtest_df['date'], backtest_df['cum_portfolio'],
                        alpha=0.15, color=self.colors['portfolio'])

        ax.set_title("Cumulative Returns vs BTC Benchmark", fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (x)')
        ax.legend(loc='upper left', frameon=False)

        # Bottom-right label: excess return ratio
        excess = backtest_df['cum_portfolio'].iloc[-1] / backtest_df['cum_btc'].iloc[-1]
        ax.text(
            0.98, 0.05, f"Excess ≈ {excess:.1f}x",
            transform=ax.transAxes, fontsize=12, ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/01_cumulative_returns.png", dpi=300, bbox_inches='tight')
        plt.show()
    # === 2️⃣ Drawdown analysis ===
    def plot_drawdown_analysis(self, backtest_df):
        fig, ax = plt.subplots(figsize=(12, 7))
        portfolio_dd = (backtest_df['cum_portfolio'] / backtest_df['cum_portfolio'].cummax() - 1) * 100
        btc_dd = (backtest_df['cum_btc'] / backtest_df['cum_btc'].cummax() - 1) * 100

        ax.fill_between(backtest_df['date'], portfolio_dd, 0,
                        alpha=0.6, color=self.colors['portfolio'], label='Portfolio Drawdown')
        ax.fill_between(backtest_df['date'], btc_dd, 0,
                        alpha=0.4, color=self.colors['benchmark'], label='BTC Drawdown')

        max_dd_port = portfolio_dd.min()
        max_dd_btc = btc_dd.min()

        # Mark the date of portfolio's maximum drawdown
        max_dd_port_idx = portfolio_dd.idxmin()
        max_dd_port_date = backtest_df['date'][max_dd_port_idx]
        y_val = portfolio_dd.iloc[max_dd_port_idx]

        ax.axvline(max_dd_port_date, color='red', linestyle='--', alpha=0.6)

        # Offset label slightly to the right and above
        shifted_date = max_dd_port_date + pd.Timedelta(days=14)
        ax.text(
            shifted_date, y_val + 2,
            f'Max DD {max_dd_port:.1f}%',
            color='red', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

        ax.set_title(
            f"Drawdown Analysis\nPortfolio MaxDD {max_dd_port:.1f}% | BTC {max_dd_btc:.1f}%",
            fontsize=16, fontweight='bold'
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/02_drawdown_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

    # === 3️⃣ Weekly return distribution ===
    def plot_return_distribution(self, backtest_df, metrics):
        fig, ax = plt.subplots(figsize=(12, 7))
        portfolio_ret = backtest_df['portfolio_ret'] * 100
        btc_ret = backtest_df['btc_ret'] * 100

        # Histogram comparison between portfolio and BTC
        ax.hist(portfolio_ret, bins=30, alpha=0.7, density=True,
                color=self.colors['positive'], label='Portfolio')
        ax.hist(btc_ret, bins=30, alpha=0.5, density=True,
                color=self.colors['benchmark'], label='BTC')

        # Mean return lines
        ax.axvline(portfolio_ret.mean(), color=self.colors['positive'], linestyle='--',
                   label=f'Portfolio Mean {portfolio_ret.mean():.2f}%')
        ax.axvline(btc_ret.mean(), color=self.colors['benchmark'], linestyle='--',
                   label=f'BTC Mean {btc_ret.mean():.2f}%')

        # Win rate label at the top-right corner
        win_rate = metrics["win_rate"] * 100
        ax.text(
            0.98, 0.95,
            f"Win Rate: {win_rate:.1f}%\nVol σ={portfolio_ret.std():.2f}%",
            transform=ax.transAxes, ha='right', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

        ax.set_title('Weekly Return Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Weekly Return (%)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/03_return_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()

    # === 4️⃣ Key metrics summary ===
    def plot_key_metrics_summary(self, metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        table_data = [
            ['Win Rate', f'{metrics["win_rate"]:.1%}', '-'],
            ['Outperform BTC', f'{metrics["outperform_rate"]:.1%}', '-'],
            ['Annual Return', f'{metrics["portfolio_cagr"]:.2%}', f'{metrics["btc_cagr"]:.2%}'],
            ['Volatility', f'{metrics["portfolio_vol"]:.2%}', f'{metrics["btc_vol"]:.2%}'],
            ['Sharpe Ratio', f'{metrics["portfolio_sharpe"]:.2f}', f'{metrics["btc_sharpe"]:.2f}'],
            ['Max Drawdown', f'{metrics["portfolio_mdd"]:.2%}', f'{metrics["btc_mdd"]:.2%}'],
            ['Information Ratio', f'{metrics["info_ratio"]:.2f}', '-'],
        ]
        table = ax.table(cellText=table_data,
                         colLabels=['Metric', 'Portfolio', 'BTC Benchmark'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax.set_title('Key Backtest Metrics', fontsize=16, fontweight='bold', pad=30)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/04_metrics_summary.png", dpi=300, bbox_inches='tight')
        plt.show()
    # === 5️⃣ Top assets frequency + weights ===
    def plot_asset_frequency_and_weights(self, recommendations):
        if not recommendations:
            return
        freq_map, weight_map = {}, {}
        for rec in recommendations.values():
            for a, w in zip(rec['assets'], rec['weights']):
                freq_map[a] = freq_map.get(a, 0) + 1
                weight_map.setdefault(a, []).append(w)
        avg_weight = {a: np.mean(w) for a, w in weight_map.items()}

        # Show only Top 5 + aggregate others
        sorted_assets = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_assets[:5]
        others_sum = sum([x[1] for x in sorted_assets[5:]])
        if others_sum > 0:
            top5.append(('Others', others_sum))

        assets = [x[0] for x in top5]
        freqs = [x[1] for x in top5]
        weights = [avg_weight.get(x, np.nan) for x in assets]

        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twiny()

        bars = ax1.barh(assets, freqs, color=self.colors['portfolio'], alpha=0.7)
        ax1.set_xlabel('Selection Count')
        ax1.set_ylabel('Asset')
        ax1.grid(alpha=0.3)

        ax2.plot(weights, assets, marker='o', color=self.colors['positive'], linewidth=2, label='Avg Weight')
        ax2.set_xlabel('Average Weight')

        # Calculate Top 3 concentration ratio
        top3_sum = sum([w for w in weights[:3] if not np.isnan(w)])
        ax1.set_title(f'Top5 Asset Frequency & Avg Weight\nTop3 Concentration ≈ {top3_sum:.1%}',
                      fontsize=15, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/05_top_assets_frequency_weights.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _cat(self, v: float) -> tuple[str, str]:
        if v < 20: return "Extreme Fear", "#8B0000"
        if v < 40: return "Fear", "#FF4500"
        if v < 60: return "Neutral", "#FFD700"
        if v < 80: return "Greed", "#90EE90"
        return "Extreme Greed", "#006400"

    def _fear_greed_gauge(self, df: pd.DataFrame, fname: str):
        import matplotlib.pyplot as plt
        import numpy as np

        recent = df[df["date"] >= df["date"].max() - timedelta(days=6)]
        avg = recent["market_sentiment_index"].mean() * 100

        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection="polar"))

        bounds = [0, 20, 40, 60, 80, 100]
        colors = ["#8B0000", "#FF4500", "#FFD700", "#90EE90", "#006400"]
        labels = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

        for i in range(5):
            theta = np.linspace(np.pi * bounds[i] / 100, np.pi * bounds[i + 1] / 100, 100)
            ax.fill_between(theta, 0.4, 1, color=colors[i], alpha=0.85)

        # Pointer
        angle = np.pi * avg / 100
        ax.plot([angle, angle], [0, 0.85], color="black", lw=3)
        ax.plot(angle, 0, "ko", ms=10)

        # Add central number
        ax.text(np.pi / 2, 0.1, f"{avg:.0f}", fontsize=32, ha="center", va="center", weight="bold")

        # Ticks
        for sc in [0, 25, 50, 75, 100]:
            ang = np.pi * sc / 100
            ax.text(ang, 1.05, f"{sc}", ha="center", va="center", fontsize=10, color="black")

        # Category label
        cat, col = self._cat(avg)
        ax.text(np.pi / 2, -0.1, f"Last 7-day Avg\nCurrent: {cat}",
                ha="center", va="center", fontsize=13, color=col, weight="bold")

        # Clean up chart style
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, np.pi)
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()

    def plot_macro_sentiment_gauge(self):
        """Plot a sentiment gauge based on market_sentiment_index from Stage2 data"""
        try:
            # Define portable path
            macro_path = Path("~/Desktop/QuantEye/results/macro_sentiment_index.csv").expanduser()

            # Read macro sentiment data
            df = pd.read_csv(macro_path)
            df["date"] = pd.to_datetime(df["date"])

            # Save the last 7 days of data
            recent_macro = df[df["date"] >= df["date"].max() - timedelta(days=6)]
            save_path = Path("~/Desktop/QuantEye/results/macro_sentiment_last7days.csv").expanduser()
            recent_macro.to_csv(save_path, index=False)

            print(f"✅ Saved: last 7-day macro sentiment to {save_path.resolve()}")

            # --- Use the previously defined _fear_greed_gauge function ---
            self._fear_greed_gauge(df, os.path.join(self.save_dir, "06_macro_fear_greed_gauge.png"))

            print("✅ Saved: 06_macro_fear_greed_gauge.png")

        except Exception as e:
            print(f"❌ Error plotting macro sentiment gauge: {e}")

# Runner (integrated save logic)
# ============================
class OptimizationRunner:
    """Portfolio Optimization Runner (Institutional Version)"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.optimizer = PortfolioOptimizer(self.config)
        self.analyzer = PerformanceAnalyzer()
        self.visualizer = SeparatedVisualization()

    def run_optimization_analysis(self, stage2_file):
        print("🚀 Starting Portfolio Optimization Analysis")
        print("=" * 60)

        print("📊 Loading Stage2 data...")
        try:
            df_stage2 = pd.read_csv(stage2_file)
            print(f"✅ Data loaded successfully: {df_stage2.shape}")
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return None, None, None

        print("\n🔄 Running portfolio optimization...")
        recommendations, backtest_df, opt_stats = self.optimizer.portfolio_optimization_pipeline(df_stage2)

        if not recommendations:
            print("❌ No recommendations generated")
            return None, None, None

        # Print the latest 3 recommended portfolios
        self._display_sample_recommendations(recommendations)

        # Performance metrics analysis
        if not backtest_df.empty:
            metrics = self.analyzer.analyze_performance(backtest_df, recommendations)
            self._display_performance_metrics(metrics)

            # ✅ Save core results
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(os.path.join(self.visualizer.save_dir, "metrics_summary.csv"), index=False)

            # Save the latest recommended portfolios
            last3 = list(recommendations.items())[-3:]
            latest_df = []
            for date, combo in last3:
                for asset, weight in zip(combo['assets'], combo['weights']):
                    latest_df.append({
                        "date": date,
                        "asset": asset,
                        "weight": weight,
                        "exp_ret": combo["exp_ret"],
                        "exp_vol": combo["exp_vol"],
                        "sharpe": combo["sharpe"]
                    })
            pd.DataFrame(latest_df).to_csv(os.path.join(self.visualizer.save_dir, "latest_recommendations.csv"),
                                           index=False)

            # Save complete backtest results
            backtest_df.to_csv(os.path.join(self.visualizer.save_dir, "backtest_results.csv"), index=False)

            # ✅ Generate only the charts required for institutional decision-making
            print("\n🎨 Creating institutional decision charts...")
            self.visualizer.create_institutional_charts(backtest_df, recommendations, metrics)
            self.save_chart_data_and_csv(backtest_df, metrics, recommendations, save_dir=self.visualizer.save_dir)
        else:
            print("\n⚠️ No backtest data available for performance analysis")
            metrics = {}

        return recommendations, backtest_df, metrics

    def save_chart_data_and_csv(self, backtest_df, metrics, recommendations, save_dir='charts'):
        os.makedirs(save_dir, exist_ok=True)

        # 1. Cumulative returns
        try:
            cols = ['date', 'cum_portfolio', 'cum_btc']
            if all(c in backtest_df.columns for c in cols):
                backtest_df[cols].to_csv(os.path.join(save_dir, 'chart_data_cumulative_returns.csv'), index=False)
                print("✅ Saved: chart_data_cumulative_returns.csv")
            else:
                print("⚠️ Skipped cumulative returns CSV – missing columns")
        except Exception as e:
            print("❌ Error saving cumulative return CSV:", e)

        # 2. Drawdown
        try:
            if 'cum_portfolio' in backtest_df.columns and 'cum_btc' in backtest_df.columns:
                dd_df = pd.DataFrame({
                    'date': backtest_df['date'],
                    'portfolio_drawdown': (
                                backtest_df['cum_portfolio'] / backtest_df['cum_portfolio'].cummax() - 1),
                    'btc_drawdown': (backtest_df['cum_btc'] / backtest_df['cum_btc'].cummax() - 1)
                })
                dd_df.to_csv(os.path.join(save_dir, 'chart_data_drawdown.csv'), index=False)
                print("✅ Saved: chart_data_drawdown.csv")
            else:
                print("⚠️ Skipped drawdown CSV – columns missing")
        except Exception as e:
            print("❌ Error saving drawdown CSV:", e)

        # 3. Metrics
        try:
            if metrics:
                pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False)
                print("✅ Saved: metrics_summary.csv")
        except Exception as e:
            print("❌ Error saving metrics:", e)

        # 4. Latest recommended portfolio
        try:
            if recommendations:
                latest_date, latest_combo = list(recommendations.items())[-1]
                latest_df = []
                for asset, weight in zip(latest_combo['assets'], latest_combo['weights']):
                    latest_df.append({
                        "date": latest_date,
                        "asset": asset,
                        "weight": weight,
                        "exp_ret": latest_combo["exp_ret"],
                        "exp_vol": latest_combo["exp_vol"],
                        "sharpe": latest_combo["sharpe"]
                    })
                pd.DataFrame(latest_df).to_csv(os.path.join(save_dir, "latest_recommendations.csv"), index=False)
                print("✅ Saved: latest_recommendations.csv")
        except Exception as e:
            print("❌ Error saving latest recommendation:", e)

    def _display_sample_recommendations(self, recommendations):
        print("\n✅ Latest Portfolio Recommendations:")
        for date, combo in list(recommendations.items())[-3:]:
            print(f"\n📅 {date} ({combo['status']}) - Assets: {combo['n_assets']}")
            print(
                f"   Expected Return: {combo['exp_ret']:.4f}, Volatility: {combo['exp_vol']:.4f}, Sharpe: {combo['sharpe']:.2f}")
            for asset, weight in zip(combo['assets'], combo['weights']):
                print(f"   {asset:<8} {weight:.3f}")

    def _display_performance_metrics(self, metrics):
        print(f"\n📈 Backtest Performance Metrics:")
        print(f"📊 Portfolio Performance:")
        print(f"   Annual Return: {metrics['portfolio_cagr']:.2%}")
        print(f"   Annual Volatility: {metrics['portfolio_vol']:.2%}")
        print(f"   Sharpe Ratio: {metrics['portfolio_sharpe']:.2f}")
        print(f"   Max Drawdown: {metrics['portfolio_mdd']:.2%}")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"\n📊 BTC Benchmark:")
        print(f"   Annual Return: {metrics['btc_cagr']:.2%}")
        print(f"   Annual Volatility: {metrics['btc_vol']:.2%}")
        print(f"   Sharpe Ratio: {metrics['btc_sharpe']:.2f}")
        print(f"   Max Drawdown: {metrics['btc_mdd']:.2%}")
        print(f"\n📊 Relative Performance:")
        print(f"   Excess Return: {metrics['portfolio_cagr'] - metrics['btc_cagr']:.2%}")
        print(f"   Outperform Rate: {metrics['outperform_rate']:.2%}")
        print(f"   Information Ratio: {metrics['info_ratio']:.2f}")

if __name__ == "__main__":
    stage2_file = "results/station2/station2_feature_matrix.csv"
    runner = OptimizationRunner()
    try:
        recommendations, backtest_df, metrics = runner.run_optimization_analysis(stage2_file)
        print("\n✅ Analysis completed! Core charts + CSV saved to 'charts/'")

    except FileNotFoundError:
        print(f"❌ File not found: {stage2_file}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
