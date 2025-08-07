"""
Station 3: Enhanced Portfolio Optimization Engine
"""
import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

# ================================
# Configuration Parameters Class
# ================================
class Config:
    """Portfolio optimization configuration parameters"""
    GAMMA = 1.5  # Risk aversion parameter
    TOP_N = 8    # Number of top candidate assets
    LOOKBACK_WEEKS = 4  # Historical data lookback window (in weeks)
    MIN_ASSETS = 2  # Minimum number of assets in the portfolio
    MAX_WEIGHT = 0.3  # Maximum weight for a single asset
    MIN_LIQUIDITY = 0.5  # Minimum liquidity requirement
    MAX_MDD = -0.9  # Maximum drawdown limit
    SHRINKAGE = 1e-3  # Covariance shrinkage parameter

class PortfolioOptimizer:
    """Main class for portfolio optimization"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.optimization_history = []

    def multi_factor_score(self, df):
        """
        Multi-factor scoring system
        Combines momentum, trading volume, volatility, and sentiment factors
        """
        # Base factor preprocessing
        momentum = df['momentum_28'].fillna(df['momentum_28'].median())
        volume = df['v_42d'].fillna(df['v_42d'].median())
        volatility = df['volatility_28'].fillna(df['volatility_28'].median())
        sentiment = df['avg_compound'].fillna(0)

        # Z-score normalization
        momentum_z = self._zscore_normalize(momentum)
        volume_z = self._zscore_normalize(volume)
        volatility_z = self._zscore_normalize(volatility)
        sentiment_z = self._zscore_normalize(sentiment)

        # Combined factor score
        base_score = (
            0.35 * momentum_z +      # Momentum factor weight 35%
            0.25 * volume_z -        # Volume factor weight 25%
            0.25 * volatility_z +    # Volatility factor weight -25% (negative means lower volatility is better)
            0.15 * sentiment_z       # Sentiment factor weight 15%
        )

        return base_score

    def _zscore_normalize(self, series):
        """Z-score normalization"""
        mean_val = series.mean()
        std_val = series.std()
        return (series - mean_val) / (std_val + 1e-8)

    def robust_covariance_estimation(self, returns_df, method='ledoit_wolf'):
        """
        Robust covariance estimation
        Supports multiple estimation methods
        """
        returns_clean = returns_df.fillna(0)

        if method == 'ledoit_wolf':
            try:
                lw = LedoitWolf()
                lw.fit(returns_clean.values)
                cov = lw.covariance_
            except:
                # Fallback: sample covariance
                cov = np.cov(returns_clean.T)
        elif method == 'sample':
            cov = np.cov(returns_clean.T)
        else:
            # Exponentially weighted covariance
            cov = returns_clean.ewm(span=20).cov().iloc[-len(returns_clean.columns):].values

        # Ensure positive semi-definiteness
        cov = (cov + cov.T) / 2
        eigenvals = np.linalg.eigvals(cov)
        if np.min(eigenvals) <= 0:
            cov += np.eye(cov.shape[0]) * (abs(np.min(eigenvals)) + self.config.SHRINKAGE)

        return cov

    def mean_variance_optimization(self, returns_df, max_weight=None, gamma=None):
        """
        Mean-variance optimization
        Uses multiple solvers to ensure robustness
        """
        if max_weight is None:
            max_weight = self.config.MAX_WEIGHT
        if gamma is None:
            gamma = self.config.GAMMA

        n = returns_df.shape[1]

        # Expected return preprocessing (Winsorize outliers)
        mu = returns_df.mean().values
        mu_q1, mu_q99 = np.percentile(mu, [1, 99])
        mu = np.clip(mu, mu_q1, mu_q99)

        # Covariance estimation
        cov = self.robust_covariance_estimation(returns_df)

        # Optimization variables
        w = cp.Variable(n)

        # Objective: maximize utility = expected return - γ/2 * risk
        portfolio_ret = mu @ w
        portfolio_risk = cp.quad_form(w, cov)
        objective = cp.Maximize(portfolio_ret - gamma * portfolio_risk)

        # Constraints
        constraints = [
            cp.sum(w) == 1,      # Weights sum to 1
            w >= 0,              # Long-only constraint
            w <= max_weight      # Maximum weight per asset
        ]

        # Try multiple solvers
        solvers = [cp.ECOS, cp.SCS, cp.OSQP]

        for solver in solvers:
            try:
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=solver, verbose=False)

                if w.value is not None and not np.any(np.isnan(w.value)) and prob.status == 'optimal':
                    weights = np.maximum(w.value, 0)
                    weights = weights / weights.sum()  # Normalize
                    return weights, mu, cov, 'optimal'
            except Exception as e:
                continue

        # All solvers failed, use fallback strategy
        print("⚠️ MVO optimization failed, using fallback strategy")
        fallback_weights = self._fallback_strategy(mu, cov)
        return fallback_weights, mu, cov, 'fallback'
    def _fallback_strategy(self, mu, cov):
        """
        Fallback strategy: combines risk parity, return-weighted, and equal-weighted allocations
        """
        n = len(mu)

        # Method 1: Risk parity
        vol = np.sqrt(np.diag(cov))
        risk_parity = (1.0 / (vol + 1e-8))
        risk_parity = risk_parity / risk_parity.sum()

        # Method 2: Return-weighted
        mu_positive = np.maximum(mu, 0)
        if mu_positive.sum() > 0:
            return_weighted = mu_positive / mu_positive.sum()
        else:
            return_weighted = np.ones(n) / n

        # Method 3: Equal-weighted
        equal_weight = np.ones(n) / n

        # Combined strategy
        weights = 0.4 * risk_parity + 0.4 * return_weighted + 0.2 * equal_weight
        return weights / weights.sum()

    def data_quality_check(self, df):
        """Data quality check and basic statistics"""
        print(f"📊 Data Quality Check:")
        print(f"   Raw data shape: {df.shape}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Number of assets: {df['symbol'].nunique()}")

        # Missing value statistics
        missing_cols = df.isnull().sum()
        if missing_cols.sum() > 0:
            print(f"   Major missing values: {missing_cols[missing_cols > 0].head()}")

        # Return statistics
        if 'return' in df.columns:
            ret_stats = df['return'].describe()
            print(f"   Return stats: mean={ret_stats['mean']:.4f}, std={ret_stats['std']:.4f}")

            # Extreme value statistics
            q1, q99 = df['return'].quantile([0.01, 0.99])
            extreme_count = ((df['return'] < q1) | (df['return'] > q99)).sum()
            print(f"   Number of extreme returns: {extreme_count} ({extreme_count/len(df)*100:.1f}%)")

        return df

    def portfolio_optimization_pipeline(self, stage2_df):
        """
        Main pipeline for portfolio optimization
        """
        df = stage2_df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Data quality check
        df = self.data_quality_check(df)

        # Asset filtering
        initial_count = len(df)
        df = df[
            (df['liquidity_score'] > self.config.MIN_LIQUIDITY) &
            (df['mdd_30d'] > self.config.MAX_MDD)
        ]
        print(f"📋 Retained after filtering: {len(df)}/{initial_count} ({len(df)/initial_count*100:.1f}%)")

        # Calculate multi-factor scores
        df['enhanced_score'] = self.multi_factor_score(df)

        # Group by week for optimization
        weekly_groups = df.groupby(pd.Grouper(key='date', freq='W'))

        all_recommendations = {}
        portfolio_dates, portfolio_returns, btc_returns = [], [], []
        optimization_stats = {'optimal': 0, 'fallback': 0, 'empty': 0}

        for week_date, group in weekly_groups:
            if group.empty:
                optimization_stats['empty'] += 1
                continue

            # Select top assets
            top_assets = group.sort_values('enhanced_score', ascending=False).head(self.config.TOP_N)['symbol'].unique()

            if len(top_assets) < self.config.MIN_ASSETS:
                print(f"⚠️ {week_date.date()} Not enough candidate assets ({len(top_assets)}) → Skipped")
                optimization_stats['empty'] += 1
                continue

            # Build historical return matrix
            start_window = week_date - pd.Timedelta(weeks=self.config.LOOKBACK_WEEKS)
            hist_df = df[
                (df['symbol'].isin(top_assets)) &
                (df['date'] >= start_window) &
                (df['date'] < week_date)
            ]

            if hist_df.empty:
                optimization_stats['empty'] += 1
                continue

            # Construct return matrix
            ret_matrix = hist_df.pivot(index='date', columns='symbol', values='return')

            # Data sufficiency check
            min_observations = max(3, self.config.LOOKBACK_WEEKS // 2)
            ret_matrix = ret_matrix.dropna(axis=1, thresh=min_observations)

            if ret_matrix.shape[1] < self.config.MIN_ASSETS or ret_matrix.shape[0] < min_observations:
                print(f"⚠️ {week_date.date()} Insufficient historical data → Skipped")
                optimization_stats['empty'] += 1
                continue

            # MVO optimization
            weights, mu, cov, status = self.mean_variance_optimization(ret_matrix)
            optimization_stats[status] += 1

            assets = list(ret_matrix.columns)

            # Compute portfolio metrics
            port_mu = mu @ weights
            port_sigma = np.sqrt(weights.T @ cov @ weights)
            sharpe = port_mu / port_sigma if port_sigma > 0 else 0

            # Save recommendation
            all_recommendations[str(week_date.date())] = {
                'assets': assets,
                'weights': list(np.round(weights, 4)),
                'exp_ret': round(port_mu, 4),
                'exp_vol': round(port_sigma, 4),
                'sharpe': round(sharpe, 2),
                'status': status,
                'n_assets': len(assets)
            }

            # Backtest calculation
            next_week_df = df[
                (df['date'] >= week_date) &
                (df['date'] < week_date + pd.Timedelta(days=7))
                ]

            if not next_week_df.empty:
                # Portfolio return
                next_week_ret_df = next_week_df[
                    next_week_df['symbol'].isin(assets)
                ].pivot(index='date', columns='symbol', values='return').fillna(0)

                if not next_week_ret_df.empty:
                    common_assets = [c for c in next_week_ret_df.columns if c in assets]
                    if len(common_assets) >= self.config.MIN_ASSETS:
                        idx_map = [assets.index(a) for a in common_assets]
                        aligned_weights = np.array([weights[i] for i in idx_map])
                        aligned_weights = aligned_weights / aligned_weights.sum()

                        next_week_ret = (next_week_ret_df[common_assets].values @ aligned_weights).mean()

                        # BTC benchmark
                        btc_df = next_week_df[next_week_df['symbol'] == 'BTC']
                        btc_week_ret = btc_df['return'].mean() if not btc_df.empty else np.nan

                        portfolio_dates.append(week_date)
                        portfolio_returns.append(next_week_ret)
                        btc_returns.append(btc_week_ret)

            # Optimization statistics
        total_weeks = sum(optimization_stats.values())
        if total_weeks > 0:
            print(f"\n📊 Optimization Statistics:")
            print(f"   Total weeks: {total_weeks}")
            print(
                f"   Optimal solutions: {optimization_stats['optimal']} ({optimization_stats['optimal'] / total_weeks * 100:.1f}%)")
            print(
                f"   Fallback strategies: {optimization_stats['fallback']} ({optimization_stats['fallback'] / total_weeks * 100:.1f}%)")
            print(
                f"   Insufficient data: {optimization_stats['empty']} ({optimization_stats['empty'] / total_weeks * 100:.1f}%)")

        # Construct backtest results
        backtest_df = pd.DataFrame({
            'date': portfolio_dates,
            'portfolio_ret': portfolio_returns,
            'btc_ret': btc_returns
        }).dropna()

        if not backtest_df.empty:
            backtest_df['cum_portfolio'] = (1 + backtest_df['portfolio_ret']).cumprod()
            backtest_df['cum_btc'] = (1 + backtest_df['btc_ret']).cumprod()

        return all_recommendations, backtest_df, optimization_stats


class PerformanceAnalyzer:
    """Performance analyzer"""

    @staticmethod
    def analyze_performance(backtest_df, recommendations):
        """Detailed performance analysis"""
        if backtest_df.empty:
            print("⚠️ Backtest data is empty")
            return {}

        # Basic metrics
        portfolio_ret = backtest_df['portfolio_ret']
        btc_ret = backtest_df['btc_ret']

        # Annualized return
        periods_per_year = 52  # Weekly frequency
        n_periods = len(portfolio_ret)

        portfolio_cagr = (backtest_df['cum_portfolio'].iloc[-1]) ** (periods_per_year / n_periods) - 1
        btc_cagr = (backtest_df['cum_btc'].iloc[-1]) ** (periods_per_year / n_periods) - 1

        # Volatility
        portfolio_vol = portfolio_ret.std() * np.sqrt(periods_per_year)
        btc_vol = btc_ret.std() * np.sqrt(periods_per_year)

        # Sharpe ratio
        portfolio_sharpe = portfolio_ret.mean() / portfolio_ret.std() * np.sqrt(periods_per_year)
        btc_sharpe = btc_ret.mean() / btc_ret.std() * np.sqrt(periods_per_year)

        # Maximum drawdown
        portfolio_mdd = (backtest_df['cum_portfolio'] / backtest_df['cum_portfolio'].cummax() - 1).min()
        btc_mdd = (backtest_df['cum_btc'] / backtest_df['cum_btc'].cummax() - 1).min()

        # Win rate
        win_rate = (portfolio_ret > 0).mean()
        outperform_rate = (portfolio_ret > btc_ret).mean()

        # Information ratio
        excess_ret = portfolio_ret - btc_ret
        info_ratio = excess_ret.mean() / excess_ret.std() * np.sqrt(periods_per_year) if excess_ret.std() > 0 else 0

        metrics = {
            'portfolio_cagr': portfolio_cagr,
            'btc_cagr': btc_cagr,
            'portfolio_vol': portfolio_vol,
            'btc_vol': btc_vol,
            'portfolio_sharpe': portfolio_sharpe,
            'btc_sharpe': btc_sharpe,
            'portfolio_mdd': portfolio_mdd,
            'btc_mdd': btc_mdd,
            'win_rate': win_rate,
            'outperform_rate': outperform_rate,
            'info_ratio': info_ratio,
            'total_periods': n_periods
        }

        return metrics


# Convenience functions
def run_portfolio_optimization(stage2_df, config=None):
    """Convenience function to run portfolio optimization"""
    optimizer = PortfolioOptimizer(config)
    return optimizer.portfolio_optimization_pipeline(stage2_df)


def analyze_portfolio_performance(backtest_df, recommendations):
    """Convenience function to analyze portfolio performance"""
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze_performance(backtest_df, recommendations)


if __name__ == "__main__":
    print("📈 Station 3: Enhanced Portfolio Optimization Engine loaded")
    print("Usage:")
    print("  from station3 import run_portfolio_optimization, analyze_portfolio_performance")
    print("  recommendations, backtest_df, stats = run_portfolio_optimization(your_data)")
    print("  metrics = analyze_portfolio_performance(backtest_df, recommendations)")


