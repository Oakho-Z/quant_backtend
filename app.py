from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from run_station1 import main as run_station1_main
from run_station3 import OptimizationRunner, Config
from Station1 import run_stage1, run_stage2
from pathlib import Path
import json
import random
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Process, Queue
app = Flask(__name__)
CORS(app)

def generate_random(min_val, max_val):
    return random.uniform(min_val, max_val)

@app.route('/drawdown_data', methods=['GET'])
def drawdown_data():
    df = pd.read_csv('./results/station3/chart_data_drawdown.csv')
    data = []
    for i in range(len(df)):
        # 先转datetime，再转格式化字符串
        date_str = datetime.strptime(df.iloc[i]['date'], '%Y-%m-%d').strftime('%Y-%m-%d')
        dd = df.iloc[i]['portfolio_drawdown']
        data.append({"date": date_str, "drawdown": dd * 100})
    return jsonify({"drawdown": data})

@app.route('/performance_data', methods=['GET'])
def performance_data():
    # 读取真实的累计收益数据
    df = pd.read_csv('./results/station3/chart_data_cumulative_returns.csv')

    # 构造累计收益列表
    cumulativeReturns = []
    for _, row in df.iterrows():
        cumulativeReturns.append({
            "date": row["date"],
            "portfolio": row["cum_portfolio"],
            "btc": row["cum_btc"]
        })

    return jsonify({
        "cumulativeReturns": cumulativeReturns,
    })

@app.route('/weekly_market_mood_gauge_data', methods=['GET'])
def weekly_market_mood_gauge_data():
    # 读取真实的累计收益数据
    df = pd.read_csv('./results/macro_sentiment_last7days.csv')

    return jsonify({
        "value": df["market_sentiment_index"].values[0],
    })


@app.route('/performance_tab_chart_data', methods=['GET'])
def performance_tab_chart_data():
    try:
        portfolioMetrics = pd.read_csv("./results/station3/metrics_summary.csv")
        row = portfolioMetrics.iloc[0]

        recommendations_df = pd.read_csv("./results/station3/latest_recommendations.csv")

        # 构造推荐资产列表
        recommendations = []
        for _, rec_row in recommendations_df.iterrows():
            recommendations.append({
                "asset": rec_row["asset"],
                "date": rec_row["date"],
                "weight": round(rec_row["weight"] * 100, 2),  # 如果 weight 是 0~1 的小数
                "exp_ret": round(rec_row["exp_ret"], 4),
                "exp_vol": round(rec_row["exp_vol"], 4),
                "sharpe": round(rec_row["sharpe"], 2)
            })

        data = {
            "portfolioMetrics": {
                "portfolio_cagr": round(row["portfolio_cagr"], 2),
                "btc_cagr": round(row["btc_cagr"], 2),
                "portfolio_sharpe": round(row["portfolio_sharpe"], 2),
                "btc_sharpe": round(row["btc_sharpe"], 2),
                "portfolio_mdd": round(row["portfolio_mdd"], 2),
                "btc_mdd": round(row["btc_mdd"], 2),
                "win_rate": round(row["win_rate"] * 100, 1),  # 如果是 0~1 的小数
                "outperform_rate": round(row["outperform_rate"] * 100, 1),
            },
            "recommendations": recommendations
        }

        response = make_response(jsonify(data))
        response.headers['Cache-Control'] = 'no-store'
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def long_task(gamma, top_n, result_queue):
    run_stage1(
        api_key=None,
        mode='crypto',
        base_dir=Path("./results"),
        pages=[1, 2],
        top_limit=100,
        history_limit=600,
        currency="USD"
    )

    run_stage2(
        market_csv=Path("./results/crypto/data/stage_1_crypto_data.csv"),
        news_csv=Path("./results/news/data/stage_1_news_raw.csv"),
        out_dir=Path("./results/station2")
    )

    config = Config()
    config.GAMMA = gamma
    config.TOP_N = top_n

    runner = OptimizationRunner(config)
    recs, backtest_df, metrics = runner.run_optimization_analysis(
        Path("./results/station2/station2_feature_matrix.csv")
    )

    # 可有可无，主线程不再取数据了
    result_queue.put("done")


@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    data = request.get_json(force=True)
    gamma = data.get("gamma", 1.5)
    top_n = data.get("top_n", 8)

    result_queue = Queue()
    p = Process(target=long_task, args=(gamma, top_n, result_queue))
    p.start()
    p.join()

    return jsonify({"status": "success"})

@app.route("/chart_data", methods=["GET"])
def chart_data():
    try:
        cum_path = Path("./charts/cumulative_returns.json")
        dd_path = Path("./charts/drawdown.json")

        cumulative = json.load(open(cum_path)) if cum_path.exists() else []
        drawdown = json.load(open(dd_path)) if dd_path.exists() else []

        return jsonify({
            "cumulative_returns": cumulative,
            "drawdown": drawdown
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5053, debug=True)
