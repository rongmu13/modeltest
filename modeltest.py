# -*- coding: utf-8 -*-
# 🌾 LET'S AI・収量予測ツール（回帰専用・日本語UI）
# 機能：Excel/CSVアップロード → 目的変数/説明変数の選択 → モデル選択 → 指標(R²/RMSE/MAE/調整R²)
#       → 可視化（予測 vs 実測、残差、単回帰の曲線）→ 結果ダウンロード

import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="モデルテスト", layout="wide")
st.title("モデルテスト")
st.caption("Excel/CSV をアップロード → 目的変数/説明変数を選択 → モデルを実行 → R²/RMSE/MAE を確認・可視化・ダウンロード")

# ------------------ サイドバー：設定 ------------------
with st.sidebar:
    st.header("⚙️ 設定")
    standardize = st.checkbox("数値特徴量を標準化（推奨）", value=True)
    show_adj_r2 = st.checkbox("調整R²（Adjusted R²）を表示", value=True)
    show_residuals = st.checkbox("残差プロットを表示", value=False)

# ------------------ メイン：データアップロード ------------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("📤 1) データのアップロード")
    up = st.file_uploader("Excel（.xlsx/.xls）または CSV を選択", type=["xlsx", "xls", "csv"])
    df = None
    sheet_name_sel = None
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                xls = pd.ExcelFile(up)
                if len(xls.sheet_names) > 1:
                    sheet_name_sel = st.selectbox("シートを選択", xls.sheet_names)
                df = pd.read_excel(up, sheet_name=sheet_name_sel or 0, engine="openpyxl")
            st.success(f"読み込み成功：{df.shape[0]} 行 × {df.shape[1]} 列")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"読み込み失敗：{e}")

with right:
    st.subheader("🧭 2) 列の選択 & 方法")
    if df is None:
        st.info("先にデータをアップロードしてください。")
    else:
        # 数値列のみ候補に
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 0:
            st.error("数値列が検出されませんでした。ファイル内容をご確認ください。")
            st.stop()

        target_col = st.selectbox("目的変数", num_cols, index=0)
        feature_candidates = [c for c in num_cols if c != target_col]
        default_X = feature_candidates[:min(8, len(feature_candidates))]
        feature_cols = st.multiselect("説明変数（複数選択可）", feature_candidates, default=default_X)

        st.markdown("**回帰モデルを選択**（分類モデルは含みません）：")
        method = st.selectbox(
            "モデル",
            [
                "線形回帰（Linear Regression）",
                "リッジ回帰（Ridge）",
                "ラッソ回帰（Lasso）",
                "ElasticNet",
                "PLS回帰（Partial Least Squares）",
                "多項式回帰（単変量）",
                "指数回帰（単変量）",
            ],
            index=0,
        )

        # パラメータ
        st.subheader("🧪 3) モデルパラメータ")
        params = {}
        if "Ridge" in method:
            params["alpha"] = st.number_input("alpha（Ridge）", min_value=0.0, value=1.0, step=0.1)
        elif "Lasso" in method:
            params["alpha"] = st.number_input("alpha（Lasso）", min_value=0.0, value=0.001, step=0.001, format="%.4f")
            params["max_iter"] = st.number_input("max_iter", min_value=1000, value=20000, step=1000)
        elif "ElasticNet" in method:
            params["alpha"] = st.number_input("alpha", min_value=0.0, value=0.01, step=0.01)
            params["l1_ratio"] = st.slider("l1_ratio（0=Ridge, 1=Lasso）", 0.0, 1.0, 0.5, 0.05)
            params["max_iter"] = st.number_input("max_iter", min_value=1000, value=20000, step=1000)
        elif "PLS" in method:
            max_comp = max(1, min(len(feature_cols), df.shape[0]-1))
            params["n_components"] = st.number_input("成分数（n_components）", min_value=1, max_value=max_comp, value=min(3, max_comp), step=1)
        elif "多項式回帰" in method:
            if len(feature_cols) != 1:
                st.warning("多項式回帰は**説明変数が1列**のときのみ有効です。1列だけ選択してください。")
            params["degree"] = st.slider("多項式の次数", 1, 6, 2)
        elif "指数回帰" in method:
            if len(feature_cols) != 1:
                st.warning("指数回帰は**説明変数が1列**のときのみ有効です。1列だけ選択してください。")

        st.divider()
        run = st.button("🚀 4) モデルを実行", use_container_width=True)

# ------------------ 実行：学習・可視化・出力 ------------------
if up is not None and df is not None and run:
    if target_col is None or len(feature_cols) == 0:
        st.error("目的変数 y と、少なくとも1つの説明変数 X を選択してください。")
        st.stop()

    y = df[target_col].astype(float).values
    X = df[feature_cols].astype(float).values
    n, p = X.shape

    # 標準化（推奨）
    scaler = StandardScaler() if standardize else None
    Xs = scaler.fit_transform(X) if scaler is not None else X.copy()

    # 指標関数
    def calc_metrics(y_true, y_pred, p_features):
        r2 = r2_score(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        if show_adj_r2:
            adj_r2 = 1.0 - (1.0 - r2) * ((len(y_true) - 1) / max(1, len(y_true) - p_features - 1))
        else:
            adj_r2 = None
        return r2, rmse, mae, adj_r2

    # 学習 & 予測
    y_pred = None
    model_name = method

    if method.startswith("線形回帰"):
        mdl = LinearRegression()
        mdl.fit(Xs, y)
        y_pred = mdl.predict(Xs)

    elif method.startswith("リッジ回帰"):
        mdl = Ridge(alpha=float(params["alpha"]))
        mdl.fit(Xs, y)
        y_pred = mdl.predict(Xs)

    elif method.startswith("ラッソ回帰"):
        mdl = Lasso(alpha=float(params["alpha"]), max_iter=int(params["max_iter"]))
        mdl.fit(Xs, y)
        y_pred = mdl.predict(Xs)

    elif method.startswith("ElasticNet"):
        mdl = ElasticNet(alpha=float(params["alpha"]), l1_ratio=float(params["l1_ratio"]), max_iter=int(params["max_iter"]))
        mdl.fit(Xs, y)
        y_pred = mdl.predict(Xs)

    elif method.startswith("PLS回帰"):
        n_comp = int(params["n_components"])
        mdl = PLSRegression(n_components=n_comp)
        mdl.fit(Xs, y)
        y_pred = mdl.predict(Xs).ravel()

    elif method.startswith("多項式回帰"):
        if len(feature_cols) != 1:
            st.error("多項式回帰は**説明変数が1列**のときのみ実行可能です。")
            st.stop()
        degree = int(params["degree"])
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        Xp = poly.fit_transform(Xs)  # 標準化後に多項式展開
        mdl = LinearRegression()
        mdl.fit(Xp, y)
        y_pred = mdl.predict(Xp)

    elif method.startswith("指数回帰"):
        if len(feature_cols) != 1:
            st.error("指数回帰は**説明変数が1列**のときのみ実行可能です。")
            st.stop()
        if np.any(y <= 0):
            st.error("指数回帰は y>0 が必要です（対数を取るため）。データをご確認ください。")
            st.stop()
        y_log = np.log(y)
        mdl = LinearRegression()
        mdl.fit(Xs, y_log)
        y_log_pred = mdl.predict(Xs)
        y_pred = np.exp(y_log_pred)
        model_name = "指数回帰（y = a * exp(bx)）"

    else:
        st.error("未知のモデルです。")
        st.stop()

    # 指標表示
    # 多項式の自由度カウントは目安（次数+1）にする
    p_for_adj = p if "多項式回帰" not in method else min(X.shape[0]-1, int(params["degree"])+1)
    r2, rmse, mae, adj_r2 = calc_metrics(y, y_pred, p_for_adj)

    st.success("✅ モデル実行が完了しました")
    st.subheader("📊 モデル指標")
    met_cols = st.columns(4 if show_adj_r2 else 3)
    met_cols[0].metric("決定係数 R²", f"{r2:.4f}")
    met_cols[1].metric("RMSE", f"{rmse:.4f}")
    met_cols[2].metric("MAE", f"{mae:.4f}")
    if show_adj_r2:
        met_cols[3].metric("調整R²", f"{adj_r2:.4f}")

    # 予測結果テーブル
    out_df = pd.DataFrame({
        "実測値": y,
        "予測値": y_pred,
        "残差": y - y_pred
    })
    st.dataframe(out_df.head(20), use_container_width=True)

    # 可視化：予測 vs 実測
    st.subheader("📈 予測値と実測値の比較")
    viz_df = pd.DataFrame({"実測値": y, "予測値": y_pred})
    st.scatter_chart(viz_df, x="実測値", y="予測値", height=420)

    # 残差プロット（任意）
    if show_residuals:
        st.subheader("🪵 残差プロット（実測 − 予測）")
        res_df = pd.DataFrame({"Index": np.arange(len(y)), "残差": y - y_pred})
        st.line_chart(res_df, x="Index", y="残差", height=260)

    # 単変量の曲線可視化（多項式/指数）
    if (method.startswith("多項式回帰") or method.startswith("指数回帰")) and len(feature_cols) == 1:
        st.subheader("📉 単変量フィッティング（散布図＋曲線）")
        x_raw = df[feature_cols[0]].astype(float).values.reshape(-1, 1)
        x_grid = np.linspace(x_raw.min(), x_raw.max(), 200).reshape(-1, 1)

        # 標準化
        if standardize:
            scaler_x = StandardScaler().fit(x_raw)
            xr = scaler_x.transform(x_raw)
            xg = scaler_x.transform(x_grid)
        else:
            xr = x_raw
            xg = x_grid

        if method.startswith("多項式回帰"):
            degree = int(params["degree"])
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            mdl_curve = LinearRegression().fit(poly.fit_transform(xr), y)
            yg = mdl_curve.predict(poly.transform(xg))
        else:
            # 指数回帰：log(y) を線形近似してから逆変換
            if np.any(y <= 0):
                st.warning("指数回帰の可視化をスキップ（y>0 条件を満たしていません）。")
                yg = None
            else:
                y_log = np.log(y)
                mdl_curve = LinearRegression().fit(xr, y_log)
                yg = np.exp(mdl_curve.predict(xg))

        # 散布図
        curve_df = pd.DataFrame({
            "x": x_raw.ravel(),
            "y": y
        })
        st.scatter_chart(curve_df, x="x", y="y", height=420)

        # フィット曲線
        if yg is not None:
            line_df = pd.DataFrame({"x": x_grid.ravel(), "fit": yg})
            st.line_chart(line_df, x="x", y="fit", height=420)

    # ダウンロード：結果Excel
    st.subheader("⬇️ 結果をダウンロード")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export_df = pd.concat([df.reset_index(drop=True), out_df], axis=1)
        export_df.to_excel(writer, index=False, sheet_name="predictions")
        pd.DataFrame([{
            "model": model_name,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            **({"Adjusted_R2": adj_r2} if show_adj_r2 else {})
        }]).to_excel(writer, index=False, sheet_name="metrics")
    st.download_button(
        "Excel をダウンロード（元データ + 予測 + 指標）",
        data=buf.getvalue(),
        file_name="yield_regression_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    with st.expander("🔍 実行情報"):
        st.json({
            "model": model_name,
            "n_samples": int(n),
            "n_features": int(p),
            "standardized": standardize,
            "params": params
        })


