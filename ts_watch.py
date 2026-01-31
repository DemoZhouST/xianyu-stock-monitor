# -*- coding: utf-8 -*-
"""
ts_watch.py（TuShare：港股 + A股 | JSON watchlist + 无状态穿越触发推送）

✅ 你只维护 watchlist_cn.json：
- 每只票：代码 s、成本 c、profit/loss 阈值(整数百分比)、enabled
- enabled=true 才监控

✅ 推送逻辑（不存状态、不吵）：
- 用 TuShare 日线的 close / pre_close 计算 今日盈亏 / 昨日盈亏
- 只有出现以下情况才推送：
  1) 昨天未触达阈值，今天触达（阈值穿越）
  2) 同方向升级档位（亏损1档->2档 / 盈利1档->2档）
  3) 方向切换（profit <-> loss）
- 其余情况不推

✅ 数据抓取（只用 TuShare，每股只调 1 次）：
- 港股：pro.hk_daily
- A股：pro.daily
- 近20周周线：日线本地 resample("W-FRI")
"""

import os
import json
import time
import re
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import tushare as ts
import requests


# =========================================================
# 0) Key：本地方便测试（环境变量会覆盖）
# =========================================================
TUSHARE_TOKEN_LOCAL = "206afa7301d014b0b970a1f2319307b1464d4976181dd87"
SERVERCHAN_SENDKEY_LOCAL = "Tfq6wgUsMXOhzn6EiC7uyRHI8"

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "").strip() or TUSHARE_TOKEN_LOCAL
SERVERCHAN_SENDKEY = os.getenv("SERVERCHAN_SENDKEY", "").strip() or SERVERCHAN_SENDKEY_LOCAL

WATCHLIST_PATH = os.getenv("WATCHLIST_PATH", "watchlist_cn.json")

# TuShare 限速：建议每股睡一会（原来 32~38s 非常稳，但会慢；可按需要调小）
SLEEP_SEC_MIN = float(os.getenv("TS_SLEEP_MIN", "32"))
SLEEP_SEC_MAX = float(os.getenv("TS_SLEEP_MAX", "38"))

# 默认不发每日简报，只发触发提醒
PUSH_DAILY_SUMMARY = os.getenv("PUSH_DAILY_SUMMARY", "0").strip() == "1"
PUSH_IF_NO_PREV = os.getenv("PUSH_IF_NO_PREV", "0").strip() == "1"


# =========================================================
# 1) Server酱推送
# =========================================================
def push_serverchan(title: str, content: str):
    if not SERVERCHAN_SENDKEY or SERVERCHAN_SENDKEY.startswith("你的"):
        print("未配置 SERVERCHAN_SENDKEY，跳过推送。")
        return
    url = f"https://sctapi.ftqq.com/{SERVERCHAN_SENDKEY}.send"
    data = {"title": title, "desp": content}
    try:
        resp = requests.post(url, data=data, timeout=10)
        print("Server酱推送结果：", resp.text[:200])
    except Exception as e:
        print("Server酱推送失败：", e)


# =========================================================
# 2) 代码规范化 & 读取 watchlist
# =========================================================
def _infer_cn_suffix(code6: str) -> str:
    """6位数字推断A股后缀：上交所(.SH) or 深交所(.SZ)"""
    assert code6.isdigit() and len(code6) == 6
    if code6.startswith(("60", "61", "68", "603", "605")) or code6[0] == "6":
        return code6 + ".SH"
    else:
        return code6 + ".SZ"

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    if s.endswith((".HK", ".SH", ".SZ")):
        core = s[:-3]
        if core.isdigit():
            if s.endswith(".HK"):
                return core.zfill(5) + ".HK"
            else:
                return core.zfill(6) + s[-3:]
        return s

    if s.isdigit():
        if len(s) == 5:
            return s.zfill(5) + ".HK"
        if len(s) == 6:
            return _infer_cn_suffix(s.zfill(6))

    if re.fullmatch(r"\d{5}", s):
        return s + ".HK"
    if re.fullmatch(r"\d{6}", s):
        return _infer_cn_suffix(s)

    return s

def _pct_list_to_ratio(pcts: List[Any]) -> List[float]:
    out = []
    for x in (pcts or []):
        try:
            out.append(float(x) / 100.0)
        except Exception:
            continue
    return out

def load_watchlist(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_watch_cfg() -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    raw = load_watchlist(WATCHLIST_PATH)
    items = raw.get("items", [])
    if not isinstance(items, list) or not items:
        raise RuntimeError("watchlist_cn.json 格式不对：需要 items 数组且不能为空。")

    watch_cfg: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        sym = normalize_symbol(it.get("s", ""))
        if not sym or not sym.endswith((".HK", ".SH", ".SZ")):
            continue

        enabled = bool(it.get("enabled", True))

        cost = it.get("c", None)
        try:
            cost = float(cost) if cost is not None else None
        except Exception:
            cost = None

        profit_levels = sorted(_pct_list_to_ratio(it.get("profit", [])))
        loss_levels = _pct_list_to_ratio(it.get("loss", []))
        # loss 应该是负数，越亏越小；我们后面会 reverse 排序处理“从轻亏到深亏升级”
        loss_levels = sorted(loss_levels)

        watch_cfg[sym] = {
            "enabled": enabled,
            "cost": cost,
            "profit_levels": profit_levels,
            "loss_levels": loss_levels,
        }

    symbols = [s for s, cfg in watch_cfg.items() if cfg.get("enabled")]
    if not symbols:
        raise RuntimeError("watchlist_cn.json 里没有任何 enabled=true 的标的。")
    return watch_cfg, symbols


# =========================================================
# 3) TuShare 初始化
# =========================================================
if not TUSHARE_TOKEN or TUSHARE_TOKEN.startswith("你的"):
    raise ValueError("未配置 TuShare Token：请设置环境变量 TUSHARE_TOKEN 或填入 TUSHARE_TOKEN_LOCAL。")

pro = ts.pro_api(TUSHARE_TOKEN)


# =========================================================
# 4) 抓取：每股只调 1 次（日线），并取 close + pre_close + 周线
# =========================================================
def _fetch_daily_any(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    自动路由：
      - 港股：pro.hk_daily
      - A股： pro.daily
    尽量统一字段：trade_date, open, high, low, close, pre_close, vol, amount
    """
    if ts_code.endswith(".HK"):
        df = pro.hk_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    else:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    if df is None or df.empty:
        return pd.DataFrame()

    # 统一
    if "trade_date" not in df.columns:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # TuShare A股 daily 有 pre_close；港股 hk_daily 有时也有 pre_close（若没有就会是 NA）
    for c in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA

    return df[["trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"]]

def get_weekly20_and_quotes(ts_code: str, back_weeks: int = 160):
    """
    一次接口（日线）拿齐：
      - 最新收盘价 close
      - 昨收 pre_close（用于穿越触发）
      - 近20周周线 = 日线 resample W-FRI
      - last_trade_date
    """
    end_date = datetime.today().strftime("%Y%m%d")
    start_date = (datetime.today() - timedelta(weeks=back_weeks)).strftime("%Y%m%d")

    daily = _fetch_daily_any(ts_code, start_date, end_date)
    if daily.empty:
        return pd.DataFrame(), None, None, None

    last_close = float(daily["close"].iloc[-1])
    last_pre_close = daily["pre_close"].iloc[-1]
    last_pre_close = float(last_pre_close) if pd.notna(last_pre_close) else None
    last_trade_date = daily["trade_date"].iloc[-1].date()

    idx = daily.set_index("trade_date")
    wk = pd.DataFrame({
        "open":   idx["open"].resample("W-FRI").first(),
        "high":   idx["high"].resample("W-FRI").max(),
        "low":    idx["low"].resample("W-FRI").min(),
        "close":  idx["close"].resample("W-FRI").last(),
        "volume": idx["vol"].resample("W-FRI").sum(min_count=1),
    }).dropna(subset=["open", "high", "low", "close"], how="any").reset_index().rename(columns={"trade_date": "date"})
    wk.insert(0, "symbol", ts_code)
    weekly_20 = wk.tail(20).reset_index(drop=True)

    return weekly_20, last_close, last_pre_close, last_trade_date

def fetch_all_symbols(symbols_list: List[str]):
    weekly_dict: Dict[str, pd.DataFrame] = {}
    latest_rows = []
    for sym in symbols_list:
        ts_code = normalize_symbol(sym)
        print(f"抓取（周线20 + 最新收盘 + 昨收）：{ts_code}")
        wk20, last_px, pre_close, last_dt = get_weekly20_and_quotes(ts_code)
        weekly_dict[ts_code] = wk20
        latest_rows.append({"symbol": ts_code, "price": last_px, "pre_close": pre_close, "trade_date": last_dt})

        time.sleep(random.uniform(SLEEP_SEC_MIN, SLEEP_SEC_MAX))

    latest_df = pd.DataFrame(latest_rows, columns=["symbol", "price", "pre_close", "trade_date"])

    parts = []
    for sym, df in weekly_dict.items():
        if df is not None and not df.empty:
            parts.append(df.set_index("date")["close"].rename(sym))
    wide_df = pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()

    return weekly_dict, latest_df, wide_df


# =========================================================
# 5) 周线分析逻辑（沿用你原来的）
# =========================================================
def _safe_last_ma(series: pd.Series, window: int):
    if series is None or series.empty or len(series) < window:
        return None
    return series.rolling(window).mean().iloc[-1]

def classify_trend(close_series: pd.Series, current_price: float) -> dict:
    s = close_series.dropna().astype(float)
    ma4 = _safe_last_ma(s, 4)
    ma8 = _safe_last_ma(s, 8)
    ma16 = _safe_last_ma(s, 16)
    if ma4 is None or ma8 is None or ma16 is None:
        return {"status": "数据不足", "ma4": ma4, "ma8": ma8, "ma16": ma16}

    if ma4 > ma8 > ma16 and current_price > ma4:
        status = "强势上升"
    elif ma4 < ma8 < ma16 and current_price < ma4:
        status = "强势下降"
    else:
        spread = (max(ma4, ma8, ma16) - min(ma4, ma8, ma16)) / ma8 if ma8 else 1.0
        near_ma8 = abs(current_price - ma8) / ma8 if ma8 else 1.0
        if spread < 0.05 and near_ma8 <= 0.08:
            status = "横盘震荡"
        else:
            status = "弱势上升" if ma4 > ma8 else "弱势下降"

    return {"status": status, "ma4": ma4, "ma8": ma8, "ma16": ma16}

def price_position(close_series: pd.Series, current_price: float, lookback: int = 16) -> dict:
    s = close_series.dropna().astype(float).tail(lookback)
    if s.empty:
        return {"pos_pct": None, "level": "数据不足", "hi": None, "lo": None}
    hi, lo = float(s.max()), float(s.min())
    rng = hi - lo
    if hi <= 0 or rng <= 0:
        return {"pos_pct": None, "level": "数据不足", "hi": hi, "lo": lo}
    pos = (current_price - lo) / rng
    level = "高位" if pos > 0.70 else ("低位" if pos < 0.30 else "中位")
    return {"pos_pct": float(pos), "level": level, "hi": hi, "lo": lo}

def peak_drawdown_buy_signal(close_series: pd.Series, current_price: float) -> dict:
    s = close_series.dropna().astype(float).tail(16)
    if s.empty:
        return {"score": 0, "buy": False, "drawdown": None, "details": {}}
    peak = float(s.max())
    dd = (peak - current_price) / peak if peak > 0 else None
    ma8 = _safe_last_ma(close_series.dropna().astype(float), 8)
    ma16 = _safe_last_ma(close_series.dropna().astype(float), 16)
    trend = classify_trend(close_series, current_price)["status"]

    score, details = 0, {}
    if dd is not None:
        if dd >= 0.15:
            score += 3; details["dd>=15%"] = 3
        if dd >= 0.25:
            score += 2; details["dd>=25%"] = 2
    if ma16 is not None and current_price < ma16:
        score += 2; details["px<MA16"] = 2
    if ma8 is not None and current_price < ma8:
        score += 1; details["px<MA8"] = 1
    if trend in ("强势下降", "弱势下降"):
        score += 2; details["downtrend"] = 2

    return {"score": score, "buy": score >= 5, "drawdown": dd, "details": details}

def volatility_20w(close_series: pd.Series) -> Optional[float]:
    rets = close_series.dropna().astype(float).pct_change().dropna().tail(20)
    if rets.empty:
        return None
    return float(rets.std() * 100.0)

def dynamic_take_profit(cost: Optional[float], current_price: float, close_series: pd.Series, trend_status: str) -> dict:
    if cost is None or cost <= 0:
        return {"available": False, "reason": "未提供成本价"}

    vol_pct = volatility_20w(close_series)
    if vol_pct is not None and vol_pct > 8.0:
        vol_factor = 1.3
    elif vol_pct is not None and vol_pct < 4.0:
        vol_factor = 0.8
    else:
        vol_factor = 1.0

    if trend_status == "强势上升":
        trend_factor = 1.2
    elif trend_status in ("强势下降", "弱势下降"):
        trend_factor = 0.8
    else:
        trend_factor = 1.0

    base_thr, base_sell = [0.20, 0.35, 0.50], [0.10, 0.15, 0.25]
    final_thr = [round(t * vol_factor * trend_factor, 4) for t in base_thr]

    pr = (current_price - cost) / cost
    actions, total = [], 0.0
    for t, s in zip(final_thr, base_sell):
        if pr >= t:
            actions.append({"threshold": t, "sell_ratio": s})
            total += s

    return {
        "available": True,
        "vol_pct": vol_pct,
        "vol_factor": vol_factor,
        "trend_factor": trend_factor,
        "final_thresholds": final_thr,
        "profit_ratio": pr,
        "plan": actions,
        "total_sell_ratio": total,
    }

def averaging_recommendation(cost: Optional[float], current_price: float, weekly_close: pd.Series, weekly_volume: pd.Series,
                             support_near_pct: float = 0.02) -> dict:
    if cost is None or cost <= 0:
        return {"available": False, "reason": "未提供成本价"}

    loss = (cost - current_price) / cost
    s_close = weekly_close.dropna().astype(float)
    s_vol = weekly_volume.dropna().astype(float)
    if s_close.empty or s_vol.empty:
        return {"available": False, "reason": "数据不足"}

    lo8 = float(s_close.tail(8).min())
    lo16 = float(s_close.tail(16).min())
    cur_vol = float(s_vol.iloc[-1]) if len(s_vol) else None
    avg8_vol = float(s_vol.tail(8).mean()) if len(s_vol) else None
    vol_ratio = (cur_vol / avg8_vol) if (cur_vol is not None and avg8_vol and avg8_vol > 0) else None

    def near(px, sup, tol):
        return False if not sup or sup <= 0 else abs(px - sup) / sup <= tol

    near8 = near(current_price, lo8, support_near_pct)
    near16 = near(current_price, lo16, support_near_pct)

    decision = {"action": "不加仓", "size": 0.0}
    if loss >= 0.25:
        decision = {"action": "重新评估投资逻辑", "size": 0.0}
    elif loss >= 0.15 and (near16 or (vol_ratio is not None and vol_ratio < 0.5)):
        decision = {"action": "加仓", "size": 0.30}
    elif loss >= 0.08 and (near8 or (vol_ratio is not None and vol_ratio < 0.7)):
        decision = {"action": "加仓", "size": 0.20}

    return {
        "available": True,
        "loss_ratio": float(loss),
        "support8": lo8,
        "support16": lo16,
        "near8": near8,
        "near16": near16,
        "volume_ratio": vol_ratio,
        "decision": decision,
    }

def risk_control(weekly_close: pd.Series, cost: Optional[float], current_price: float, trend_status: str) -> dict:
    s20 = weekly_close.dropna().astype(float).tail(20)
    if s20.empty:
        return {"available": False, "reason": "数据不足"}

    hi, lo = float(s20.max()), float(s20.min())
    mdd = (hi - lo) / hi if hi > 0 else None
    if mdd is None:
        return {"available": False, "reason": "数据不足"}

    if mdd > 0.35:
        risk, pos = "高风险", "10%-15%"
    elif mdd >= 0.20:
        risk, pos = "中风险", "15%-20%"
    else:
        risk, pos = "低风险", "20%-25%"

    stop, warn = None, None
    if cost is not None and cost > 0:
        loss = (cost - current_price) / cost
        if loss >= 0.30:
            stop = "浮亏≥30%，强制止损"
        elif loss >= 0.20 and trend_status in ("强势下降", "弱势下降"):
            warn = "浮亏≥20%且下降趋势，风险警告"

    return {
        "available": True,
        "mdd": float(mdd),
        "risk_level": risk,
        "position_suggestion": pos,
        "stop_loss": stop,
        "warning": warn,
    }

def analyze_symbol(symbol: str, weekly_df: pd.DataFrame, current_price: float, cost: Optional[float]) -> dict:
    close = weekly_df["close"]
    volume = weekly_df["volume"] if "volume" in weekly_df.columns else pd.Series([], dtype=float)

    trend = classify_trend(close, current_price)
    pos = price_position(close, current_price, lookback=16)
    buy_sig = peak_drawdown_buy_signal(close, current_price)
    take_profit = dynamic_take_profit(cost, current_price, close, trend["status"])
    avg_rec = averaging_recommendation(cost, current_price, close, volume)
    risk = risk_control(close, cost, current_price, trend["status"])

    return {
        "symbol": symbol,
        "current_price": current_price,
        "trend": trend,
        "position": pos,
        "buy_signal": buy_sig,
        "take_profit": take_profit,
        "averaging": avg_rec,
        "risk": risk,
    }

def summarize_result(res: dict) -> str:
    sym = res["symbol"]
    trend = res["trend"]["status"]
    pos = res["position"]
    pos_str = "未知"
    if pos.get("pos_pct") is not None:
        pos_str = f"{pos['level']}({round(pos['pos_pct'] * 100, 1)}%)"
    dd = res["buy_signal"].get("drawdown")
    dd_str = "-" if dd is None else f"{round(dd * 100, 1)}%"
    buy_phrase = "→关注买入" if res["buy_signal"].get("buy") else ""

    # 盈亏（由外层拼接）
    risk = res["risk"]
    risk_phrase = f"，{risk['risk_level']}→仓位{risk['position_suggestion']}" if risk.get("available") else ""
    return f"{sym}：{trend}({pos_str})，4个月回撤{dd_str}{buy_phrase}{risk_phrase}"


# =========================================================
# 6) 无状态“穿越触发”（昨日 vs 今日）
# =========================================================
def zone_level(pr: float, profit_levels: list, loss_levels: list) -> Tuple[str, int]:
    profit_levels = sorted(profit_levels or [])
    loss_levels = sorted(loss_levels or [])

    pl = 0
    for i, thr in enumerate(profit_levels, start=1):
        if pr >= thr:
            pl = i

    ll = 0
    # loss 阈值如 [-0.10, -0.20, -0.35]，reverse 后从 -0.10 开始判断，逐步到更深亏
    for i, thr in enumerate(sorted(loss_levels, reverse=True), start=1):
        if pr <= thr:
            ll = i

    if pl > 0:
        return "profit", pl
    if ll > 0:
        return "loss", ll
    return "none", 0

def should_push_cross_only(pr_now: float, pr_prev: Optional[float], profit_levels: list, loss_levels: list) -> Dict[str, Any]:
    z_now, lvl_now = zone_level(pr_now, profit_levels, loss_levels)

    if z_now == "none":
        return {"push": False, "reason": "正常区", "zone": z_now, "level": lvl_now}

    if pr_prev is None:
        if PUSH_IF_NO_PREV:
            return {"push": True, "reason": "无昨收(允许触发)", "zone": z_now, "level": lvl_now}
        return {"push": False, "reason": "无昨收(不触发)", "zone": z_now, "level": lvl_now}

    z_prev, lvl_prev = zone_level(pr_prev, profit_levels, loss_levels)

    if z_prev == "none" and z_now != "none":
        return {"push": True, "reason": "阈值穿越", "zone": z_now, "level": lvl_now}
    if z_prev != z_now:
        return {"push": True, "reason": "方向切换", "zone": z_now, "level": lvl_now}
    if z_prev == z_now and lvl_now > lvl_prev:
        return {"push": True, "reason": "情况升级", "zone": z_now, "level": lvl_now}

    return {"push": False, "reason": "仍在同档位", "zone": z_now, "level": lvl_now}


def _fmt_float(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "NA"
        if isinstance(x, float) and pd.isna(x):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


# =========================================================
# 7) 主程序
# =========================================================
if __name__ == "__main__":
    watch_cfg, symbols = build_watch_cfg()
    print("监控标的：", symbols)

    weekly_dict, latest_df, _wide_df = fetch_all_symbols(symbols)
    latest_map = {row["symbol"]: row for _, row in latest_df.iterrows()} if not latest_df.empty else {}

    all_lines = []
    alert_lines = []

    for sym in symbols:
        cfg = watch_cfg.get(sym, {})
        cost = cfg.get("cost")
        profit_levels = cfg.get("profit_levels", [])
        loss_levels = cfg.get("loss_levels", [])

        lrow = latest_map.get(sym, pd.Series({}))
        price = lrow.get("price", None)
        pre_close = lrow.get("pre_close", None)
        trade_date = lrow.get("trade_date", None)

        if price is None or (isinstance(price, float) and pd.isna(price)):
            all_lines.append(f"{sym}：未获取到最新收盘价，跳过")
            continue
        if cost is None or cost <= 0:
            all_lines.append(f"{sym}：现价{_fmt_float(price)}，但未设置成本 c，无法做阈值触发")
            continue

        pr_now = (float(price) - float(cost)) / float(cost)
        pr_prev = None
        if pre_close is not None and not (isinstance(pre_close, float) and pd.isna(pre_close)):
            try:
                pr_prev = (float(pre_close) - float(cost)) / float(cost)
            except Exception:
                pr_prev = None

        wdf = weekly_dict.get(sym)
        if wdf is not None and not wdf.empty:
            res = analyze_symbol(sym, wdf, float(price), cost)
            base_summary = summarize_result(res)
        else:
            base_summary = f"{sym}：周线缺失，仅做阈值提醒（可能因数据不足/限速）"

        dec = should_push_cross_only(pr_now, pr_prev, profit_levels, loss_levels)

        pr_now_pct = pr_now * 100.0
        pr_prev_pct = (pr_prev * 100.0) if pr_prev is not None else None

        line = (
            f"{base_summary}\n"
            f"- 交易日：{trade_date}  成本：{_fmt_float(cost)}  收盘：{_fmt_float(price)}  昨收：{_fmt_float(pre_close)}\n"
            f"- 今日盈亏：{pr_now_pct:+.1f}%"
            + (f"  昨日盈亏：{pr_prev_pct:+.1f}%" if pr_prev_pct is not None else "")
            + f"\n- 触发判定：{dec['zone']} 档位{dec['level']}（{dec['reason']}）"
        )

        all_lines.append(line)
        if dec["push"]:
            alert_lines.append(line)

    print("\n==================== 全部结果（打印） ====================")
    print("\n\n---\n\n".join(all_lines) if all_lines else "(空)")

    if alert_lines:
        title = time.strftime("港股/A股 触发提醒 %Y-%m-%d %H:%M", time.localtime())
        body = "\n\n---\n\n".join(alert_lines) + "\n\n> 这是阈值提醒，不代表必须交易；按长期计划执行。"
        push_serverchan(title, body)
    else:
        if PUSH_DAILY_SUMMARY:
            title = time.strftime("港股/A股 每日简报 %Y-%m-%d %H:%M", time.localtime())
            body = "\n\n---\n\n".join(all_lines) if all_lines else "(空)"
            push_serverchan(title, body)
        else:
            print("\n今日无触发提醒：不推送。")
