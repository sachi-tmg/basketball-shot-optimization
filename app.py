import os
import math
import joblib
import numpy as np
import streamlit as st

THRESHOLD = 0.50
DEBUG = False

MODEL_DIR = "models"
MAKE_PATHS = [
    f"{MODEL_DIR}/random_forest_model.joblib",
    f"{MODEL_DIR}/logistic_regression_model.joblib",
]
TYPE_PATHS = [] 

COURT_IMAGE_CANDIDATES = [
    "basketball_court.png",
]

@st.cache(allow_output_mutation=True)
def _load_first(paths):
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p), p
    return None, None

@st.cache(allow_output_mutation=True)
def load_models():
    make_model, make_path = _load_first(MAKE_PATHS)
    type_model, type_path = _load_first(TYPE_PATHS)
    return (make_model, make_path, type_model, type_path)

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def build_row_by_names(model, features_dict):
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    else:
        cols = list(features_dict.keys())
    row = [float(features_dict.get(c, 0.0)) for c in cols]
    return np.array([row]), cols

def predict_proba_pos(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return float(proba[0, 1])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        return 1.0 / (1.0 + math.exp(-score))
    pred = float(model.predict(X)[0])
    return max(0.0, min(1.0, pred))

def advice_text(shot_type, make_prob, distance_ft, period, seconds_left, margin):
    tips = []

    if shot_type == "3PT":
        if distance_ft >= 26:
            tips.append("The three is deep; use a screen or step in to shorten the shot.")
        else:
            tips.append("Hunt a cleaner catch-and-shoot three (corner or off a kick-out).")
    else:
        if 12 <= distance_ft < 22:
            tips.append("Mid-range looks are lower value; attack the rim or relocate for a corner three.")
        elif distance_ft < 8:
            tips.append("If the paint is crowded, pump-fake to draw a foul or kick to an open shooter.")

    if seconds_left > 300 and make_prob < THRESHOLD:
        tips.append("Plenty of clock left, reset the action and work for a higher-quality look.")
    elif 24 <= seconds_left <= 300 and make_prob < THRESHOLD:
        tips.append("Use a quick action like P&R or handoff to create separation before shooting.")
    else:
        if margin < 0:
            tips.append("Create a set three rather than a contested pull-up." if shot_type == "3PT"
                        else "Down late, consider a kick-out three or quick two then foul.")
        else:
            tips.append("Protect the possession, get a safe look or force a switch for a mismatch.")

    if margin <= -5 and shot_type == "2PT":
        tips.append("You are trailing, a higher-value three may be preferable if it is open.")
    if margin >= 5 and make_prob < THRESHOLD:
        tips.append("You are ahead, prioritize a high-percentage look and use clock.")

    seen, out = set(), []
    for t in tips:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return " ".join(out) if out else "Consider passing or running a set play to improve shot quality."

def heuristic_is_three(distance_ft: float) -> int:
    return int(distance_ft >= 22.0)

# ================== UI ==================
st.set_page_config(page_title="Shot Recommendation", layout="wide")
st.title("Basketball Shot Recommendation")

make_model, make_path, type_model, type_path = load_models()

if make_model is None:
    st.error("No make/miss model found in models/. Expected one of: " + ", ".join(MAKE_PATHS))
    st.stop()

left, right = st.columns([2, 1])

with left:
    st.subheader("Game Context")
    shot_distance = st.slider("Shot Distance (feet)", 0, 35, 15)
    period_num = st.selectbox("Period (Quarter)", [1, 2, 3, 4], index=0)
    seconds_left = st.slider("Seconds Left in Period", 0, 720, 360)
    score_margin = st.slider("Score Margin (your team lead; negative if trailing)", -30, 30, 0)
    run_btn = st.button("Compute Recommendation")

with right:
    st.subheader("Context")
    mm, ss = divmod(int(seconds_left), 60)
    st.write(f"Period: {period_num}")
    st.write(f"Time left: {mm:02d}:{ss:02d}")
    st.write(f"Score margin: {score_margin}")
    st.write(f"Distance: {shot_distance} ft")
    st.caption(f"Decision threshold fixed at {THRESHOLD:.0%} make probability.")

    # Court diagram
    img_path = first_existing(COURT_IMAGE_CANDIDATES)
    if img_path:
        st.image(img_path, caption="2PT vs 3PT distances (22 ft corners, 23.75 ft arc)", use_column_width=True)
    else:
        st.info("Court diagram not found. Place basketball_court.png in project root or visualizations/.")

st.markdown("---")

if run_btn:
    # 1) Shot type
    X_type_dict = {
        "shot_distance": shot_distance,
        "period_num": period_num,
        "seconds_left": seconds_left,
        "score_margin": score_margin,
    }
    if type_model is not None:
        X_type, type_cols = build_row_by_names(type_model, X_type_dict)
        is_three_pred = int(type_model.predict(X_type)[0]) if hasattr(type_model, "predict") \
                        else int(predict_proba_pos(type_model, X_type) >= 0.5)
        prob_three = predict_proba_pos(type_model, X_type)
        prob_two = 1.0 - prob_three
        shot_type_str = "3PT" if is_three_pred else "2PT"
    else:
        is_three_pred = heuristic_is_three(shot_distance)
        shot_type_str = "3PT" if is_three_pred else "2PT"
        prob_three = float(is_three_pred)
        prob_two = 1.0 - prob_three
        type_cols = list(X_type_dict.keys())

    # 2) Make probability
    X_make_dict = {
        "shot_distance": shot_distance,
        "period_num": period_num,
        "seconds_left": seconds_left,
        "score_margin": score_margin,
        "is_three": is_three_pred,
    }
    X_make, make_cols = build_row_by_names(make_model, X_make_dict)
    make_prob = predict_proba_pos(make_model, X_make)

    # ----- Output -----
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Recommendation")
        st.markdown(f"**Recommended Shot Type:** {shot_type_str} (3PT: {prob_three:.1%}, 2PT: {prob_two:.1%})")

        pct = int(round(make_prob * 100))
        st.progress(max(0, min(100, pct)))
        st.write(f"Estimated make probability: {make_prob:.3f}")

        if make_prob >= THRESHOLD:
            st.success("Recommended: Take the shot.")
        else:
            st.warning("Not recommended: " + advice_text(
                shot_type=shot_type_str,
                make_prob=make_prob,
                distance_ft=shot_distance,
                period=period_num,
                seconds_left=seconds_left,
                margin=score_margin,
            ))

    if DEBUG:
        with c2:
            with st.expander("Inputs sent to each model", expanded=False):
                st.markdown("Type model features (order):")
                st.code(", ".join(type_cols))
                st.json({k: float(v) for k, v in X_type_dict.items()})
                st.markdown("Make model features (order):")
                st.code(", ".join(make_cols))
                st.json({k: float(v) for k, v in X_make_dict.items()})

    st.markdown("### Why this recommendation?")
    st.markdown(
        f"- Shot type resolved as **{shot_type_str}** from **{shot_distance} ft** in this context (Q{period_num}, {seconds_left}s left, margin {score_margin}).\n"
        f"- Make probability estimated at **{make_prob:.0%}**; decision rule uses a fixed **{THRESHOLD:.0%}** cutoff.\n"
        "- Feature intuition: longer distance lowers make percent, time and margin affect urgency and shot quality, 3PT vs 2PT changes base difficulty.\n"
    )

st.caption("Shot recommendation made with Streamlit by Sachi")

