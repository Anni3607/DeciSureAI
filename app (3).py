
# DeciSure AI - Streamlit front-end
# Save this text to a file named app.py and run `streamlit run app.py`

import streamlit as st
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math
import numpy as np

# ---- include the same utility functions used in the notebook ----
# For brevity in the app we reimplement minimal needed functions (you can import from a module instead)

analyzer = SentimentIntensityAnalyzer()

def parse_numeric_attr(attr):
    if attr is None:
        return None
    if isinstance(attr, (int,float)):
        return float(attr)
    s = str(attr).lower().replace(',', '').replace(' ', '')
    s = ''.join(ch for ch in s if ch.isdigit() or ch=='.' or ch=='-')
    try:
        return float(s)
    except:
        return None

def sentiment_score(text):
    if not text:
        return 0.0
    return analyzer.polarity_scores(str(text))['compound']

def text_score_similarity(goal_text, option_text):
    if not goal_text or not option_text:
        return 0.0
    gset = set(str(goal_text).lower().split())
    oset = set(str(option_text).lower().split())
    if len(gset)==0:
        return 0.0
    overlap = gset.intersection(oset)
    return len(overlap) / len(gset)

def normalize_scores(scores):
    vals = np.array(list(scores.values()), dtype=float)
    minv, maxv = vals.min(), vals.max()
    if math.isclose(maxv, minv):
        return {k: 0.5 for k in scores}
    return {k: (v - minv) / (maxv - minv) for k,v in scores.items()}

def compute_option_features(option_dict, user_goal_text=None):
    features = {}
    price = parse_numeric_attr(option_dict.get('price'))
    rating = parse_numeric_attr(option_dict.get('rating'))
    risk = parse_numeric_attr(option_dict.get('risk'))
    desc = option_dict.get('description', '')
    sent = sentiment_score(desc)
    sim = text_score_similarity(user_goal_text, desc) if user_goal_text else 0.0
    features['price'] = -price if price is not None else None
    features['rating'] = rating if rating is not None else None
    features['risk'] = -risk if risk is not None else None
    features['sentiment'] = sent
    features['similarity'] = sim
    return features

def score_two_options(optA, optB, user_goal_text=None, weights=None):
    default_weights = {'price': 1.0, 'rating': 1.0, 'risk': 1.0, 'sentiment': 0.5, 'similarity': 1.5}
    if weights is None:
        weights = default_weights
    fA = compute_option_features(optA, user_goal_text=user_goal_text)
    fB = compute_option_features(optB, user_goal_text=user_goal_text)
    per_criterion_score = {}
    criteria = list(set(list(fA.keys()) + list(fB.keys())))
    for c in criteria:
        valA = fA.get(c); valB = fB.get(c)
        if valA is None and valB is None:
            rawA = rawB = 0.0
        elif valA is None:
            rawA = valB; rawB = valB
        elif valB is None:
            rawA = valA; rawB = valA
        else:
            rawA = valA; rawB = valB
        per_criterion_score[c] = {'A': rawA, 'B': rawB}
    normalized = {}
    contribs = {}
    scoreA = 0.0; scoreB = 0.0
    for c, vals in per_criterion_score.items():
        nm = normalize_scores({'A': vals['A'], 'B': vals['B']})
        w = float(weights.get(c, 0.0))
        contribA = w * nm['A']; contribB = w * nm['B']
        scoreA += contribA; scoreB += contribB
        contribs[c] = {'A': contribA, 'B': contribB, 'weight': w, 'raw': vals}
        normalized[c] = nm
    finalA = scoreA; finalB = scoreB
    expA = math.exp(finalA); expB = math.exp(finalB)
    probA = expA / (expA + expB); probB = expB / (expA + expB)
    chosen = 'A' if probA >= probB else 'B'
    # Simple counterfactual
    counterfactual = None
    for c in sorted(normalized.keys(), key=lambda x: -abs(contribs[x]['A'] - contribs[x]['B'])):
        tempA_adj = finalA; tempB_adj = finalB
        if chosen == 'A':
            tempA_adj -= 0.5 * contribs[c]['A']
        else:
            tempB_adj -= 0.5 * contribs[c]['B']
        eA = math.exp(tempA_adj); eB = math.exp(tempB_adj)
        pA = eA / (eA + eB); pB = eB / (eA + eB)
        new_choice = 'A' if pA >= pB else 'B'
        if new_choice != chosen:
            counterfactual = {
                'criterion': c,
                'effect': f"Reducing '{c}' contribution for the winning option by 50% would change winner to {new_choice}.",
                'old_probs': {'A': round(probA,3),'B': round(probB,3)},
                'new_probs': {'A': round(pA,3),'B': round(pB,3)}
            }
            break
    return {
        'chosen': chosen,
        'probabilities': {'A': round(probA,4), 'B': round(probB,4)},
        'final_scores': {'A': round(finalA,4), 'B': round(finalB,4)},
        'contributions': contribs,
        'counterfactual': counterfactual,
        'normalized': normalized,
        'raw_features': {'A': fA, 'B': fB}
    }

# ---- Streamlit UI ----
st.set_page_config(page_title="DeciSure AI", layout="wide")
st.title("DeciSure AI — Two-option Decision Maker (Prototype)")

with st.sidebar:
    st.header("Options input")
    st.markdown("Enter two options to compare. Provide numeric fields where possible for better results.")

# Main input area
col1, col2 = st.columns(2)
with col1:
    st.subheader("Option A")
    A_name = st.text_input("Name (A)", value="Option A")
    A_price = st.text_input("Price (A)", value="")
    A_rating = st.text_input("Rating (A)", value="")
    A_risk = st.text_input("Risk (A) [lower better]", value="")
    A_desc = st.text_area("Description (A)", value="")

with col2:
    st.subheader("Option B")
    B_name = st.text_input("Name (B)", value="Option B")
    B_price = st.text_input("Price (B)", value="")
    B_rating = st.text_input("Rating (B)", value="")
    B_risk = st.text_input("Risk (B) [lower better]", value="")
    B_desc = st.text_area("Description (B)", value="")

st.subheader("User goal / context (short)")
user_goal = st.text_input("What do you prioritize? (e.g., 'best for photography under 40k')", "")

st.subheader("Optional: adjust criteria weights (higher = more important)")
w_price = st.slider("Price weight", 0.0, 3.0, 1.0, 0.1)
w_rating = st.slider("Rating weight", 0.0, 3.0, 1.0, 0.1)
w_risk = st.slider("Risk weight", 0.0, 3.0, 1.0, 0.1)
w_sent = st.slider("Sentiment weight", 0.0, 2.0, 0.5, 0.1)
w_sim = st.slider("Similarity to goal weight", 0.0, 3.0, 1.5, 0.1)

weights = {'price': w_price, 'rating': w_rating, 'risk': w_risk, 'sentiment': w_sent, 'similarity': w_sim}

if st.button("Decide"):
    optA = {'name': A_name, 'price': A_price, 'rating': A_rating, 'risk': A_risk, 'description': A_desc}
    optB = {'name': B_name, 'price': B_price, 'rating': B_rating, 'risk': B_risk, 'description': B_desc}
    result = score_two_options(optA, optB, user_goal_text=user_goal, weights=weights)
    st.success(f"Final choice: Option {result['chosen']} ({A_name if result['chosen']=='A' else B_name})")
    st.write("Probabilities:", result['probabilities'])
    st.write("Final scores:", result['final_scores'])

    st.subheader("Detailed per-criterion contributions")
    for c, v in result['contributions'].items():
        st.markdown(f"- **{c}** (weight {v['weight']}): A_contrib={round(v['A'],3)}  —  B_contrib={round(v['B'],3)}")

    st.subheader("Counterfactual / Robustness check")
    if result['counterfactual']:
        st.info(result['counterfactual']['effect'])
        st.write("Old probs:", result['counterfactual']['old_probs'])
        st.write("New probs if changed:", result['counterfactual']['new_probs'])
    else:
        st.write("No single-criterion small change would flip decision (robust).")

    st.subheader("Raw features (for transparency)")
    st.write(result['raw_features'])
    st.caption("Tip: Provide numeric ratings/prices and richer descriptions to get better decisions.")
