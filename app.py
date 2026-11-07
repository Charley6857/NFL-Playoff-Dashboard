import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="NFL Playoff Dashboard", layout="wide")
BASE = os.path.dirname(os.path.abspath(__file__))


CSV_PATH = os.path.join(BASE, "team_seasons.csv")


@st.cache_data
def load(file_stamp):
    # file_stamp (the file's last-modified time) makes Streamlit reload the data
    # whenever team_seasons.csv is replaced, instead of serving a stale cached copy.
    return pd.read_csv(CSV_PATH)


df = load(os.path.getmtime(CSV_PATH))
latest = int(df.season.max())
counts = df.groupby("season").made_playoffs.sum()
done = sorted(counts[counts > 0].index)  # finished seasons (playoffs known)
hist = df[df.season.isin(done)]

st.title("NFL Playoff Dashboard")
st.caption("Who is on track to make the playoffs, and what should teams do about it?")

season = st.sidebar.selectbox("Season (Descriptive tab)",
                              sorted(df.season.unique(), reverse=True))

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Descriptive", "2. Diagnostic", "3. Predictive", "4. Prescriptive"])

# ---------------- 1. DESCRIPTIVE ----------------
with tab1:
    st.header(f"What happened: {season} standings")
    s = df[df.season == season].sort_values("win_pct", ascending=False)
    st.dataframe(
        s[["team", "wins", "games", "win_pct", "pf", "pa",
           "point_diff", "turnover_margin"]].reset_index(drop=True))
    fig = px.bar(s, x="team", y=["pf", "pa"], barmode="group",
                 title=f"Points scored vs. allowed, {season}",
                 labels={"value": "Points", "variable": ""})
    st.plotly_chart(fig)
    top = s.iloc[0]
    st.info(f"Takeaway: {top.team} has the best record in {season} "
            f"({top.wins:g} wins in {int(top.games)} games, "
            f"point differential {top.point_diff:+.0f}).")

# ---------------- 2. DIAGNOSTIC ----------------
with tab2:
    st.header("Why it happened: what separates playoff teams?")
    feats = ["pf_pg", "pa_pg", "to_margin_pg", "off_epa", "def_epa", "win_pct"]
    feature_labels = {
        "pf_pg": "Points Scored Per Game",
        "pa_pg": "Points Allowed Per Game",
        "to_margin_pg": "Turnover Margin Per Game",
        "off_epa": "Offensive Expected Points Added (EPA)",
        "def_epa": "Defensive Expected Points Added (EPA)",
        "win_pct": "Win Percentage",
        "made_playoffs": "Made Playoffs",
    }
    corr = hist[feats + ["made_playoffs"]].corr()
    display_corr = corr.rename(index=feature_labels, columns=feature_labels)
    fig = px.imshow(display_corr, text_auto=".2f", color_continuous_scale="RdBu",
                    zmin=-1, zmax=1, title="Correlation heatmap (2015-last finished season)")
    st.plotly_chart(fig)

    x = st.selectbox("Stat to compare against win %", feats[:-1],
                     format_func=lambda name: feature_labels[name])
    plot = hist.assign(playoffs=hist.made_playoffs.map({1: "Made", 0: "Missed"}))
    fig = px.scatter(plot, x=x, y="win_pct", color="playoffs", hover_data=["team", "season"],
                     labels={**feature_labels, "playoffs": "Playoff Result"})
    m, b = np.polyfit(hist[x], hist.win_pct, 1)
    xs = np.linspace(hist[x].min(), hist[x].max(), 50)
    fig.add_scatter(x=xs, y=m * xs + b, mode="lines", name="trend line")
    st.plotly_chart(fig)

    c = corr["made_playoffs"].drop(["made_playoffs", "win_pct"])
    best = c.abs().idxmax()
    st.info(f"Takeaway: '{feature_labels[best]}' has the strongest link to making the playoffs "
            f"(correlation {c[best]:+.2f}). Note: lower Defensive EPA and Points Allowed Per Game are better, "
            f"so negative correlations there are good.")

# ---------------- 3. PREDICTIVE ----------------
FEATS = ["pf_pg", "pa_pg", "to_margin_pg"]


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression())


test_season = done[-1]
train = df[df.season.isin(done[:-1])]
test = df[df.season == test_season]
check = make_model().fit(train[FEATS], train.made_playoffs)
acc = accuracy_score(test.made_playoffs, check.predict(test[FEATS]))
auc = roc_auc_score(test.made_playoffs, check.predict_proba(test[FEATS])[:, 1])

final = make_model().fit(hist[FEATS], hist.made_playoffs)
cur = df[df.season == latest].copy()
cur["playoff_prob"] = final.predict_proba(cur[FEATS])[:, 1]

with tab3:
    st.header(f"What will happen: {latest} playoff probabilities")
    c1, c2 = st.columns(2)
    c1.metric(f"Accuracy on held-out {test_season}", f"{acc:.1%}")
    c2.metric("AUC (1.0 = perfect)", f"{auc:.2f}")
    p = cur.sort_values("playoff_prob", ascending=False)
    fig = px.bar(p, x="team", y="playoff_prob", title=f"Playoff probability by team, {latest}",
                 labels={"playoff_prob": "Probability"})
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_traces(hovertemplate="%{x}: %{y:.2%}<extra></extra>")
    st.plotly_chart(fig)
    top_probability = p.iloc[0].playoff_prob
    probability_label = ">99.99%" if 0.99995 <= top_probability < 1 else f"{top_probability:.2%}"
    st.info(f"Takeaway: {p.iloc[0].team} is the most likely playoff team right now "
            f"({probability_label}). Model: logistic regression on points scored, "
            f"points allowed, and turnover margin per game, trained on {done[0]}-{done[-2]} "
            f"and tested on {test_season}. Early-season stats are noisy, so treat these as a snapshot.")

# ---------------- 4. PRESCRIPTIVE ----------------
def prob(pf, pa, to):
    return final.predict_proba(pd.DataFrame([[pf, pa, to]], columns=FEATS))[0, 1]


with tab4:
    st.header("What should be done: what-if for a team")
    st.caption("Every slider works the same way: moving it to the RIGHT (positive) "
               "is an IMPROVEMENT for the team. Moving it left makes the team worse.")
    team = st.selectbox("Team", sorted(cur.team))
    r = cur[cur.team == team].iloc[0]
    base = prob(r.pf_pg, r.pa_pg, r.to_margin_pg)

    d_pf = st.slider("Points scored per game: increase by", -7.0, 7.0, 0.0, 0.5)
    d_pa = st.slider("Points allowed per game: reduce by (positive = better defense)",
                     -7.0, 7.0, 0.0, 0.5)
    d_to = st.slider("Turnover margin per game: improve by", -1.5, 1.5, 0.0, 0.25)

    new_pf = r.pf_pg + d_pf
    new_pa = r.pa_pg - d_pa          # reducing points allowed is the improvement
    new_to = r.to_margin_pg + d_to
    new = prob(new_pf, new_pa, new_to)

    st.dataframe(pd.DataFrame({
        "Stat (per game)": ["Points scored", "Points allowed", "Turnover margin"],
        "Current": [round(r.pf_pg, 2), round(r.pa_pg, 2), round(r.to_margin_pg, 2)],
        "Adjusted": [round(new_pf, 2), round(new_pa, 2), round(new_to, 2)],
    }), hide_index=True)
    c1, c2 = st.columns(2)
    c1.metric("Current playoff probability", f"{base:.2%}")
    c2.metric("Adjusted playoff probability", f"{new:.2%}", f"{(new - base) * 100:+.2f} points")

    levers = {
        "Score 3 more points per game": prob(r.pf_pg + 3, r.pa_pg, r.to_margin_pg),
        "Allow 3 fewer points per game": prob(r.pf_pg, r.pa_pg - 3, r.to_margin_pg),
        "Improve turnover margin by 0.5 per game": prob(r.pf_pg, r.pa_pg, r.to_margin_pg + 0.5),
    }
    gain = pd.DataFrame({"lever": list(levers), "gain": [(v - base) * 100 for v in levers.values()]})
    fig = px.bar(gain, x="lever", y="gain", title=f"{team}: playoff-probability gain by improvement",
                 labels={"gain": "Gain (percentage points)"})
    st.plotly_chart(fig)
    top = gain.sort_values("gain", ascending=False).iloc[0]
    st.success(f"Recommendation for {team}: prioritize '{top.lever.lower()}'. "
               f"It raises playoff probability by about {top.gain:.1f} points.")
