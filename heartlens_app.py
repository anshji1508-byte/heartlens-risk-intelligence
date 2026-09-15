from pathlib import Path
import joblib, shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="HeartLens | Risk Intelligence", page_icon="❤", layout="wide")
ROOT = Path(__file__).parent
@st.cache_resource
def load_artifact(): return joblib.load(ROOT / "heart_disease_model.joblib")
artifact=load_artifact(); model,threshold=artifact["model"],artifact["threshold"]
# HeartLens uses a single permanent dark theme.
night_mode = True
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:#f6f8fc}
header[data-testid="stHeader"]{background:transparent!important;box-shadow:none!important}
[data-testid="stToolbar"]{visibility:hidden!important;height:0!important}
#MainMenu{visibility:hidden!important}
footer{visibility:hidden!important}
.hero{position:relative;overflow:hidden;background:linear-gradient(120deg,#0b1f3a,#123e63 55%,#176b73);background-size:200% 200%;animation:heroIn .7s ease-out both,gradientFlow 10s ease infinite;padding:2.5rem 2.8rem;border-radius:24px;color:white;margin-bottom:1.4rem;box-shadow:0 14px 35px #0e2c4b2e}
.hero h1{font-family:'Space Grotesk';font-size:2.6rem;margin:0 0 .4rem;letter-spacing:-.04em;color:#ffffff}.hero p{color:#ffffff!important;font-size:1.05rem;margin:0;max-width:760px}
.pill{display:inline-block;padding:.35rem .8rem;border:1px solid #73d5d1;border-radius:999px;color:#9ce9e5;font-size:.78rem;margin-bottom:.9rem}
.section-title{font-family:'Space Grotesk';color:#0b1f3a;font-size:1.45rem;font-weight:700;margin:1.1rem 0 .2rem}.muted{color:#607089;font-size:.92rem;margin-bottom:1.5rem!important}
.risk-high{background:#fff1f1;border-left:6px solid #e14b56;padding:1rem 1.2rem;border-radius:14px;color:#79242d}.risk-low{background:#ecfbf8;border-left:6px solid #17a68d;padding:1rem 1.2rem;border-radius:14px;color:#126353}
.disclaimer{background:#fff8e8;border:1px solid #f1d28a;color:#6b4c13;border-radius:14px;padding:.9rem 1rem;font-size:.88rem}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:#eaf0f7;padding:6px;border-radius:12px}
.stTabs [data-baseweb="tab"]{color:#17324d!important;background:#ffffff;border-radius:9px;padding:10px 18px;font-weight:700}
.stTabs [aria-selected="true"]{color:#ffffff!important;background:#176b73!important}
.stTabs [data-baseweb="tab-highlight"]{background:#176b73!important}
.stTabs [data-baseweb="tab-panel"]{background:#ffffff;border:1px solid #e4eaf2;border-radius:0 0 14px 14px;padding:1rem 1.2rem;color:#17324d}
.stTabs [data-baseweb="tab-panel"] p{color:#17324d!important;line-height:1.65}
[data-testid="stMetric"]{background:#ffffff;border:1px solid #dce6f0;border-radius:16px;padding:.85rem 1rem;box-shadow:0 5px 18px #17324d12}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{color:#52677d!important;font-weight:600!important}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:#0b1f3a!important;font-weight:800!important}
[data-testid="stMetricDelta"],[data-testid="stMetricDelta"] *{color:#176b73!important}
div[data-testid="stMarkdownContainer"] p,div[data-testid="stMarkdownContainer"] strong{color:#17324d!important}
[data-testid="stExpander"]{border:2px solid #0b3b5a!important;border-radius:14px!important;overflow:hidden}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:active{background:#ffffff!important;color:#0b3b5a!important;font-weight:700}
[data-testid="stExpander"] summary svg{fill:#0b3b5a!important}
section[data-testid="stSidebar"] *{color:#17324d!important}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] textarea{color:#17324d!important;background:#ffffff!important}
section[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#ffffff!important;color:#17324d!important}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *{color:#17324d!important}
section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div,section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{background:#d9eef5!important}
section[data-testid="stSidebar"] .stForm,section[data-testid="stSidebar"] .stForm>div{background:transparent!important}
section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:#0b1f3a!important}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] small{color:#294b63!important}
section[data-testid="stSidebar"] [data-baseweb="input"]>div{background:transparent!important;border:1px solid #7fb4c5!important}
section[data-testid="stSidebar"] [data-baseweb="select"]>div{background:transparent!important;border:1px solid #7fb4c5!important}
section[data-testid="stSidebar"] [data-baseweb="select"] span{color:#17324d!important}
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"]{background:#176b73!important;border-color:#176b73!important}
@keyframes heroIn{from{opacity:0;transform:translateY(-16px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes softPulse{0%,100%{box-shadow:0 5px 20px #df5b6814}50%{box-shadow:0 8px 28px #df5b6840}}
@keyframes gradientFlow{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
@keyframes floatOrb{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(-18px,12px) scale(1.08)}}
@keyframes tabPop{0%{transform:scale(.96);opacity:.75}60%{transform:scale(1.03)}100%{transform:scale(1);opacity:1}}
.hero:after{content:"";position:absolute;width:180px;height:180px;border-radius:50%;right:8%;top:-80px;background:#73d5d133;filter:blur(2px);animation:floatOrb 6s ease-in-out infinite}
.hero{animation:heroIn .7s ease-out both}
.section-title,.stMetric,.stDataFrame,.stPlotlyChart,.element-container:has(.risk-high),.element-container:has(.risk-low){animation:fadeUp .55s ease-out both}
.stMetric{transition:transform .22s ease,box-shadow .22s ease;border-radius:14px;padding:.45rem}
.stMetric:hover{transform:translateY(-4px);box-shadow:0 9px 24px #17324d1c}
.risk-high{animation:fadeUp .55s ease-out both,softPulse 2.8s ease-in-out 1s infinite}
.risk-low{animation:fadeUp .55s ease-out both}
.stButton>button{transition:transform .2s ease,box-shadow .2s ease,background .2s ease}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 18px #176b7340}
.stTabs [data-baseweb="tab"]{transition:none!important;transform:none!important}
.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:focus,
.stTabs [data-baseweb="tab"]:active{transform:none!important;background:#ffffff;color:#17324d!important}
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"]:hover,
.stTabs [data-baseweb="tab"][aria-selected="true"]:focus{background:#176b73!important;color:#ffffff!important}
.stTabs button[aria-selected="true"]{background:#176b73!important;color:#ffffff!important;animation:tabPop .3s ease-out}
.stTabs button[aria-selected="true"],
.stTabs button[aria-selected="true"]>div,
.stTabs button[aria-selected="true"]>div>div{background:#176b73!important;color:#ffffff!important}
.stTabs button[aria-selected="true"] *{color:#ffffff!important;background:transparent!important}
.stTabs button:not([aria-selected="true"]),
.stTabs button:not([aria-selected="true"])>div,
.stTabs button:not([aria-selected="true"])>div>div{background:#ffffff!important;color:#17324d!important}
.stTabs button:not([aria-selected="true"]):hover{background:#f4fbff!important;color:#17324d!important;transform:none!important}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
</style>""",unsafe_allow_html=True)
if night_mode:
    st.markdown("""<style>
    .stApp{background:#081421;color:#e8f0f7}
    [data-testid="stHeader"],[data-testid="stSidebar"]{background:#0d2031}
    section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div,section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{background:#0d2031!important}
    section[data-testid="stSidebar"] .stForm,section[data-testid="stSidebar"] .stForm>div{background:transparent!important}
    [data-testid="stSidebar"] *{color:#e8f0f7!important}
    section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] small{color:#cfe4ef!important}
    section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] textarea{color:#e8f0f7!important;background:transparent!important}
    section[data-testid="stSidebar"] [data-baseweb="select"]>div{background:transparent!important;color:#e8f0f7!important;border:1px solid #37627c!important}
    section[data-testid="stSidebar"] [data-baseweb="input"]>div{background:transparent!important;border:1px solid #37627c!important}
    section[data-testid="stSidebar"] [data-baseweb="select"] span{color:#e8f0f7!important}
    section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"]{background:#49c6b8!important;border-color:#49c6b8!important}
    .section-title{color:#e8f0f7}.muted{color:#a9c0d1}
    [data-testid="stMetric"]{background:#10293b;border:1px solid #24465c;box-shadow:0 5px 18px #00000030}
    [data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{color:#a9c0d1!important}
    [data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:#f4fbff!important}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:active{background:#000000!important;color:#ffffff!important}
    [data-testid="stExpander"] summary svg{fill:#ffffff!important}
    .stTabs [data-baseweb="tab-list"]{background:#10293b}
    .stTabs [data-baseweb="tab"]{background:#18364a;color:#e8f0f7!important}
    .stTabs [data-baseweb="tab"]:hover,.stTabs [data-baseweb="tab"]:focus,.stTabs [data-baseweb="tab"]:active{background:#18364a!important;color:#e8f0f7!important;transform:none!important}
    .stTabs [aria-selected="true"]{background:#27a99d!important;color:#ffffff!important}
    .stTabs [data-baseweb="tab"][aria-selected="true"],.stTabs [data-baseweb="tab"][aria-selected="true"]:hover,.stTabs [data-baseweb="tab"][aria-selected="true"]:focus{background:#27a99d!important;color:#ffffff!important}
    .stTabs button[aria-selected="true"]{background:#27a99d!important;color:#ffffff!important;animation:tabPop .3s ease-out}
    .stTabs button[aria-selected="true"],
    .stTabs button[aria-selected="true"]>div,
    .stTabs button[aria-selected="true"]>div>div{background:#27a99d!important;color:#ffffff!important}
    .stTabs button[aria-selected="true"] *{color:#ffffff!important;background:transparent!important}
    .stTabs button:not([aria-selected="true"]),
    .stTabs button:not([aria-selected="true"])>div,
    .stTabs button:not([aria-selected="true"])>div>div{background:#18364a!important;color:#e8f0f7!important}
    .stTabs button:not([aria-selected="true"]):hover{background:#21465c!important;color:#e8f0f7!important;transform:none!important}
    .stTabs [data-baseweb="tab-panel"]{background:#10293b;border-color:#24465c;color:#e8f0f7}
    .stTabs [data-baseweb="tab-panel"] p{color:#e8f0f7!important}
    div[data-testid="stMarkdownContainer"] p,div[data-testid="stMarkdownContainer"] strong{color:#e8f0f7!important}
    </style>""",unsafe_allow_html=True)
st.markdown("""<style>
/* Sidebar is intentionally identical in light and night modes. */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"]>div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] .stForm,
section[data-testid="stSidebar"] .stForm>div{
    background:#123e63!important;
}
section[data-testid="stSidebar"] *{
    color:#ffffff!important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="input"]>div,
section[data-testid="stSidebar"] [data-baseweb="select"]>div{
    background:#05080c!important;
    color:#ffffff!important;
    border:1px solid #4b718c!important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] small{
    color:#ffffff!important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] *,
section[data-testid="stSidebar"] [data-baseweb="slider"] *{
    color:#ffffff!important;
}
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"]{
    background:#73d5d1!important;
    border-color:#73d5d1!important;
}
section[data-testid="stSidebar"] [data-baseweb="select"]{
    background:#05080c!important;
    border:1px solid #4b718c!important;
    border-radius:9px!important;
}
section[data-testid="stSidebar"] [data-baseweb="select"]>div,
section[data-testid="stSidebar"] [data-baseweb="select"] input{
    background:transparent!important;
    border:0!important;
    color:#ffffff!important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg{
    fill:#ffffff!important;
}
.stTabs [data-baseweb="tab-list"]{background:transparent!important}
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:focus,
.stTabs [data-baseweb="tab"]:active,
.stTabs button,
.stTabs button:hover,
.stTabs button:focus,
.stTabs button:active,
.stTabs button[aria-selected="true"],
.stTabs button:not([aria-selected="true"]),
.stTabs button[aria-selected="true"]>div,
.stTabs button[aria-selected="true"]>div>div,
.stTabs button:not([aria-selected="true"])>div,
.stTabs button:not([aria-selected="true"])>div>div{
    background:transparent!important;
    box-shadow:none!important;
    transform:none!important;
    color:#e8f0f7!important;
}
.stTabs button[aria-selected="true"]{animation:tabPop .3s ease-out}
.stTabs button *{background:transparent!important;color:#e8f0f7!important}
.stTabs [data-baseweb="tab-highlight"]{background:#49c6b8!important;height:3px!important}
.stTabs [role="tab"],
.stTabs [role="tab"]:hover,
.stTabs [role="tab"]:focus,
.stTabs [role="tab"]:active,
.stTabs button[data-baseweb="tab"],
.stTabs button[data-baseweb="tab"]:hover,
.stTabs button[data-baseweb="tab"]:focus,
.stTabs button[data-baseweb="tab"]:active{
    background-color:transparent!important;
    background-image:none!important;
    box-shadow:none!important;
    outline:none!important;
    border-color:transparent!important;
    color:#e8f0f7!important;
    -webkit-tap-highlight-color:transparent!important;
}
.stTabs [role="tab"][aria-selected="true"],
.stTabs [role="tab"][aria-selected="true"]:hover,
.stTabs [role="tab"][aria-selected="true"]:focus{
    background-color:transparent!important;
    color:#ffffff!important;
    animation:tabPop .3s ease-out;
}
</style>""",unsafe_allow_html=True)
st.markdown("""<style>
div[data-testid="stMarkdownContainer"] .hero,
div[data-testid="stMarkdownContainer"] .hero h1,
div[data-testid="stMarkdownContainer"] .hero p{
    color:#ffffff!important;
}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><span class="pill">MODEL-POWERED PREVENTIVE CARE</span><h1>HeartLens</h1><p>Explainable heart-disease risk intelligence for faster screening, smarter outreach, and more informed care conversations.</p></div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Patient profile"); st.caption("Enter a profile to generate an explainable risk estimate.")
    with st.form("patient_form"):
        age=st.slider("Age",18,90,54); sex=st.selectbox("Sex",["Male","Female"])
        chest=st.selectbox("Chest pain type",["Asymptomatic","Non-Anginal Pain","Atypical Angina","Typical Angina"])
        smoker=st.selectbox("Smoking status",["Never","Former","Current"]); family=st.checkbox("Family history of heart disease")
        angina=st.checkbox("Exercise-induced angina"); wearable=st.checkbox("Wearable owner"); st.markdown("### Vitals & lipids")
        sbp=st.number_input("Systolic BP",85,181,128); dbp=st.number_input("Diastolic BP",50,126,81)
        chol=st.number_input("Total cholesterol",90,314,189); hdl=st.number_input("HDL",18,110,55); ldl=st.number_input("LDL",35,207,103)
        trig=st.number_input("Triglycerides",35,390,152); sugar=st.number_input("Fasting blood sugar",60,204,119)
        hba1c=st.number_input("HbA1c",4.0,8.6,5.8,step=.1); bmi=st.number_input("BMI",15.0,43.3,25.3,step=.1)
        rhr=st.number_input("Resting heart rate",48,111,81); maxhr=st.number_input("Max heart rate achieved",93,210,165); std=st.number_input("ST depression",0.0,6.5,1.0,step=.1)
        st.markdown("### Lifestyle & context")
        alcohol=st.number_input("Alcohol units / week",0.0,45.9,5.7,step=.1); exercise=st.number_input("Exercise minutes / week",0,366,139)
        sleep=st.number_input("Sleep hours",3.1,11.0,7.0,step=.1); stress=st.number_input("Stress score",0.0,100.0,48.0,step=.1)
        steps=st.number_input("Daily steps",500,13950,6164); diet=st.number_input("Diet quality score",4.8,100.0,59.5,step=.1)
        submitted=st.form_submit_button("Analyze risk",type="primary",width="stretch")

def make_profile():
    profile = pd.DataFrame([{"age":age,"sex":sex,"resting_bp_systolic":sbp,"resting_bp_diastolic":dbp,"cholesterol_total":chol,"hdl":hdl,"ldl":ldl,"triglycerides":trig,"fasting_blood_sugar":sugar,"hba1c":hba1c,"bmi":bmi,"resting_heart_rate":rhr,"max_heart_rate_achieved":maxhr,"chest_pain_type":chest,"exercise_induced_angina":angina,"st_depression":std,"family_history":family,"smoker_status":smoker,"alcohol_units_per_week":alcohol,"exercise_minutes_per_week":exercise,"sleep_hours":sleep,"stress_score":stress,"wearable_owner":wearable,"daily_steps":steps,"diet_quality_score":diet}])
    profile["chest_pain_type"] = pd.Categorical(profile["chest_pain_type"], categories=["Asymptomatic","Non-Anginal Pain","Atypical Angina","Typical Angina"])
    return profile
if "profile" not in st.session_state or submitted: st.session_state.profile=make_profile()
profile=st.session_state.profile; probability=float(model.predict_proba(profile)[0,1]); flag=probability>=threshold

st.markdown('<div class="section-title">Risk command center</div>',unsafe_allow_html=True); st.markdown('<div class="muted">A probability estimate, operating threshold, and the evidence behind the prediction.</div>',unsafe_allow_html=True)
c1,c2,c3,c4=st.columns(4); c1.metric("Risk probability",f"{probability:.1%}"); c2.metric("Decision threshold",f"{threshold:.2f}"); c3.metric("Model PR-AUC",f"{artifact['metrics']['pr_auc']:.3f}"); c4.metric("Model version",artifact["model_version"])
if flag: st.markdown(f'<div class="risk-high"><b>Priority review flag</b><br>Estimated probability is {probability:.1%}, above the {threshold:.0%} operating threshold. Validate the measurements and consider timely clinical review.</div>',unsafe_allow_html=True)
else: st.markdown(f'<div class="risk-low"><b>Lower-risk model output</b><br>Estimated probability is {probability:.1%}, below the {threshold:.0%} operating threshold. Continue routine prevention and monitoring.</div>',unsafe_allow_html=True)

st.markdown('<div class="section-title">Risk intensity</div>',unsafe_allow_html=True)
st.progress(min(probability,1.0),text=f"{probability:.1%} estimated probability")
g1,g2,g3=st.columns(3)
g1.markdown("**🫀 Cardiac signals**<br><span class='muted'>Heart-rate and stress-test patterns</span>",unsafe_allow_html=True)
g2.markdown("**🧬 Metabolic signals**<br><span class='muted'>Lipids, glucose, BMI and blood pressure</span>",unsafe_allow_html=True)
g3.markdown("**🌿 Lifestyle signals**<br><span class='muted'>Exercise, smoking, sleep and activity</span>",unsafe_allow_html=True)

construction=model.named_steps["construction"]; prep=model.named_steps["preprocessing"]; selector=model.named_steps["selection"]; estimator=model.named_steps["estimator"]
selected=selector.transform(prep.transform(construction.transform(profile))); names=prep.get_feature_names_out()[selector.get_support()]
explainer=shap.TreeExplainer(estimator); values=explainer.shap_values(selected)
if isinstance(values,list): values=values[1]
values=np.asarray(values)[0]
shap_df=pd.DataFrame({"Feature":names,"Contribution":values,"Magnitude":np.abs(values)}).sort_values("Magnitude",ascending=False)
top=pd.concat([shap_df.nlargest(5,"Contribution"),shap_df.nsmallest(5,"Contribution")]).drop_duplicates().sort_values("Contribution")
left,right=st.columns([1.2,1])
with left:
    st.markdown('<div class="section-title">Why this score?</div>',unsafe_allow_html=True); st.caption("Positive contributions push risk higher; negative contributions push it lower.")
    fig,ax=plt.subplots(figsize=(8,4.8)); labels=top.Feature.str.replace("Numerical__","",regex=False).str.replace("Ohe__","",regex=False).str.replace("Ord__","",regex=False)
    ax.barh(labels,top.Contribution,color=["#df5b68" if x>0 else "#19a68d" for x in top.Contribution]); ax.axvline(0,color="#26364d",lw=1); ax.set_xlabel("SHAP contribution"); ax.grid(axis="x",alpha=.2); plt.tight_layout(); st.pyplot(fig,clear_figure=True)
with right:
    st.markdown('<div class="section-title">Actionable signals</div>',unsafe_allow_html=True)
    for _,row in top.sort_values("Contribution",ascending=False).head(6).iterrows():
        direction="raises" if row.Contribution>0 else "reduces"
        st.markdown(f"**{row.Feature.replace('Numerical__','').replace('Ohe__','').replace('Ord__','')}**  \n{direction.title()} the model score by {abs(row.Contribution):.3f} in this profile.")

with st.expander("Patient profile and model notes"):
    st.dataframe(profile.T.rename(columns={0:"Value"}).astype(str),width="stretch")
    st.markdown('<div class="disclaimer">HeartLens is a decision-support prototype, not a diagnosis. Do not start, stop, or change treatment based only on this output.</div>',unsafe_allow_html=True)
st.markdown('<div class="section-title">How teams can use this</div>',unsafe_allow_html=True)
t1,t2,t3=st.tabs(["Care teams","Population programs","Governance"])
with t1: st.write("Use the flag to prioritize review queues. Use SHAP contributors to validate data quality and guide conversations about lipids, blood pressure, metabolic health, exercise response, smoking, and family history.")
with t2: st.write("Aggregate patterns can support lipid-management, hypertension monitoring, diabetes-prevention, smoking-cessation, cardiac rehabilitation, and wellness outreach. Keep outreach voluntary and privacy-preserving.")
with t3: st.write("Recalibrate the threshold when referral capacity or the cost of missed cases changes. Monitor calibration, subgroup performance, false negatives, drift, privacy, and consent before production deployment.")
st.markdown("---"); st.caption("HeartLens • Explainable heart-disease risk intelligence • Supplied 2026 dataset")
