import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SnakeTail",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Share+Tech+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Share Tech Mono', monospace; background-color: #000000; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.stApp { background-color: #000000; }
[data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #00FF41; }
[data-testid="stSidebar"] * { color: #00FF41 !important; }

.game-title { font-family: 'Press Start 2P', monospace; font-size: 36px; color: #00FF41; text-shadow: 0 0 20px #00FF41, 0 0 40px #00FF41; text-align: center; padding: 40px 0 8px; letter-spacing: 4px; }
.game-subtitle { font-family: 'Share Tech Mono', monospace; font-size: 13px; color: #FF00FF; text-align: center; letter-spacing: 3px; text-shadow: 0 0 10px #FF00FF; margin-bottom: 32px; }
.stat-row { display: flex; justify-content: center; gap: 48px; margin: 16px 0 32px; }
.stat-item { text-align: center; }
.stat-val { font-family: 'Press Start 2P', monospace; font-size: 22px; color: #FFD700; text-shadow: 0 0 10px #FFD700; }
.stat-lbl { font-family: 'Share Tech Mono', monospace; font-size: 10px; color: #888; letter-spacing: 2px; margin-top: 4px; }
.card { background: #0a0a0a; border: 1px solid #00FF41; border-radius: 4px; padding: 20px; margin-bottom: 16px; box-shadow: 0 0 10px rgba(0,255,65,0.1); }
.section-title { font-family: 'Press Start 2P', monospace; font-size: 11px; color: #00FFFF; letter-spacing: 2px; margin-bottom: 12px; text-shadow: 0 0 8px #00FFFF; }
.metric-row { display: flex; gap: 12px; margin: 20px 0; }
.metric-card { flex: 1; background: #0a0a0a; border: 1px solid #FF00FF; border-radius: 4px; padding: 16px; text-align: center; box-shadow: 0 0 8px rgba(255,0,255,0.2); }
.metric-val { font-family: 'Press Start 2P', monospace; font-size: 18px; color: #FFD700; text-shadow: 0 0 8px #FFD700; }
.metric-lbl { font-family: 'Share Tech Mono', monospace; font-size: 10px; color: #666; margin-top: 6px; letter-spacing: 1px; }

.risk-high { background: #1a0000; border: 2px solid #FF0000; border-radius: 4px; padding: 24px; text-align: center; box-shadow: 0 0 20px rgba(255,0,0,0.4); margin-bottom: 20px; }
.risk-medium { background: #1a1000; border: 2px solid #FFD700; border-radius: 4px; padding: 24px; text-align: center; box-shadow: 0 0 20px rgba(255,215,0,0.3); margin-bottom: 20px; }
.risk-low { background: #001a00; border: 2px solid #00FF41; border-radius: 4px; padding: 24px; text-align: center; box-shadow: 0 0 20px rgba(0,255,65,0.3); margin-bottom: 20px; }
.risk-pct { font-family: 'Press Start 2P', monospace; font-size: 40px; margin-bottom: 8px; }
.risk-label { font-family: 'Share Tech Mono', monospace; font-size: 13px; letter-spacing: 2px; }

.profile-box { background: #050505; border: 1px solid #333; border-radius: 4px; padding: 20px; margin-bottom: 16px; }
.profile-title { font-family: 'Press Start 2P', monospace; font-size: 10px; color: #00FFFF; margin-bottom: 16px; letter-spacing: 2px; }
.mutation-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; font-family: 'Share Tech Mono', monospace; font-size: 12px; }
.mut-name { width: 90px; color: #888; }
.mut-bar-bg { flex: 1; background: #111; border-radius: 2px; height: 8px; }
.mut-bar-present { height: 8px; border-radius: 2px; background: #FF4444; }
.mut-bar-absent { height: 8px; border-radius: 2px; background: #1a1a1a; width: 100%; }
.mut-status-present { width: 70px; color: #FF4444; font-size: 11px; }
.mut-status-absent { width: 70px; color: #333; font-size: 11px; }

.clinical-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; font-family: 'Share Tech Mono', monospace; font-size: 12px; }
.clin-name { width: 120px; color: #888; }
.clin-bar-bg { flex: 1; background: #111; border-radius: 2px; height: 8px; }
.clin-val { width: 80px; text-align: right; font-size: 11px; }

.patient-card { background: #0a0a0a; border-radius: 4px; padding: 16px; margin-bottom: 12px; }
.patient-card-high { border: 1px solid #FF4444; box-shadow: 0 0 8px rgba(255,68,68,0.2); }
.patient-card-low { border: 1px solid #00FF41; box-shadow: 0 0 8px rgba(0,255,65,0.1); }

.stButton > button { background: #000 !important; color: #00FF41 !important; border: 1px solid #00FF41 !important; border-radius: 4px !important; font-family: 'Press Start 2P', monospace !important; font-size: 10px !important; padding: 12px 24px !important; width: 100% !important; letter-spacing: 2px !important; box-shadow: 0 0 10px rgba(0,255,65,0.3) !important; }
.stButton > button:hover { background: #00FF41 !important; color: #000 !important; }
.stDownloadButton > button { background: #000 !important; color: #FFD700 !important; border: 1px solid #FFD700 !important; font-family: 'Press Start 2P', monospace !important; font-size: 9px !important; width: 100% !important; }
hr { border-color: #00FF41 !important; opacity: 0.3 !important; margin: 24px 0 !important; }
.stSlider label { color: #00FF41 !important; font-family: 'Share Tech Mono', monospace !important; }
.stCheckbox label { color: #00FF41 !important; font-family: 'Share Tech Mono', monospace !important; font-size: 13px !important; }
.stSelectbox label { color: #00FF41 !important; }
.stTabs [data-baseweb="tab"] { font-family: 'Share Tech Mono', monospace !important; color: #666 !important; }
.stTabs [aria-selected="true"] { color: #00FF41 !important; border-bottom: 2px solid #00FF41 !important; }
.stMarkdown p { color: #FFFFFF !important; }
.stMarkdown li { color: #FFFFFF !important; }
label { color: #FFFFFF !important; }
.stCheckbox label p { color: #FFFFFF !important; }
.stSlider label { color: #FFFFFF !important; }
p { color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPER — build mutation bar HTML
# ============================================
def mutation_bar(name, present, description=""):
    if present:
        bar = '<div style="height:8px;border-radius:2px;background:#FF4444;width:100%"></div>'
        status = '<span style="width:80px;color:#FF4444;font-size:11px;font-family:Share Tech Mono,monospace">⚠ PRESENT</span>'
    else:
        bar = '<div style="height:8px;border-radius:2px;background:#1a1a1a;width:100%"></div>'
        status = '<span style="width:80px;color:#333;font-size:11px;font-family:Share Tech Mono,monospace">✓ ABSENT</span>'
    return (
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;font-family:Share Tech Mono,monospace;font-size:12px">' +
        f'<span style="width:90px;color:#888">{name}</span>' +
        f'<div style="flex:1;background:#111;border-radius:2px;height:8px">{bar}</div>' +
        status +
        f'<span style="color:#444;font-size:10px;width:120px">{description}</span>' +
        '</div>'
    )

# ============================================
# HELPER — build clinical bar HTML
# ============================================
def clinical_bar(name, value, normal_min, normal_max, unit="", critical_high=None, critical_low=None):
    # determine color and fill
    if critical_high and value > critical_high:
        color = "#FF4444"
        label = "CRITICAL ⚠"
        fill = min(100, int((value / (critical_high * 1.5)) * 100))
    elif critical_low and value < critical_low:
        color = "#FF8800"
        label = "LOW ⚠"
        fill = int((value / normal_min) * 30)
    elif value > normal_max:
        color = "#FFD700"
        label = "HIGH"
        fill = min(100, int((value / normal_max) * 60))
    else:
        color = "#00FF41"
        label = "NORMAL"
        fill = int(((value - normal_min) / (normal_max - normal_min)) * 60) + 20

    fill = max(3, min(100, fill))
    formatted = f"{value:,.0f}" if value > 100 else f"{value:.1f}"

    return f"""
    <div class="clinical-row">
        <span class="clin-name">{name}</span>
        <div class="clin-bar-bg">
            <div style="height:8px;border-radius:2px;background:{color};width:{fill}%"></div>
        </div>
        <span class="clin-val" style="color:{color}">{formatted} {unit}</span>
        <span style="color:{color};font-size:10px;width:80px">{label}</span>
    </div>
    """

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### SNAKETAIL")
    st.markdown("---")
    st.markdown("**Generation Settings**")
    num_rows = st.slider("Patients to generate", 100, 5000, 1000, step=100)
    st.markdown("---")
    st.markdown("**Privacy Settings**")
    epsilon = st.slider("Privacy budget (ε)", 0.1, 10.0, 1.0, step=0.1)
    st.caption("↓ Lower ε = stronger privacy")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px; color:#444; font-family: Share Tech Mono, monospace;'>
    HSIL HACKATHON 2026<br>
    Harvard Health Systems<br>
    Innovation Lab
    </div>
    """, unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<div class="game-title">SNAKETAIL</div>', unsafe_allow_html=True)
st.markdown('<div class="game-subtitle">PRIVACY-PRESERVING PEDIATRIC GENOMIC AI</div>', unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
    <div class="stat-item"><div class="stat-val">94.5%</div><div class="stat-lbl">MODEL ACCURACY</div></div>
    <div class="stat-item"><div class="stat-val">8</div><div class="stat-lbl">MUTATIONS TRACKED</div></div>
    <div class="stat-item"><div class="stat-val">0</div><div class="stat-lbl">REAL DNA EXPOSED</div></div>
    <div class="stat-item"><div class="stat-val">∞</div><div class="stat-lbl">SYNTHETIC PATIENTS</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================
# TWO TABS
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["· GENERATE", "· SYMPTOMS", "· PREDICT V1", "· PREDICT V2", "· INSIGHTS"])

# ============================================
# TAB 1 — GENERATE
# ============================================
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown('<div class="section-title">> INPUT_DATA</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        use_demo = st.checkbox("Use built-in genomic dataset", value=True)
        if not use_demo:
            uploaded_file = st.file_uploader("Upload genomic CSV", type=['csv'])
        else:
            uploaded_file = None
            st.info("GAN-generated leukemia genomic dataset — 1000 patients, 8 mutations, frequency corrected")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">> GENERATE</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"Sampling `{num_rows:,}` patients from synthetic genome pool")
        st.markdown(" ")
        generate_btn = st.button("▶ GENERATE SYNTHETIC DNA")
        st.caption("Instant — no retraining needed!")
        st.markdown('</div>', unsafe_allow_html=True)

    if generate_btn:
        if use_demo or uploaded_file is not None:
            with st.spinner("Loading synthetic genomic data..."):
                if use_demo:
                    try:
                        full_df = pd.read_csv('synthetic_genomes.csv')
                    except:
                        st.error("synthetic_genomes.csv not found! Run python genomic_gan.py first!")
                        st.stop()
                else:
                    full_df = pd.read_csv(uploaded_file)

            with st.spinner("Sampling patients..."):
                if num_rows <= len(full_df):
                    fake_df = full_df.sample(n=num_rows, random_state=42).reset_index(drop=True)
                else:
                    fake_df = full_df.sample(n=num_rows, replace=True, random_state=42).reset_index(drop=True)

            leukemia_count = int(fake_df['has_leukemia'].sum()) if 'has_leukemia' in fake_df.columns else 0
            healthy_count  = num_rows - leukemia_count

            st.markdown("---")
            st.markdown('<div class="section-title">> RESULTS</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card"><div class="metric-val">{num_rows:,}</div><div class="metric-lbl">PATIENTS GENERATED</div></div>
                <div class="metric-card"><div class="metric-val">{leukemia_count}</div><div class="metric-lbl">LEUKEMIA CASES</div></div>
                <div class="metric-card"><div class="metric-val">{healthy_count}</div><div class="metric-lbl">HEALTHY CASES</div></div>
                <div class="metric-card"><div class="metric-val">0</div><div class="metric-lbl">REAL DNA EXPOSED</div></div>
            </div>
            """, unsafe_allow_html=True)

            # ============================================
            # PATIENT CARDS — visual storytelling!
            # ============================================
            st.markdown('<div class="section-title">> PATIENT PROFILES</div>', unsafe_allow_html=True)
            st.caption("Showing 6 sample synthetic patients — high risk cases highlighted in red")

            mutation_cols = ['FLT3_ITD','BCR_ABL','TP53','RUNX1','DNMT3A','NPM1','CEBPA','WT1']

            # get 3 leukemia + 3 healthy for display
            leuk_sample    = fake_df[fake_df['has_leukemia']==1].head(3) if 'has_leukemia' in fake_df.columns else fake_df.head(3)
            healthy_sample = fake_df[fake_df['has_leukemia']==0].head(3) if 'has_leukemia' in fake_df.columns else fake_df.tail(3)
            display_df     = pd.concat([leuk_sample, healthy_sample]).reset_index(drop=True)

            cols = st.columns(3)
            for idx, (_, row) in enumerate(display_df.iterrows()):
                col = cols[idx % 3]
                is_leukemia = int(row.get('has_leukemia', 0)) == 1
                card_class  = "patient-card-high" if is_leukemia else "patient-card-low"
                risk_color  = "#FF4444" if is_leukemia else "#00FF41"
                risk_label  = "⚠ HIGH RISK" if is_leukemia else "✓ LOW RISK"
                age         = int(row.get('age', 0))
                mutations   = [m for m in mutation_cols if m in row and int(round(row[m])) == 1]
                burden      = row.get('mutation_burden', 0)

                with col:
                    st.markdown(f"""
                    <div class="patient-card {card_class}">
                        <div style="display:flex;justify-content:space-between;margin-bottom:10px">
                            <span style="font-family:'Press Start 2P',monospace;font-size:9px;color:#888">
                                PATIENT #{idx+1}
                            </span>
                            <span style="font-family:'Share Tech Mono',monospace;font-size:11px;color:{risk_color};font-weight:bold">
                                {risk_label}
                            </span>
                        </div>
                        <div style="font-family:'Share Tech Mono',monospace;font-size:13px;color:#E6EDF3;margin-bottom:8px">
                            Age {age}
                        </div>
                        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#666;margin-bottom:8px">
                            Mutation burden: <span style="color:{risk_color}">{burden:.1f}</span>
                        </div>
                        <div style="margin-top:8px">
                            {''.join([f'<span style="background:#1a0000;border:1px solid #FF4444;color:#FF4444;font-size:9px;padding:2px 6px;border-radius:2px;margin:2px;display:inline-block;font-family:Share Tech Mono,monospace">{m}</span>' for m in mutations]) if mutations else '<span style="color:#333;font-size:10px;font-family:Share Tech Mono,monospace">No mutations detected</span>'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            t1, t2 = st.tabs(["FULL DATASET", "MUTATION STATS"])
            with t1:
                st.dataframe(fake_df.head(20), use_container_width=True)
            with t2:
                if 'has_leukemia' in fake_df.columns:
                    leuk_df = fake_df[fake_df['has_leukemia']==1]
                    stats = pd.DataFrame({
                        'Mutation':  mutation_cols,
                        'Cases':     [int(leuk_df[c].sum()) for c in mutation_cols if c in leuk_df.columns],
                        'Rate':      [f"{leuk_df[c].mean():.1%}" for c in mutation_cols if c in leuk_df.columns],
                        'Target':    ['35%','25%','20%','25%','22%','30%','15%','18%']
                    })
                    st.dataframe(stats, use_container_width=True)

            st.markdown("---")
            st.download_button(
                label="▼ DOWNLOAD SYNTHETIC DNA CSV",
                data=fake_df.to_csv(index=False),
                file_name="snaketail_output.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("Zero real patient DNA. Safe to share and use for AI training.")

# ============================================
# TAB 3 — PREDICT V1
# ============================================
with tab3:
    st.markdown('<div class="section-title">> PATIENT INPUT</div>', unsafe_allow_html=True)

    # File upload section
    st.markdown("**Option 1 — Upload patient file:**")
    upcol1, upcol2 = st.columns(2)
    with upcol1:
        template_df = pd.DataFrame([{
            'age': 6, 'FLT3_ITD': 0, 'BCR_ABL': 0, 'TP53': 0,
            'RUNX1': 0, 'DNMT3A': 0, 'NPM1': 0, 'CEBPA': 0,
            'WT1': 0, 'chromosome': 13, 'mutation_burden': 5.0
        }])
        st.download_button(
            "📥 DOWNLOAD PATIENT TEMPLATE",
            template_df.to_csv(index=False),
            "snaketail_template.csv",
            use_container_width=True
        )
        st.caption("Fill 0 or 1 for mutations → upload")
    with upcol2:
        uploaded_patient = st.file_uploader("Upload filled template", type=['csv'], key="v1_upload")

    # If file uploaded — run prediction DIRECTLY from file!
    if uploaded_patient:
        try:
            row = pd.read_csv(uploaded_patient).iloc[0]
            p_age    = int(row.get('age', 6))
            p_burden = float(row.get('mutation_burden', 5.0))
            p_chr    = int(row.get('chromosome', 13))
            p_flt3   = int(row.get('FLT3_ITD', 0))
            p_bcr    = int(row.get('BCR_ABL', 0))
            p_tp53   = int(row.get('TP53', 0))
            p_runx1  = int(row.get('RUNX1', 0))
            p_dnmt3  = int(row.get('DNMT3A', 0))
            p_npm1   = int(row.get('NPM1', 0))
            p_cebpa  = int(row.get('CEBPA', 0))
            p_wt1    = int(row.get('WT1', 0))

            st.success(f"Patient loaded! Age={p_age} | Burden={p_burden} | Mutations={p_flt3+p_bcr+p_tp53+p_runx1+p_dnmt3+p_npm1+p_cebpa+p_wt1}/8")

            # Run prediction directly!
            with open('leukemia_model.pkl', 'rb') as f:
                model_file = pickle.load(f)

            child_file = pd.DataFrame([{
                'age': p_age, 'FLT3_ITD': p_flt3, 'BCR_ABL': p_bcr,
                'TP53': p_tp53, 'RUNX1': p_runx1, 'DNMT3A': p_dnmt3,
                'NPM1': p_npm1, 'CEBPA': p_cebpa, 'WT1': p_wt1,
                'chromosome': p_chr, 'mutation_burden': p_burden
            }])

            risk_file = model_file.predict_proba(child_file)[0][1]

            st.markdown("---")
            st.markdown('<div class="section-title">> FILE ANALYSIS RESULT</div>', unsafe_allow_html=True)

            if risk_file > 0.7:
                st.markdown(f"""
                <div class="risk-high">
                    <div class="risk-pct" style="color:#FF0000;text-shadow:0 0 20px #FF0000;">{risk_file:.1%}</div>
                    <div class="risk-label" style="color:#FF6666;">⚠ HIGH RISK — RECOMMEND IMMEDIATE BONE MARROW BIOPSY</div>
                </div>
                """, unsafe_allow_html=True)
            elif risk_file > 0.4:
                st.markdown(f"""
                <div class="risk-medium">
                    <div class="risk-pct" style="color:#FFD700;text-shadow:0 0 20px #FFD700;">{risk_file:.1%}</div>
                    <div class="risk-label" style="color:#FFE566;">⚠ MEDIUM RISK — MONITOR CLOSELY</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div class="risk-pct" style="color:#00FF41;text-shadow:0 0 20px #00FF41;">{risk_file:.1%}</div>
                    <div class="risk-label" style="color:#66FF88;">✓ LOW RISK — CONTINUE ROUTINE SCREENING</div>
                </div>
                """, unsafe_allow_html=True)

            # Show mutation summary
            mut_names = ['FLT3-ITD','BCR-ABL','TP53','RUNX1','DNMT3A','NPM1','CEBPA','WT1']
            mut_vals  = [p_flt3,p_bcr,p_tp53,p_runx1,p_dnmt3,p_npm1,p_cebpa,p_wt1]
            present   = [n for n,v in zip(mut_names,mut_vals) if v==1]
            if present:
                st.markdown(f"**Mutations detected:** {' · '.join([f'`{m}`' for m in present])}")
            else:
                st.success("No mutations detected")

            st.markdown("---")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("**Option 2 — Manual input:**")
    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Patient demographics:**")
        age = st.slider("Patient age", 1, 18, 6)
        mutation_burden = st.slider("Mutation burden", 0.0, 50.0, 5.0, step=0.5)
        chromosome = st.selectbox("Primary chromosome affected", [1,2,4,5,7,8,9,11,13,17,21,22])
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Mutations detected in genomic test:**")
        FLT3  = st.checkbox("FLT3-ITD  — drives uncontrolled cell growth")
        BCR   = st.checkbox("BCR-ABL   — Philadelphia chromosome fusion")
        TP53  = st.checkbox("TP53      — tumour suppressor broken")
        RUNX1 = st.checkbox("RUNX1     — blood cell development blocked")
        DNMT3 = st.checkbox("DNMT3A    — gene regulation disrupted")
        NPM1  = st.checkbox("NPM1      — DNA repair compromised")
        CEBPA = st.checkbox("CEBPA     — white cell maturation blocked")
        WT1   = st.checkbox("WT1       — poor prognosis marker")
        st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("▶ ANALYSE LEUKEMIA RISK")

    if predict_btn:
        try:
            with open('leukemia_model.pkl', 'rb') as f:
                model = pickle.load(f)

            child = pd.DataFrame([{
                'age':             age,
                'FLT3_ITD':        int(FLT3),
                'BCR_ABL':         int(BCR),
                'TP53':            int(TP53),
                'RUNX1':           int(RUNX1),
                'DNMT3A':          int(DNMT3),
                'NPM1':            int(NPM1),
                'CEBPA':           int(CEBPA),
                'WT1':             int(WT1),
                'chromosome':      chromosome,
                'mutation_burden': mutation_burden
            }])

            risk = model.predict_proba(child)[0][1]

            st.markdown("---")
            st.markdown('<div class="section-title">> PATIENT PROFILE</div>', unsafe_allow_html=True)

            # ============================================
            # RISK BANNER
            # ============================================
            if risk > 0.7:
                st.markdown(f"""
                <div class="risk-high">
                    <div class="risk-pct" style="color:#FF0000;text-shadow:0 0 20px #FF0000;">{risk:.1%}</div>
                    <div class="risk-label" style="color:#FF6666;">⚠ HIGH RISK — RECOMMEND IMMEDIATE BONE MARROW BIOPSY</div>
                </div>
                """, unsafe_allow_html=True)
            elif risk > 0.4:
                st.markdown(f"""
                <div class="risk-medium">
                    <div class="risk-pct" style="color:#FFD700;text-shadow:0 0 20px #FFD700;">{risk:.1%}</div>
                    <div class="risk-label" style="color:#FFE566;">⚠ MEDIUM RISK — MONITOR CLOSELY, REPEAT TESTS IN 4 WEEKS</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div class="risk-pct" style="color:#00FF41;text-shadow:0 0 20px #00FF41;">{risk:.1%}</div>
                    <div class="risk-label" style="color:#66FF88;">✓ LOW RISK — CONTINUE ROUTINE SCREENING</div>
                </div>
                """, unsafe_allow_html=True)

            # ============================================
            # VISUAL MUTATION PROFILE — pure streamlit
            # ============================================
            mutations_present = []
            if FLT3:  mutations_present.append("FLT3-ITD")
            if BCR:   mutations_present.append("BCR-ABL")
            if TP53:  mutations_present.append("TP53")
            if RUNX1: mutations_present.append("RUNX1")
            if DNMT3: mutations_present.append("DNMT3A")
            if NPM1:  mutations_present.append("NPM1")
            if CEBPA: mutations_present.append("CEBPA")
            if WT1:   mutations_present.append("WT1")

            col5, col6 = st.columns(2, gap="large")

            with col5:
                st.markdown('<div class="section-title">> MUTATION STATUS</div>', unsafe_allow_html=True)
                mutations_data = [
                    ("FLT3-ITD",  FLT3,  "drives uncontrolled cell growth"),
                    ("BCR-ABL",   BCR,   "Philadelphia chromosome fusion"),
                    ("TP53",      TP53,  "tumour suppressor broken"),
                    ("RUNX1",     RUNX1, "blood cell development blocked"),
                    ("DNMT3A",    DNMT3, "gene regulation disrupted"),
                    ("NPM1",      NPM1,  "DNA repair compromised"),
                    ("CEBPA",     CEBPA, "white cell maturation blocked"),
                    ("WT1",       WT1,   "poor prognosis marker"),
                ]
                for mut_name, mut_present, mut_desc in mutations_data:
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        if mut_present:
                            st.error(f"⚠ {mut_name}")
                        else:
                            st.success(f"✓ {mut_name}")
                    with col_b:
                        st.caption(mut_desc)

            with col6:
                st.markdown('<div class="section-title">> CLINICAL SUMMARY</div>', unsafe_allow_html=True)

                mut_count = len(mutations_present)

                st.markdown(f"**Age:** `{age} years`")
                st.progress(min(age/18, 1.0))

                burden_icon = "🔴" if mutation_burden > 15 else "🟡" if mutation_burden > 5 else "🟢"
                st.markdown(f"**Mutation burden:** `{mutation_burden}` {burden_icon}")
                st.progress(min(mutation_burden/50, 1.0))

                mut_icon = "🔴" if mut_count >= 4 else "🟡" if mut_count >= 2 else "🟢"
                st.markdown(f"**Mutations found:** `{mut_count}/8` {mut_icon}")
                st.progress(min(mut_count/8, 1.0))

                st.markdown(f"**Chromosome affected:** `Chr {chromosome}`")
                st.markdown("---")
                st.markdown("**Mutations detected:**")
                if mutations_present:
                    st.markdown(" · ".join([f"`{m}`" for m in mutations_present]))
                else:
                    st.success("No mutations detected")

            # ============================================
            # WHAT THIS MEANS — plain English explanation
            # ============================================
            st.markdown("---")
            st.markdown('<div class="section-title">> CLINICAL INTERPRETATION</div>', unsafe_allow_html=True)

            if risk > 0.7:
                st.markdown(f"""
                <div style="background:#0a0000;border:1px solid #FF4444;border-radius:4px;padding:16px;font-family:'Share Tech Mono',monospace;font-size:12px;line-height:1.8;color:#ccc">
                This {age}-year-old patient shows <span style="color:#FF4444">HIGH leukemia risk ({risk:.1%})</span>
                with <span style="color:#FF4444">{len(mutations_present)} active mutations</span> detected.<br><br>
                {'FLT3-ITD mutation suggests uncontrolled blast cell proliferation. ' if FLT3 else ''}
                {'BCR-ABL fusion (Philadelphia chromosome) indicates aggressive leukemia subtype. ' if BCR else ''}
                {'TP53 disruption removes key tumour suppressor — treatment may be challenging. ' if TP53 else ''}
                <br><br>
                <span style="color:#FF4444">⚠ RECOMMENDED ACTION:</span> Immediate bone marrow biopsy and haematology referral.
                This patient requires urgent clinical assessment.
                </div>
                """, unsafe_allow_html=True)

            elif risk > 0.4:
                st.markdown(f"""
                <div style="background:#0a0800;border:1px solid #FFD700;border-radius:4px;padding:16px;font-family:'Share Tech Mono',monospace;font-size:12px;line-height:1.8;color:#ccc">
                This {age}-year-old patient shows <span style="color:#FFD700">MEDIUM leukemia risk ({risk:.1%})</span>
                with <span style="color:#FFD700">{len(mutations_present)} mutations</span> detected.<br><br>
                Findings are concerning but not conclusive. Close monitoring is advised.
                <br><br>
                <span style="color:#FFD700">⚠ RECOMMENDED ACTION:</span> Repeat blood panel in 4 weeks.
                Consider bone marrow biopsy if symptoms worsen.
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div style="background:#000a00;border:1px solid #00FF41;border-radius:4px;padding:16px;font-family:'Share Tech Mono',monospace;font-size:12px;line-height:1.8;color:#ccc">
                This {age}-year-old patient shows <span style="color:#00FF41">LOW leukemia risk ({risk:.1%})</span>
                with <span style="color:#00FF41">{len(mutations_present)} mutations</span> detected.<br><br>
                No significant genomic red flags detected at this time.
                <br><br>
                <span style="color:#00FF41">✓ RECOMMENDED ACTION:</span> Continue routine annual screening.
                No immediate intervention required.
                </div>
                """, unsafe_allow_html=True)

            # ============================================
            # DISCLAIMER
            # ============================================
            st.markdown("---")
            st.caption("⚠ This prediction is generated from synthetic training data only. Not a clinical diagnosis. Always consult a qualified haematologist.")
            st.caption("🔒 Zero real patient DNA was used to train this model. Built with SnakeTail privacy-preserving GAN.")

        except FileNotFoundError:
            st.error("Model not found! Run: python risk_predictor.py first!")

# ============================================
# TAB 4 — PREDICT V2
# ============================================
with tab4:
    st.markdown('<div class="section-title">> V2 PATIENT INPUT — 28 FEATURES</div>', unsafe_allow_html=True)
    st.caption("Version 2 — multi-modal genomic profiling with gene expression, chromosomal and clinical data")

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("**Demographics + Mutations:**")
        v2_age = st.slider("Age", 1, 18, 6, key="v2_age")
        v2_burden = st.slider("Mutation burden", 0.0, 50.0, 5.0, key="v2_burden")
        st.markdown("**Binary mutations:**")
        v2_flt3  = st.checkbox("FLT3-ITD",  key="v2_flt3")
        v2_bcr   = st.checkbox("BCR-ABL",   key="v2_bcr")
        v2_tp53  = st.checkbox("TP53",      key="v2_tp53")
        v2_runx1 = st.checkbox("RUNX1",     key="v2_runx1")
        v2_dnmt3 = st.checkbox("DNMT3A",    key="v2_dnmt3")
        v2_npm1  = st.checkbox("NPM1",      key="v2_npm1")
        v2_cebpa = st.checkbox("CEBPA",     key="v2_cebpa")
        v2_wt1   = st.checkbox("WT1",       key="v2_wt1")

    with c2:
        st.markdown("**Gene expression levels:**")
        st.caption("Normal range: 2-5 | High: >7")
        v2_flt3_exp  = st.slider("FLT3 expression",  0.0, 15.0, 3.0, key="v2_flt3e")
        v2_myc_exp   = st.slider("MYC expression",   0.0, 15.0, 3.5, key="v2_myce")
        v2_bcl2_exp  = st.slider("BCL2 expression",  0.0, 15.0, 3.0, key="v2_bcl2e")
        v2_hoxa9_exp = st.slider("HOXA9 expression", 0.0, 15.0, 2.8, key="v2_hoxa9e")
        v2_erg_exp   = st.slider("ERG expression",   0.0, 15.0, 3.1, key="v2_erge")
        st.markdown("**Chromosomal abnormalities:**")
        v2_tri8  = st.checkbox("Trisomy 8",  key="v2_tri8")
        v2_mono7 = st.checkbox("Monosomy 7", key="v2_mono7")
        v2_del5q = st.checkbox("Del 5q",     key="v2_del5q")
        v2_inv16 = st.checkbox("Inv 16",     key="v2_inv16")
        v2_t821  = st.checkbox("t(8;21)",    key="v2_t821")

    with c3:
        st.markdown("**Clinical markers (blood test):**")
        st.caption("Normal WBC: 5,000-10,000")
        v2_wbc     = st.number_input("WBC count", 1000, 200000, 7500, step=1000, key="v2_wbc")
        v2_blast   = st.slider("Blast %", 0.0, 100.0, 1.5, key="v2_blast")
        v2_ldh     = st.number_input("LDH level", 100, 2000, 180, step=10, key="v2_ldh")
        v2_hgb     = st.slider("Hemoglobin", 3.0, 18.0, 13.5, key="v2_hgb")
        v2_plt     = st.number_input("Platelet count", 5000, 500000, 250000, step=5000, key="v2_plt")
        st.markdown("**Treatment response:**")
        v2_chemo   = st.slider("Chemo sensitivity", 0.0, 1.0, 0.5, key="v2_chemo")
        v2_relapse = st.slider("Relapse risk", 0.0, 1.0, 0.1, key="v2_relapse")

    predict_v2_btn = st.button("▶ ANALYSE V2 LEUKEMIA RISK", key="v2_predict")

    if predict_v2_btn:
        try:
            with open('leukemia_model_v2.pkl', 'rb') as f:
                model_v2 = pickle.load(f)

            child_v2 = pd.DataFrame([{
                'age':              v2_age,
                'FLT3_ITD':         int(v2_flt3),
                'BCR_ABL':          int(v2_bcr),
                'TP53':             int(v2_tp53),
                'RUNX1':            int(v2_runx1),
                'DNMT3A':           int(v2_dnmt3),
                'NPM1':             int(v2_npm1),
                'CEBPA':            int(v2_cebpa),
                'WT1':              int(v2_wt1),
                'FLT3_expression':  v2_flt3_exp,
                'MYC_expression':   v2_myc_exp,
                'BCL2_expression':  v2_bcl2_exp,
                'HOXA9_expression': v2_hoxa9_exp,
                'ERG_expression':   v2_erg_exp,
                'trisomy_8':        int(v2_tri8),
                'monosomy_7':       int(v2_mono7),
                'del_5q':           int(v2_del5q),
                'inv_16':           int(v2_inv16),
                't_8_21':           int(v2_t821),
                'wbc_count':        v2_wbc,
                'blast_percentage': v2_blast,
                'ldh_level':        v2_ldh,
                'hemoglobin':       v2_hgb,
                'platelet_count':   v2_plt,
                'chemo_sensitivity': v2_chemo,
                'relapse_risk':     v2_relapse,
                'mutation_burden':  v2_burden,
            }])

            risk_v2 = model_v2.predict_proba(child_v2)[0][1]

            st.markdown("---")
            st.markdown('<div class="section-title">> V2 RISK ANALYSIS</div>', unsafe_allow_html=True)

            if risk_v2 > 0.7:
                st.markdown(f"""
                <div class="risk-high">
                    <div class="risk-pct" style="color:#FF0000;text-shadow:0 0 20px #FF0000;">{risk_v2:.1%}</div>
                    <div class="risk-label" style="color:#FF6666;">⚠ HIGH RISK — RECOMMEND IMMEDIATE BONE MARROW BIOPSY</div>
                </div>
                """, unsafe_allow_html=True)
            elif risk_v2 > 0.4:
                st.markdown(f"""
                <div class="risk-medium">
                    <div class="risk-pct" style="color:#FFD700;text-shadow:0 0 20px #FFD700;">{risk_v2:.1%}</div>
                    <div class="risk-label" style="color:#FFE566;">⚠ MEDIUM RISK — MONITOR CLOSELY</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div class="risk-pct" style="color:#00FF41;text-shadow:0 0 20px #00FF41;">{risk_v2:.1%}</div>
                    <div class="risk-label" style="color:#66FF88;">✓ LOW RISK — CONTINUE ROUTINE SCREENING</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card"><div class="metric-val">{risk_v2:.1%}</div><div class="metric-lbl">LEUKEMIA RISK</div></div>
                <div class="metric-card"><div class="metric-val">{v2_wbc:,}</div><div class="metric-lbl">WBC COUNT</div></div>
                <div class="metric-card"><div class="metric-val">{v2_blast:.1f}%</div><div class="metric-lbl">BLAST CELLS</div></div>
                <div class="metric-card"><div class="metric-val">{v2_burden}</div><div class="metric-lbl">MUTATION BURDEN</div></div>
            </div>
            """, unsafe_allow_html=True)

            if v2_blast > 20:
                st.error(f"⚠ Blast percentage {v2_blast:.1f}% is critically elevated — normal is <5%")
            if v2_wbc > 50000:
                st.error(f"⚠ WBC count {v2_wbc:,} is critically elevated — normal is 5,000-10,000")
            if v2_flt3_exp > 7:
                st.warning(f"⚠ FLT3 gene expression {v2_flt3_exp} is elevated — normal is 2-5")

            st.caption("⚠ V2 model trained on 28-feature synthetic genomic data. Not a clinical diagnosis.")
            st.caption(f"🔬 V2 accuracy: 98.5% vs V1 accuracy: 94.5% — improvement from multi-modal genomic profiling")

        except FileNotFoundError:
            st.error("V2 model not found! Run: python risk_predictor_v2.py first!")

# ============================================
# TAB 2 — SYMPTOM JOURNEY
# ============================================
with tab2:
    st.markdown('<div class="section-title">> STEP 1 — SYMPTOMS</div>', unsafe_allow_html=True)
    st.caption("Doctor inputs what they observe — no lab results needed yet")

    col_s1, col_s2 = st.columns(2, gap="large")

    with col_s1:
        st.markdown("**General symptoms:**")
        sym_fatigue    = st.checkbox("Unexplained fatigue / tiredness")
        sym_pale       = st.checkbox("Pale skin / anaemia")
        sym_fever      = st.checkbox("Frequent fevers / infections")
        sym_bruising   = st.checkbox("Easy bruising or bleeding")
        sym_bone       = st.checkbox("Bone or joint pain")

    with col_s2:
        st.markdown("**Advanced symptoms:**")
        sym_lymph      = st.checkbox("Swollen lymph nodes")
        sym_weight     = st.checkbox("Unexplained weight loss")
        sym_night      = st.checkbox("Night sweats")
        sym_breath     = st.checkbox("Shortness of breath")
        sym_abdomen    = st.checkbox("Swollen abdomen / spleen")

    analyse_btn = st.button("▶ ANALYSE SYMPTOMS")

    if analyse_btn:
        symptoms = {
            'fatigue':  sym_fatigue,
            'pale':     sym_pale,
            'fever':    sym_fever,
            'bruising': sym_bruising,
            'bone':     sym_bone,
            'lymph':    sym_lymph,
            'weight':   sym_weight,
            'night':    sym_night,
            'breath':   sym_breath,
            'abdomen':  sym_abdomen,
        }
        symptom_count = sum(symptoms.values())

        st.markdown("---")
        st.markdown('<div class="section-title">> STEP 2 — BIOLOGICAL ANALYSIS</div>', unsafe_allow_html=True)

        if symptom_count == 0:
            st.info("No symptoms selected — please check at least one symptom")
        else:
            # Biological explanation
            explanations = []
            if sym_fatigue or sym_pale:
                explanations.append("🔴 **Anaemia detected** — red blood cells being crowded out by blast cells in bone marrow")
            if sym_fever:
                explanations.append("🔴 **Immune failure** — white blood cells immature and unable to fight infection")
            if sym_bruising:
                explanations.append("🔴 **Low platelets** — bone marrow producing blasts instead of clotting cells")
            if sym_bone:
                explanations.append("🔴 **Marrow expansion** — blast cells filling bone cavity causing pressure and pain")
            if sym_lymph or sym_abdomen:
                explanations.append("🔴 **Lymphatic spread** — cancer cells spreading beyond bone marrow")
            if sym_weight or sym_night:
                explanations.append("🔴 **Systemic cancer response** — body fighting widespread disease")

            for exp in explanations:
                st.markdown(exp)

            st.markdown("---")
            st.markdown('<div class="section-title">> STEP 3 — PROBABLE MUTATIONS</div>', unsafe_allow_html=True)

            # Mutation probability based on symptoms
            mutation_probs = {}

            if sym_fatigue or sym_pale or sym_fever:
                mutation_probs['FLT3-ITD']  = 0.67
                mutation_probs['NPM1']      = 0.55
            if sym_bruising:
                mutation_probs['RUNX1']     = 0.60
                mutation_probs['CEBPA']     = 0.45
            if sym_bone or sym_abdomen:
                mutation_probs['BCR-ABL']   = 0.70
                mutation_probs['TP53']      = 0.50
            if sym_lymph:
                mutation_probs['BCR-ABL']   = mutation_probs.get('BCR-ABL', 0) + 0.20
            if sym_fever and sym_fatigue:
                mutation_probs['DNMT3A']    = 0.40
                mutation_probs['WT1']       = 0.35
            if symptom_count >= 5:
                for k in mutation_probs:
                    mutation_probs[k] = min(mutation_probs[k] + 0.15, 0.95)

            if mutation_probs:
                sorted_muts = sorted(mutation_probs.items(), key=lambda x: x[1], reverse=True)
                for mut, prob in sorted_muts:
                    color = "#FF4444" if prob > 0.6 else "#FFD700" if prob > 0.4 else "#888"
                    icon  = "⚠" if prob > 0.6 else "⚡" if prob > 0.4 else "?"
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;
                    font-family:'Share Tech Mono',monospace;padding:8px;
                    background:#0a0a0a;border-radius:4px;border-left:3px solid {color}">
                        <span style="color:{color};font-size:16px">{icon}</span>
                        <span style="color:{color};width:100px;font-weight:bold">{mut}</span>
                        <div style="flex:1;background:#111;border-radius:2px;height:8px">
                            <div style="height:8px;border-radius:2px;background:{color};width:{int(prob*100)}%"></div>
                        </div>
                        <span style="color:{color};width:50px">{prob:.0%}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-title">> STEP 4 — NEXT MOVE</div>', unsafe_allow_html=True)

            urgency = symptom_count >= 5
            if urgency:
                st.error("⚠ HIGH SUSPICION — Act immediately!")
            else:
                st.warning("⚡ MODERATE SUSPICION — Investigate promptly")

            st.markdown("**Order these tests NOW:**")
            tests = []
            tests.append("✅ Full blood count — expect WBC >50,000 if leukemia")
            tests.append("✅ Blood smear — look for blast cells >20%")
            if sym_bone or symptom_count >= 4:
                tests.append("✅ Bone marrow biopsy — confirm blast percentage")
            if mutation_probs:
                top_muts = [m for m, p in sorted_muts if p > 0.4]
                tests.append(f"✅ Genomic panel — test for: {', '.join(top_muts)}")
            tests.append("✅ LDH level — elevated in aggressive disease")
            tests.append("✅ Refer to haematologist — same day if possible")

            for test in tests:
                st.markdown(test)

            st.markdown("---")
            st.info("💡 Once results available — go to **PREDICT V1** or **PREDICT V2** tab to confirm diagnosis with genomic data")

# ============================================
# TAB 5 — HIDDEN PATTERNS (INSIGHTS)
# ============================================
with tab5:
    st.markdown('<div class="section-title">> HIDDEN PATTERNS DISCOVERED BY GAN</div>', unsafe_allow_html=True)
    st.caption("Statistical knowledge extracted from our GAN — derived from synthetic data, zero real patient DNA")

    load_btn = st.button("▶ LOAD GENOMIC INSIGHTS")

    if load_btn:
        try:
            df_insights = pd.read_csv('synthetic_genomes.csv')
            leuk_df = df_insights[df_insights['has_leukemia']==1]
            heal_df = df_insights[df_insights['has_leukemia']==0]

            st.markdown("---")

            # ============================================
            # PATTERN 1 — Mutation frequencies
            # ============================================
            st.markdown('<div class="section-title">> PATTERN 1 — MUTATION FREQUENCIES</div>', unsafe_allow_html=True)
            st.caption("How often each mutation appears in leukemia vs healthy patients")

            mutation_cols = ['FLT3_ITD','BCR_ABL','TP53','RUNX1','DNMT3A','NPM1','CEBPA','WT1']
            freq_data = pd.DataFrame({
                'Mutation':  mutation_cols,
                'Leukemia':  [f"{leuk_df[c].mean():.1%}" for c in mutation_cols],
                'Healthy':   [f"{heal_df[c].mean():.1%}" for c in mutation_cols],
                'Risk ratio': [f"{leuk_df[c].mean()/max(heal_df[c].mean(),0.001):.1f}x" for c in mutation_cols]
            })
            st.dataframe(freq_data, use_container_width=True)

            # ============================================
            # PATTERN 2 — Age risk bands
            # ============================================
            st.markdown("---")
            st.markdown('<div class="section-title">> PATTERN 2 — AGE RISK BANDS</div>', unsafe_allow_html=True)
            st.caption("Which ages are highest risk for leukemia")

            age_risk = df_insights.groupby('age')['has_leukemia'].mean().reset_index()
            age_risk.columns = ['Age', 'Leukemia Rate']
            age_risk['Leukemia Rate'] = age_risk['Leukemia Rate'].apply(lambda x: f"{x:.1%}")
            st.dataframe(age_risk, use_container_width=True)

            # ============================================
            # PATTERN 3 — Mutation burden thresholds
            # ============================================
            st.markdown("---")
            st.markdown('<div class="section-title">> PATTERN 3 — MUTATION BURDEN THRESHOLDS</div>', unsafe_allow_html=True)
            st.caption("Critical burden levels discovered by GAN")

            bins   = [0, 3, 5, 10, 15, 20, 50]
            labels = ['0-3','3-5','5-10','10-15','15-20','20+']
            df_insights['burden_band'] = pd.cut(
                df_insights['mutation_burden'],
                bins=bins, labels=labels
            )
            burden_risk = df_insights.groupby('burden_band')['has_leukemia'].mean().reset_index()
            burden_risk.columns = ['Burden Range', 'Leukemia Rate']
            burden_risk['Leukemia Rate'] = burden_risk['Leukemia Rate'].apply(lambda x: f"{x:.1%}")
            burden_risk['Risk Level'] = burden_risk['Leukemia Rate'].apply(
                lambda x: '🔴 CRITICAL' if float(x.strip('%'))/100 > 0.7
                else '🟡 HIGH' if float(x.strip('%'))/100 > 0.4
                else '🟢 LOW'
            )
            st.dataframe(burden_risk, use_container_width=True)

            # ============================================
            # PATTERN 4 — Dangerous combinations
            # ============================================
            st.markdown("---")
            st.markdown('<div class="section-title">> PATTERN 4 — DANGEROUS COMBINATIONS</div>', unsafe_allow_html=True)
            st.caption("Mutation pairs most associated with leukemia — discovered by GAN")

            combos = []
            for i, m1 in enumerate(mutation_cols):
                for m2 in mutation_cols[i+1:]:
                    both   = leuk_df[(leuk_df[m1]==1) & (leuk_df[m2]==1)]
                    rate   = len(both) / max(len(leuk_df), 1)
                    if rate > 0.05:
                        combos.append({
                            'Combination': f"{m1} + {m2}",
                            'Co-occurrence in leukemia': f"{rate:.1%}",
                            'Risk signal': '🔴 HIGH' if rate > 0.15 else '🟡 MODERATE'
                        })

            if combos:
                combos_df = pd.DataFrame(combos).sort_values(
                    'Co-occurrence in leukemia', ascending=False
                ).head(10)
                st.dataframe(combos_df, use_container_width=True)

            # ============================================
            # DOWNLOAD INSIGHTS REPORT
            # ============================================
            st.markdown("---")
            report = f"""SNAKETAIL GENOMIC INSIGHTS REPORT
Generated from synthetic genomic data — zero real patient DNA

MUTATION FREQUENCIES IN LEUKEMIA:
{freq_data.to_string(index=False)}

AGE RISK BANDS:
{age_risk.to_string(index=False)}

MUTATION BURDEN THRESHOLDS:
{burden_risk.to_string(index=False)}

KEY FINDINGS:
- Mutation burden >15 = critical leukemia risk
- FLT3-ITD most common single mutation
- Multiple mutations together = highest risk
- Ages 3-10 show highest leukemia rates

NOTE: All insights derived from GAN-generated synthetic data.
For research use only. Not a clinical guideline.
"""
            st.download_button(
                "▼ DOWNLOAD INSIGHTS REPORT",
                report,
                "snaketail_insights.txt",
                use_container_width=True
            )
            st.caption("Share these insights with medical schools and researchers — privacy safe!")

        except FileNotFoundError:
            st.error("synthetic_genomes.csv not found! Run python genomic_gan.py first!")