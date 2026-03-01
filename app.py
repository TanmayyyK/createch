import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as GO
from sklearn.cluster import KMeans
import time
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="NirmanAI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Design ---
st.markdown("""
<style>
    /* Main Background & Text Color */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #58a6ff;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(88, 166, 255, 0.2);
    }
    div[data-testid="metric-container"] label {
        color: #8b949e;
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #238636;
    }
    
    /* Custom divider */
    hr {
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "data_generated" not in st.session_state:
    st.session_state.data_generated = False
if "building_data" not in st.session_state:
    st.session_state.building_data = None
if "clusters" not in st.session_state:
    st.session_state.clusters = None

# --- Mock Data Generation ---
@st.cache_data
def generate_sample_data(num_elements=500):
    np.random.seed(42)
    floors = np.random.randint(1, 15, num_elements)
    
    # Intentionally create clusters of standard sizes for clustering demonstration
    base_sizes = [
        (40, 40, 300), (45, 45, 300), (50, 50, 320), (60, 60, 350)
    ]
    
    elements = []
    for i in range(num_elements):
        is_outlier = np.random.rand() > 0.85
        if not is_outlier:
            # Pick a standard size with slight variation
            base = base_sizes[np.random.choice(len(base_sizes))]
            w = base[0] + np.random.choice([0, 0, 0, 5, -5])
            l = base[1] + np.random.choice([0, 0, 0, 5, -5])
            h = base[2]
        else:
            # Completely random outlier
            w = np.random.randint(30, 80)
            l = np.random.randint(30, 80)
            h = np.random.choice([280, 290, 300, 310, 320, 350])
            
        elements.append({
            "Element_ID": f"C{i+1:04d}",
            "Type": "Column",
            "Floor": floors[i],
            "Width_cm": w,
            "Length_cm": l,
            "Height_cm": h
        })
    return pd.DataFrame(elements)

# --- Sidebar Navigation ---
st.sidebar.title("🏗️ NirmanAI")
st.sidebar.markdown("*End-to-End Workflow Simulator*")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "1️⃣ Data Ingestion & Geometry",
    "2️⃣ Optimization Engine",
    "3️⃣ Automated Kitting & BoQ",
    "4️⃣ Feedback & Digital Twin"
])

st.sidebar.divider()
st.sidebar.info(
    "**Hackathon Tech Stack:**\n"
    "- Python, Pandas, NumPy\n"
    "- Scikit-learn, PyGAD, OR-Tools\n"
    "- Streamlit, Plotly"
)

# --- Page 1: Data Ingestion & Geometry Analysis ---
if page == "1️⃣ Data Ingestion & Geometry":
    st.title("Phase 1: Data Ingestion & Geometry Analysis")
    st.markdown("Upload building structural data or generate a synthetic dataset to begin. The system identifies unique elements and creates a dimensional heatmap.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Building Data")
        uploaded_file = st.file_uploader("Upload BIM/CAD Data (CSV format)", type=["csv"])
        if st.button("Generate Sample Dataset", use_container_width=True, type="primary"):
            with st.spinner("Generating complex building geometry..."):
                time.sleep(1)
                st.session_state.building_data = generate_sample_data(600)
                st.session_state.data_generated = True
                st.success("Successfully generated 600 structural elements.")
    
    if st.session_state.data_generated:
        df = st.session_state.building_data
        
        with col2:
            st.subheader("Structural Overview")
            # KPIs
            kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
            kpi_c1.metric("Total Elements", len(df))
            kpi_c2.metric("Unique Dimensions", len(df[['Width_cm', 'Length_cm', 'Height_cm']].drop_duplicates()))
            kpi_c3.metric("Max Floor", df['Floor'].max())
            
        st.divider()
        st.subheader("Structural Categorization & Raw Data")
        st.dataframe(df, use_container_width=True, height=250)
        
        st.divider()
        st.subheader("The 'Repetition Matrix'")
        st.markdown("Heatmap identifying standard dimensions vs. unique outliers.")
        
        # Create a frequency matrix
        freq_df = df.groupby(['Width_cm', 'Length_cm']).size().reset_index(name='Frequency')
        
        fig = px.density_heatmap(
            freq_df, x="Width_cm", y="Length_cm", z="Frequency",
            title="Density of Column Dimensions (Heatmap)",
            color_continuous_scale="Viridis",
            labels={'Width_cm': 'Width (cm)', 'Length_cm': 'Length (cm)'}
        )
        fig.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3')
        st.plotly_chart(fig, use_container_width=True)

# --- Page 2: Optimization Engine ---
elif page == "2️⃣ Optimization Engine":
    st.title("Phase 2: The Optimization Engine (The 'Brain')")
    st.markdown("Core Data Science layer utilizing clustering for Master Kit generation and genetic algorithms for scheduling.")
    
    if not st.session_state.data_generated:
        st.warning("Please go to Phase 1 and generate or upload data first.")
    else:
        df = st.session_state.building_data
        
        st.subheader("K-Means++ Clustering: Master Kit Generation")
        
        col_algo, col_viz = st.columns([1, 2])
        
        with col_algo:
            num_kits = st.slider("Select Target Master Kits (K)", min_value=2, max_value=10, value=5)
            if st.button("Run Clustering Engine", use_container_width=True, type="primary"):
                with st.spinner("Running K-Means++ Core..."):
                    time.sleep(1.5)
                    # Prepare data
                    X = df[['Width_cm', 'Length_cm']]
                    # Fit KMeans
                    kmeans = KMeans(n_clusters=num_kits, init='k-means++', random_state=42)
                    df['Master_Kit_ID'] = kmeans.fit_predict(X)
                    
                    centers = kmeans.cluster_centers_
                    master_kits = pd.DataFrame(centers, columns=['Base_Width', 'Base_Length']).round().astype(int)
                    master_kits['Kit_Name'] = [f"Master Kit {i+1}" for i in range(num_kits)]
                    
                    st.session_state.building_data = df  # Update with cluster IDs
                    st.session_state.clusters = master_kits
                    st.success("Clustering complete!")
            
            if st.session_state.clusters is not None:
                st.write("**Identified Master Kits:**")
                st.dataframe(st.session_state.clusters[['Kit_Name', 'Base_Width', 'Base_Length']], hide_index=True, use_container_width=True)
                
        with col_viz:
            if st.session_state.clusters is not None:
                fig2 = px.scatter(
                    df, x="Width_cm", y="Length_cm", color=df["Master_Kit_ID"].astype(str),
                    title="Column Dimension Clusters",
                    hover_data=['Element_ID'],
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                # Add cluster centers
                fig2.add_trace(GO.Scatter(
                    x=st.session_state.clusters['Base_Width'],
                    y=st.session_state.clusters['Base_Length'],
                    mode='markers', marker=dict(size=15, symbol='star', color='gold', line=dict(width=2, color='white')),
                    name='Master Kit Centers'
                ))
                
                fig2.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3', legend_title="Kit ID")
                st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        st.subheader("Genetic Scheduling & Parametric Assignment")
        if st.session_state.clusters is not None:
            col_sch1, col_sch2 = st.columns(2)
            with col_sch1:
                st.markdown("""
                **Parametric Assignment Example:**
                An outlier $42 \\times 42$ cm column is assigned the $40 \\times 40$ cm Master Kit **plus** a 2cm Extension/Filler BoQ.
                """)
                outlier_sample = df[(df['Width_cm'] % 5 != 0) | (df['Length_cm'] % 5 != 0)].head(3)
                if not outlier_sample.empty:
                    st.dataframe(outlier_sample[['Element_ID', 'Width_cm', 'Length_cm', 'Master_Kit_ID']], use_container_width=True)
            with col_sch2:
                st.markdown("""
                **Genetic Scheduling ("Pouring Cycle"):**
                Minimizing total required kits by moving Kit A systematically across floors.
                """)
                # Mock Gantt Chart
                gantt_data = pd.DataFrame([
                    dict(Task="Kit 1 (Floor 1-3)", Start='2026-03-02', Finish='2026-03-08', Resource="Master Kit 1"),
                    dict(Task="Kit 2 (Floor 1-3)", Start='2026-03-04', Finish='2026-03-10', Resource="Master Kit 2"),
                    dict(Task="Kit 1 (Floor 4-6)", Start='2026-03-09', Finish='2026-03-15', Resource="Master Kit 1"),
                ])
                fig_gantt = px.timeline(gantt_data, x_start="Start", x_end="Finish", y="Task", color="Resource")
                fig_gantt.update_yaxes(autorange="reversed")
                fig_gantt.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3', height=200)
                st.plotly_chart(fig_gantt, use_container_width=True)

# --- Page 3: Automated Kitting & BoQ Generation ---
elif page == "3️⃣ Automated Kitting & BoQ":
    st.title("Phase 3: Automated Kitting & BoQ Generation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dynamic BoQ List")
        st.markdown("Shopping list generated for a specific pour.")
        boq_data = pd.DataFrame({
            "Component": ["Standard Panel 40x300", "Corner Profile", "Extension Shim 2cm", "Heavy Duty Props", "Waler Beams", "Connecting Bolts"],
            "Quantity Required": [12, 4, 8, 16, 8, 48],
            "Unit": ["pcs", "pcs", "pcs", "pcs", "pcs", "pcs"],
            "Stock Status": ["In Stock", "In Stock", "Low Stock", "In Stock", "In Stock", "Order Needed"]
        })
        
        def highlight_stock(s):
            if s == 'Order Needed':
                return 'background-color: #7a1f1f; color: white'
            elif s == 'Low Stock':
                return 'background-color: #8c6b12; color: white'
            return 'background-color: #1f612d; color: white'
            
        st.dataframe(boq_data.style.map(highlight_stock, subset=['Stock Status']), use_container_width=True)
        
        if st.button("Submit Order for Shortages"):
            st.success("Purchase order automatically generated via ERP integration.")

    with col2:
        st.subheader("Kitting Instructions & Worker App")
        st.info("Mobile View for Site Worker: **Pour #402 - Column C0015**")
        st.markdown("""
        **Assembly Guide:**
        1. Base: Use Master Kit 1 (40x40 configuration).
        2. Adjustments: Insert 2x 1cm shims on the East face.
        3. Props: Attach 4 props at 45° angle.
        """)
        
        # Fake QR Code or visual
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/QR_code_for_mobile_English_Wikipedia.svg/200px-QR_code_for_mobile_English_Wikipedia.svg.png", width=150, caption="Scan for 3D AR Model View")

# --- Page 4: Feedback Loop & Digital Twin ---
elif page == "4️⃣ Feedback & Digital Twin":
    st.title("Phase 4: Feedback Loop & Digital Twin")
    
    # Dashboard KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Inventory Reduction", "18.4%", "+3.4% vs Target")
    col2.metric("Avg. Repetition Score", "42 uses / kit", "+12 uses")
    col3.metric("Est. Carbon Saved", "4.2 Tons", "vs traditional Method")
    
    st.divider()
    
    col_dash1, col_dash2 = st.columns(2)
    
    with col_dash1:
        st.subheader("Wear-and-Tear Tracking")
        # Generate some mock lifecycle data
        lifecycle_data = pd.DataFrame({
            "Asset_ID": ["PNL-404", "PNL-512", "PRP-801", "CRN-102", "PNL-409"],
            "Uses_Count": [82, 45, 12, 95, 78],
            "Condition_Score": [32, 85, 95, 15, 41]
        })
        
        fig3 = px.bar(lifecycle_data, x="Asset_ID", y="Uses_Count", color="Condition_Score",
                      color_continuous_scale="RdYlGn", title="Panel Lifecycle & Maintenance Triggers")
        fig3.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Maintenance Threshold (80 uses)")
        fig3.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3')
        st.plotly_chart(fig3, use_container_width=True)
        
    with col_dash2:
        st.subheader("Site Feedback Input")
        st.markdown("Allows site workers to provide feedback on assembly difficulty.")
        
        with st.form("feedback_form"):
            st.selectbox("Select Master Kit Configuration", ["Master Kit 1", "Master Kit 2", "Master Kit 3"])
            st.slider("Assembly Difficulty (1 = Easy, 10 = Very Hard)", 1, 10, 5)
            st.text_area("Additional Notes (e.g. 'Shims difficult to align')")
            if st.form_submit_button("Submit to AI Model"):
                st.success("Feedback logged! AI Clustering logic will penalize this configuration heavily in the next training epoch.")
