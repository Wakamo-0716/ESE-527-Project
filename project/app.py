import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from PIL import Image

# --- 1. Page Configuration & Academic Styling ---
st.set_page_config(
    page_title="Multimodal Sentiment Analysis | ESE 527",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .tech-card { background-color: #f1f3f5; padding: 25px; border-radius: 10px; border-left: 6px solid #1f77b4; margin-bottom: 20px; }
    .highlight { color: #d62728; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Path Handling & Data Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(rel_path):
    return os.path.join(BASE_DIR, rel_path)


def display_img(rel_path, caption="", width=None):
    path = get_path(rel_path)
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_container_width=True if width is None else False, width=width)
    else:
        st.warning(f"⚠️ Image not found: `{rel_path}`")


def load_csv(rel_path):
    path = get_path(rel_path)
    return pd.read_csv(path) if os.path.exists(path) else None


# --- 3. Sidebar: Academic Navigation ---
with st.sidebar:
    st.title("🎓 ESE-527 Project")
    st.markdown("**Project:** WashU Multimodal Sentiment Analysis")
    st.markdown("**Authors:** Nancy Wang & Ziang Deng")
    st.divider()
    page = st.radio("Navigate to", [
        "📌 Introduction & Background",
        "🌲 Data Preprocessing",
        "⚙️ Technical Methodology",
        "📊 Experimental Results",
        "🔍 Result Inspector"
    ])
    st.divider()
    st.caption("McKelvey School of Engineering")

# --- 4. Main Page Logic ---

# PAGE 1: Introduction
if page == "📌 Introduction & Background":
    st.title("Multimodal Sentiment Analysis on CMU-MOSEI Dataset")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Research Objective")
        st.write("""
        This project aims to predict sentiment intensity by jointly modeling **Text**, **Audio**, and **Visual** cues.
        We formulate the task as a continuous sentiment regression problem and investigate how different deep learning 
        architectures and fusion bottlenecks impact prediction accuracy.
        """)

        st.subheader("From Raw Video to Aligned Tensors")
        st.markdown("""
        1. **Segmentation**: Raw internet videos are segmented into utterance-level clips.
        2. **Feature Extraction**: 
            - **Text**: Word embeddings.
            - **Audio**: Frame-level acoustic features.
            - **Vision**: Frame-level facial action units.
        3. **Word-Level Alignment**: Text serves as the primary time axis. High-frequency audio and visual signals are force-aligned to match the word-level timestamps.
        """)

    with col2:
        st.info(
            "💡 **Dataset**: CMU-MOSEI\n\n[Google Drive Link](https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv)")
        st.success("✅ **Key Finding**: The LSTM-based Cross-modal Attention architecture yields the best performance.")

elif page == "⚙️ Technical Methodology":
    st.title("Technical Methodology")

    tab1, tab2 = st.tabs(["Backbone Architectures", "Fusion Strategies"])

    with tab1:
        st.header("1. Backbone Models")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("LSTM")
            st.markdown("""
            **Long Short-Term Memory** networks are designed to capture temporal dependencies 
            in sequences. Unlike standard RNNs, LSTMs use 'gates' to manage the flow of information, 
            preventing the vanishing gradient problem.

            **Key Logic:**
            - **Forget Gate**: Decides which information to discard.
            - **Input/Output Gates**: Control what goes into and out of the cell state.
            - **Bidirectionality**: Processes data in both forward and backward time steps to capture context from both sides.
            """)
            st.latex(r"h_t = \text{LSTM}(x_t, h_{t-1})")

        with col2:
            st.subheader("Transformer")
            st.markdown("""
            The **Transformer** relies entirely on self-attention mechanisms. It bypasses recursion, allowing 
            for massive parallelization and capturing long-range dependencies more effectively than LSTMs.

            **Key Logic:**
            - **Multi-Head Attention**: Allows the model to jointly attend to information from different representation subspaces.
            - **Positional Encoding**: Since there is no recurrence, we inject info about the position of tokens in the sequence.
            """)
            st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")

    with tab2:
        st.header("2. Fusion Strategies")
        st.write("How we combine Text ($x_t$), Audio ($x_a$), and Vision ($x_v$):")

        f1, f2 = st.columns(2)
        with f1:
            st.info("**Early Fusion**")
            st.write(
                "Features are joined at the input level. Simple, but assumes high correlation between modalities at the raw level.")

            st.info("**Gated Fusion**")
            st.write("Uses a learnable gate ($g$) to dynamically weigh the importance of each modality.")
            st.latex(r"h_{f} = g_t \cdot h_t + g_a \cdot h_a + g_v \cdot h_v")

        with f2:
            st.info("**Cross-Modal Attention**")
            st.write(
                "Captures interactions between modalities by using queries ($Q$) from one and keys/values ($K,V$) from another.")
            st.write("*Implementation: We used Text as the primary query to attend to Audio/Visual signals.*")

            st.info("**Tensor Fusion**")
            st.write("Models unimodal, bimodal, and trimodal interactions simultaneously using an outer product.")
            st.latex(r"Z = [z_t; 1] \otimes [z_a; 1] \otimes [z_v; 1]")

# PAGE 3: Isolation Forest
elif page == "🌲 Data Preprocessing":
    st.title("Data Preprocessing")

    display_img("data/reports/dataset_processing.png")

    st.markdown("""
    ### 🌲 Technical Explanation: Isolation Forest
    In multimodal datasets, outliers frequently originate from **signal loss, feature extraction noise, or modality desynchronization**.

    **Core Logic of iForest:**
    - **Isolation Principle**: Anomalies are typically "few and different". During random partitioning of the feature space, anomalies are easier to isolate, resulting in shorter path lengths in random decision trees.
    - **Multimodal Application**: We applied iForest to the concatenated tri-modal feature space. Since sentiment regression is highly sensitive to extreme outliers, removing these points effectively reduced the MAE (Mean Absolute Error) by preventing the fusion layer from overfitting to asynchronous noise.
    """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Statistical Summary")
        df_if = load_csv("data/reports/iforest_stats_summary.csv")
        # if df_if is not None:
        #     st.dataframe(df_if.style.highlight_max(axis=0, color='#ffcccc'), use_container_width=True)

    with col2:
        display_img("data/reports/label_distribution.png", "Post-Cleaning Sentiment Distribution")

    st.divider()
    st.subheader("🔍 Anomalous Samples Deep-dive")
    df_top = load_csv("data/reports/iforest_stats_top20_most_anomalous.csv")
    if df_top is not None:
        st.write(
            "The table below displays the top 20 samples with the highest anomaly scores, exhibiting significant outlier characteristics in the fused feature space.")
        st.dataframe(df_top, use_container_width=True)

    st.divider()
    st.subheader("Temporal Characteristics")
    mod = st.selectbox("Select Modality", ["Text", "Audio", "Vision"])
    c_a, c_b = st.columns(2)
    with c_a:
        display_img(f"data/reports/{mod.lower()}_temporal_mean.png", f"{mod} Mean")
    with c_b:
        display_img(f"data/reports/{mod.lower()}_temporal_std.png", f"{mod} Std Dev")

# PAGE 4: Results
elif page == "📊 Experimental Results":
    st.title("Comparative Performance Analysis")
    with st.container(border=True):
        st.markdown(
            "**Selection Strategy:** The study first evaluated four fusion strategies (Early, Gated, Cross-modal Attention, and Tensor) to identify the optimal multimodal architecture. The strongest framework was then systematically compared against unimodal baselines.")
        st.markdown(
            "**Evaluation Metrics:** Model performance was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Pearson Correlation to account for the imbalanced continuous label distribution.")
        st.markdown(
            "**Statistical Validation:** Bootstrap significance testing was conducted to quantify the stability of performance gaps and ensure improvements were not due to test set sampling variability.")

    # Strictly matching poster data
    poster_results = {
        "Backbone": ["LSTM", "LSTM", "LSTM", "LSTM", "Transformer", "Transformer", "Transformer", "Transformer"],
        "Setting": ["Text (Baseline)", "Text+Vision Cross", "Tri-modal Cross", "Tri-modal Gated", "Text (Baseline)",
                    "Text+Vision Cross", "Tri-modal Cross", "Tri-modal Early"],
        "MAE (↓)": [0.6236, 0.6069, 0.5975, 0.6161, 0.6436, 0.6134, 0.6056, 0.6215],
        "RMSE (↓)": [0.8256, 0.8069, 0.7912, 0.8165, 0.8536, 0.8187, 0.8132, 0.8308],
        "Corr (↑)": [0.6672, 0.7017, 0.7065, 0.6828, 0.6408, 0.6783, 0.6845, 0.6681]
    }
    df_res = pd.DataFrame(poster_results)

    # Highlight best models using Pandas styling
    st.table(
        df_res.style.highlight_min(subset=['MAE (↓)', 'RMSE (↓)'], color='#d4edda').highlight_max(subset=['Corr (↑)'],
                                                                                                  color='#d4edda'))

    st.success(
        "🎯 **Conclusion**: The LSTM-based Cross-modal Attention framework emerges as the strongest model due to its superior ability to capture fine-grained temporal interactions.")

    st.divider()
    st.subheader("Modality Scalability Analysis")
    arch_vis = st.radio("Select Comparison Chart", ["Transformer Based", "LSTM Based"],
                        horizontal=True)

    if "LSTM" in arch_vis:
        display_img("LSTM Results/cross_single_bi_tri_gated_baseline_mae.png", "Single vs Bi vs Tri (LSTM)")
        display_img("LSTM Results/cross_group_comparison_table.png", "Architecture Head-to-Head")
    else:
        display_img("Transfomer Results/cross_single_bimodal_trimodal_comparison.png",
                    "Single vs Bi vs Tri (Transformer)")
        display_img("Transfomer Results/cross_group_comparison_results.png", "Bootstrap Significance Test")
        display_img("Transfomer Results/gated_single_bimodal_trimodal_comparison.png",
                    "Single vs Bi vs Tri (Transformer)")
        display_img("Transfomer Results/gated_group_comparison_results.png", "Bootstrap Significance Test")


elif page == "🔍 Result Inspector":
    st.title("🧪 Interactive Experiment Browser")
    st.markdown("""
        Explore individual sample predictions and compare how different fusion strategies 
        handle specific multimodal inputs.
        """)

    # --- Step 1: Global Filters ---
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            backbone = st.selectbox("Backbone", ["LSTM", "Transformer"])
        with c2:
            fusion = st.selectbox("Fusion Method", ["Cross-modal", "Gated", "Early", "Tensor"])
        with c3:
            # 模拟样本范围，实际可根据 npz 长度动态获取
            sample_idx = st.number_input("Sample ID (0-500)", min_value=0, max_value=500, value=42)

    # --- Step 2: Data Loading Logic ---
    fusion_to_model = {
        "Cross-modal": "cross_attn",
        "Gated": "gated",
        "Early": "early",
        "Tensor": "tensor",
    }

    backbone_to_prefix = {
        "LSTM": "lstm",
        "Transformer": "transformer",
    }

    model_name = fusion_to_model[fusion]
    backbone_prefix = backbone_to_prefix[backbone]

    npz_path = get_path(f"scripts/demo_outputs/{backbone_prefix}_{model_name}/pred_{model_name}.npz")
    json_path = get_path(f"scripts/demo_outputs/{backbone_prefix}_{model_name}/result_{model_name}.json")

    if os.path.exists(npz_path):
        data = np.load(npz_path)
        preds = data["preds"]
        labels = data["labels"]

        # 获取当前样本数据
        curr_pred = float(preds[sample_idx])
        curr_label = float(labels[sample_idx])
        error = abs(curr_pred - curr_label)

        metrics = None
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        # --- Step 3: Visualization Display ---
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Prediction", f"{curr_pred:.4f}")
        col_m2.metric("Ground Truth", f"{curr_label:.4f}")
        col_m3.metric("Absolute Error", f"{error:.4f}", delta=f"{'-' if error < 0.5 else '+'}{error:.2f}",
                      delta_color="inverse")

        if metrics is not None:
            with st.expander("Overall test-set metrics for the selected model"):
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                m2.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                m3.metric("Corr", f"{metrics.get('corr', 0):.4f}")
                if "best_epoch" in metrics and "best_val_rmse" in metrics:
                    st.caption(
                        f"Best epoch: {metrics['best_epoch']} | Best validation RMSE: {metrics['best_val_rmse']:.4f}"
                    )

        st.divider()

        # --- Step 4: Lightweight Interpretability ---
        st.subheader("📊 Prediction vs. Truth Across Batch")
        # 展示前后10个样本的对比趋势
        start = max(0, sample_idx - 50)
        end = min(len(preds), sample_idx + 50)
        chart_data = pd.DataFrame({
            "Prediction": preds[start:end].flatten(),
            "Truth": labels[start:end].flatten()
        })
        st.line_chart(chart_data)

    else:
        st.error(f"Missing prediction file: `{npz_path}`")
        st.info(
            "Please generate prediction files first. For LSTM models trained with `train_lstm.py`, "
            "use the `--save_predictions` flag and set `--pred_dir demo_outputs`. "
            "Example output path: `demo_outputs/lstm_cross_attn/pred_cross_attn.npz`."
        )
        if backbone == "Transformer":
            st.warning(
                "Transformer prediction files have not been connected yet. "
                "You will need to export them to `demo_outputs/transformer_<model_name>/`."
            )