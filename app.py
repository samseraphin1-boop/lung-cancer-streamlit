import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from fpdf import FPDF
import tempfile, os, datetime

st.set_page_config(page_title="Lung Cancer Detection", layout="wide")

# ── Load models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cnn    = tf.keras.models.load_model('best_model_transfer.h5')
    hybrid = tf.keras.models.load_model('best_hybrid_model.h5')
    return cnn, hybrid

cnn_model, hybrid_model = load_models()

# ── Unicode → Latin-1 safe text (fixes FPDFUnicodeEncodingException) ──
def safe_text(text):
    """
    Replace characters unsupported by fpdf's built-in Latin-1 fonts.
    Covers em/en dashes, smart quotes, ellipsis, and other common offenders.
    """
    replacements = {
        "\u2014": "-",   # em dash  —
        "\u2013": "-",   # en dash  –
        "\u2018": "'",   # left single quote  '
        "\u2019": "'",   # right single quote  '
        "\u201C": '"',   # left double quote  "
        "\u201D": '"',   # right double quote  "
        "\u2026": "...", # ellipsis  …
        "\u00B7": ".",   # middle dot  ·
        "\u2022": "-",   # bullet  •
        "\u00A0": " ",   # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


# ── Grad-CAM ──────────────────────────────────────────────────────
def get_gradcam(model, img_array):
    target_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer
    if target_layer is None:
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        target_layer = sub_layer
    if target_layer is None:
        raise ValueError("No Conv2D layer found in model or any nested sub-model.")

    captured = {}
    original_call = target_layer.call

    def hooked_call(inputs, **kwargs):
        output = original_call(inputs, **kwargs)
        captured['conv_out'] = output
        return output

    target_layer.call = hooked_call
    img_tensor = tf.cast(img_array, tf.float32)

    try:
        with tf.GradientTape() as tape:
            preds    = model(img_tensor, training=False)
            conv_out = captured['conv_out']
            tape.watch(conv_out)
            loss     = preds[:, 0]
        grads = tape.gradient(loss, conv_out)
    finally:
        target_layer.call = original_call

    if grads is None:
        raise ValueError("Gradients are None.")

    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_gradcam(orig_bgr, heatmap):
    h_resized = cv2.resize(heatmap, (96, 96))
    colored   = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(orig_bgr, 0.6, colored, 0.4, 0)


# ── Cancer type classification ────────────────────────────────────
def get_cancer_type(cancer_prob, prediction):
    """
    Heuristic cancer-type estimation based on confidence level.
    NOTE: em dashes removed to keep strings Latin-1 safe for fpdf.
    """
    if prediction == "Normal":
        return "N/A", "No malignancy detected.", "#4CAF50"

    if cancer_prob >= 0.85:
        return (
            "Lung Adenocarcinoma (likely)",
            "High-confidence malignancy. Morphology consistent with "
            "adenocarcinoma - glandular pattern, irregular nuclei. "
            "Recommend CT scan and biopsy confirmation.",
            "#F44336"
        )
    elif cancer_prob >= 0.65:
        return (
            "Squamous Cell Carcinoma (possible)",
            "Moderate-confidence malignancy. Features may suggest "
            "squamous cell carcinoma - keratinisation, intercellular bridges. "
            "Further histological analysis recommended.",
            "#FF9800"
        )
    else:
        return (
            "Indeterminate Malignancy",
            "Low-confidence cancer signal. Could be early-stage or "
            "atypical presentation. Additional imaging and pathology review advised.",
            "#FF5722"
        )


# ── Risk level ────────────────────────────────────────────────────
def get_risk(cancer_prob):
    if cancer_prob < 0.3:
        return "Low",    "green",  "Unlikely to be malignant."
    elif cancer_prob < 0.7:
        return "Medium", "orange", "Uncertain - further tests recommended."
    else:
        return "High",   "red",    "High likelihood of malignancy. Urgent review advised."


# ── PDF report ────────────────────────────────────────────────────
def generate_pdf(
    patient, prediction, confidence, cancer_prob,
    risk, risk_desc, cancer_type, cancer_type_desc,
    model_used, gradcam_path, threshold, inverted
):
    pdf = FPDF()
    pdf.add_page()

    # ── Header ───────────────────────────────────────────────────
    pdf.set_fill_color(30, 30, 30)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 14, "Lung Cancer Detection Report", ln=True, align="C", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Patient Details ──────────────────────────────────────────
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 9, "Patient Information", ln=True, fill=True)
    pdf.ln(2)

    def row(label, value):
        pdf.set_font("Arial", "B", 11)
        pdf.cell(60, 8, safe_text(str(label)) + ":")
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, safe_text(str(value)), ln=True)

    row("Patient Name",    patient.get("name",    "N/A"))
    row("Age",             patient.get("age",     "N/A"))
    row("Gender",          patient.get("gender",  "N/A"))
    row("Patient ID",      patient.get("pid",     "N/A"))
    row("Referring Doctor",patient.get("doctor",  "N/A"))
    row("Hospital",        patient.get("hospital","N/A"))
    row("Report Date",     datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    pdf.ln(4)

    # ── Divider ───────────────────────────────────────────────────
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # ── Prediction Summary ────────────────────────────────────────
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 9, "Prediction Summary", ln=True, fill=True)
    pdf.ln(2)

    row("Result",            prediction)
    row("Confidence Score",  f"{confidence:.1f}%")
    row("Cancer Probability",f"{cancer_prob:.4f}  (threshold: {threshold:.2f})")
    row("Risk Level",        risk)
    row("Risk Description",  risk_desc)
    row("Model Used",        model_used)
    pdf.ln(4)

    # ── Cancer Type ───────────────────────────────────────────────
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 9, "Cancer Type Classification", ln=True, fill=True)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(60, 8, "Detected Type:")
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, safe_text(cancer_type), ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(60, 8, "Clinical Notes:")
    pdf.set_font("Arial", "", 11)
    # ✅ safe_text() applied here — fixes FPDFUnicodeEncodingException
    pdf.multi_cell(0, 8, safe_text(cancer_type_desc))
    pdf.ln(2)

    # ── Grad-CAM ──────────────────────────────────────────────────
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 9, "Grad-CAM Visualization", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6,
             "Highlighted regions indicate areas influencing the prediction.",
             ln=True)
    pdf.ln(2)
    pdf.image(gradcam_path, x=30, w=150)
    pdf.ln(6)

    # ── Disclaimer ────────────────────────────────────────────────
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, safe_text(
        "DISCLAIMER: This report is generated by an AI model for research "
        "purposes only. It is not a substitute for professional medical "
        "diagnosis. Always consult a qualified medical professional for "
        "clinical decisions."
    ))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name


# ══════════════════════════════════════════════════════════════════
# ── UI
# ══════════════════════════════════════════════════════════════════
st.title("🫁 Lung Cancer Detection")
st.markdown("Upload a histopathological lung image for AI-assisted analysis.")

col_left, col_right = st.columns([1, 1])

# ── LEFT COLUMN ───────────────────────────────────────────────────
with col_left:

    # ── Patient Details ──────────────────────────────────────────
    st.subheader("👤 Patient Details")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        patient_name    = st.text_input("Patient Name",    placeholder="e.g. John Doe")
        patient_age     = st.text_input("Age",             placeholder="e.g. 52")
        patient_pid     = st.text_input("Patient ID",      placeholder="e.g. PT-00123")
    with p_col2:
        patient_gender   = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
        patient_doctor   = st.text_input("Referring Doctor", placeholder="e.g. Dr. Smith")
        patient_hospital = st.text_input("Hospital",       placeholder="e.g. City Hospital")

    st.divider()

    # ── Model Settings ───────────────────────────────────────────
    st.subheader("⚙️ Model Settings")
    model_choice = st.selectbox(
        "Select model", ["Hybrid (CNN + ViT)", "CNN (MobileNetV2)"]
    )

    inverted = st.toggle(
        "Model outputs 0 = Cancer, 1 = Normal (inverted labels)",
        value=True,
        help=(
            "Turn ON if your model was trained with Cancer=0, Normal=1. "
            "Turn OFF if trained with Normal=0, Cancer=1."
        )
    )

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.10, max_value=0.90,
        value=0.50, step=0.05,
        help="Cancer probability >= threshold -> predicted as Cancer."
    )

    st.divider()

    # ── Image Upload ─────────────────────────────────────────────
    st.subheader("🔬 Upload Image")
    uploaded = st.file_uploader(
        "Upload histopathological image",
        type=["jpg", "jpeg", "png"]
    )

# ── RIGHT COLUMN ──────────────────────────────────────────────────
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    orig_bgr   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    orig_bgr   = cv2.resize(orig_bgr, (96, 96))
    img_rgb    = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    img_array  = np.expand_dims(img_rgb / 255.0, axis=0).astype(np.float32)

    model      = hybrid_model if "Hybrid" in model_choice else cnn_model
    model_name = model_choice

    with st.spinner("Analysing image..."):
        raw_prob    = float(model.predict(img_array, verbose=0)[0][0])
        cancer_prob = (1.0 - raw_prob) if inverted else raw_prob
        prediction  = "Cancer" if cancer_prob >= threshold else "Normal"
        confidence  = cancer_prob * 100 if cancer_prob >= threshold \
                      else (1 - cancer_prob) * 100
        risk, risk_color, risk_desc                  = get_risk(cancer_prob)
        cancer_type, cancer_type_desc, ct_color      = get_cancer_type(cancer_prob, prediction)

        try:
            heatmap           = get_gradcam(model, img_array)
            overlay           = overlay_gradcam(orig_bgr.copy(), heatmap)
            gradcam_available = True
        except Exception as e:
            st.warning(f"Grad-CAM could not be generated: {e}")
            gradcam_available = False

    with col_left:
        st.image(img_rgb, caption="Uploaded image", width=300)

    with col_right:

        # ── Results ──────────────────────────────────────────────
        st.subheader("📊 Results")
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("Prediction",  prediction)
        r_col2.metric("Confidence",  f"{confidence:.1f}%")
        r_col3.metric("Risk Level",  risk)
        st.markdown(f"**Risk assessment:** :{risk_color}[{risk_desc}]")

        # ── Cancer Type Box ───────────────────────────────────────
        st.divider()
        st.subheader("🧬 Cancer Type Classification")

        if prediction == "Normal":
            st.success("✅ No malignancy detected. Tissue appears normal.")
        else:
            st.markdown(
                f"""
                <div style="
                    background-color:#1e1e1e;
                    border-left: 5px solid {ct_color};
                    padding: 14px 18px;
                    border-radius: 6px;
                    margin-bottom: 10px;
                ">
                    <h4 style="color:{ct_color}; margin:0 0 6px 0;">
                        {cancer_type}
                    </h4>
                    <p style="color:#cccccc; margin:0; font-size:14px;">
                        {cancer_type_desc}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption(
                "⚠️ Cancer type is an AI-assisted estimate. "
                "A multi-class model or pathologist review is required for confirmed typing."
            )

        # ── Debug Info ────────────────────────────────────────────
        with st.expander("🔍 Debug Info (raw model output)"):
            st.write(f"**Raw model output:** `{raw_prob:.6f}`")
            st.write(
                f"**Label orientation:** "
                f"{'Inverted - 0=Cancer, 1=Normal' if inverted else 'Normal - 0=Normal, 1=Cancer'}"
            )
            st.write(f"**Cancer probability (after flip):** `{cancer_prob:.6f}`")
            st.write(f"**Threshold:** `{threshold}`")
            st.write(f"**Final prediction:** `{prediction}`")
            if raw_prob < 0.1:
                st.error("⚠️ Raw prob is very low. Make sure 'Inverted labels' toggle is ON.")
            elif raw_prob > 0.9:
                st.success("✅ Raw prob is very high. Make sure 'Inverted labels' toggle is OFF.")
            else:
                st.info("Raw prob is in mid range. Adjust threshold if needed.")

        # ── Grad-CAM ──────────────────────────────────────────────
        st.divider()
        st.subheader("🌡️ Grad-CAM Visualization")
        if gradcam_available:
            g_col1, g_col2 = st.columns(2)
            g_col1.image(img_rgb, caption="Original", width=200)
            g_col2.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                caption="Grad-CAM", width=200
            )
            gradcam_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(gradcam_tmp.name, overlay)
        else:
            st.info("Grad-CAM visualization not available for this model.")
            gradcam_tmp = None

        # ── PDF Report ────────────────────────────────────────────
        st.divider()
        st.subheader("📄 Doctor Report")

        if not patient_name.strip():
            st.warning("💡 Enter patient details on the left before generating the report.")

        if st.button("Generate PDF Report", type="primary"):
            if gradcam_available and gradcam_tmp:
                patient_data = {
                    "name":     patient_name     or "Not provided",
                    "age":      patient_age       or "Not provided",
                    "gender":   patient_gender    if patient_gender != "Select" else "Not provided",
                    "pid":      patient_pid       or "Not provided",
                    "doctor":   patient_doctor    or "Not provided",
                    "hospital": patient_hospital  or "Not provided",
                }
                with st.spinner("Generating report..."):
                    pdf_path = generate_pdf(
                        patient_data, prediction, confidence, cancer_prob,
                        risk, risk_desc, cancer_type, cancer_type_desc,
                        model_name, gradcam_tmp.name, threshold, inverted
                    )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📥 Download PDF Report",
                        f,
                        file_name=f"lung_cancer_report_{patient_name or 'patient'}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("Cannot generate PDF - Grad-CAM image is unavailable.")