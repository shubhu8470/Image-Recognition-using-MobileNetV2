import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from predict import predict_image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Image Recognition",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main{
    background-color:#f5f7fa;
}
.title{
    text-align:center;
    color:#0066cc;
}
.subtitle{
    text-align:center;
    color:gray;
}
.result{
    padding:15px;
    border-radius:10px;
    background:#ffffff;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 class='title'>🧠 AI Image Recognition</h1>", unsafe_allow_html=True)

st.markdown(
"<p class='subtitle'>Upload an image and MobileNetV2 will identify the object.</p>",
unsafe_allow_html=True
)

st.write("---")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About")

st.sidebar.info(
"""
This project uses

• TensorFlow

• MobileNetV2

• ImageNet Dataset

• Streamlit

Built for GitHub Portfolio.
"""
)

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        with st.spinner("Analyzing Image..."):

            predictions = predict_image(uploaded_file)

        st.success("Prediction Completed Successfully!")

        labels = []
        scores = []

        st.subheader("Top 5 Predictions")

        for _, label, probability in predictions:

            labels.append(label)

            scores.append(probability * 100)

            st.write(
                f"✅ **{label}** : {probability*100:.2f}%"
            )

        st.write("")

        fig, ax = plt.subplots(figsize=(8,4))

        ax.barh(labels, scores)

        ax.set_xlabel("Confidence (%)")

        ax.set_title("Prediction Confidence")

        ax.invert_yaxis()

        st.pyplot(fig)

        best = predictions[0]

        st.success(
            f"🏆 Best Prediction: **{best[1]}** ({best[2]*100:.2f}%)"
        )

else:

    st.info("👆 Upload an image to begin.")

# -----------------------------
# Footer
# -----------------------------
st.write("---")

st.markdown(
"""
Made with ❤️ using **TensorFlow**, **MobileNetV2**, and **Streamlit**.
"""
)