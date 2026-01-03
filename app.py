import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


st.set_page_config(
    page_title="Türkçe Duygu Analizi",
    layout="centered"
)


@st.cache_resource
def load_model():
    model_path = "bert_duygu_modeli"  # klasör adı
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


st.title(" Türkçe Duygu Analizi")




text = st.text_area("Metin gir:", height=120)
analyze = st.button("Analiz Et")


if analyze:
    if not text.strip():
        st.warning("metin giriniz:")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        labels = ["Negatif ", "Nötr ", "Pozitif "]

        confidence, pred = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred = pred.item()


        if confidence < 0.55:
            result = "Nötr "
        else:
            result = labels[pred]

        st.success(f"Tahmin edilen duygu: **{result}**")
        st.caption(f"Güven skoru: {confidence:.2f}")


        with st.expander("🔍 Detaylar"):
            st.write({
                "Negatif": float(probs[0][0]),
                "Nötr": float(probs[0][1]),
                "Pozitif": float(probs[0][2])
            })
