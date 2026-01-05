import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

st.set_page_config(page_title="Türkçe Duygu Analizi", layout="centered")

@st.cache_resource
def load_model():
    model_dir = os.path.join(os.path.dirname(__file__), "bert_duygu_modeli")

    id2label = {0: "Olumsuz", 1: "Olumlu", 2: "Tarafsız"}
    label2id = {"Olumsuz": 0, "Olumlu": 1, "Tarafsız": 2}

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    config = AutoConfig.from_pretrained(
        model_dir,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    model.eval()

    return tokenizer, model

st.title("Türkçe Duygu Analizi")

with st.spinner("Model yükleniyor..."):
    tokenizer, model = load_model()

text = st.text_area("Metin gir:", height=120)
analyze = st.button("Analiz Et")

if analyze:
    if not text.strip():
        st.warning("Metin giriniz.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        pred_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(torch.max(probs, dim=1).values.item())

        # 🔥 ŞİMDİLİK bu eşiği KALDIRIYORUZ (nötrü şişiriyordu)
        result = model.config.id2label[pred_id]

        st.success(f"Tahmin edilen duygu: **{result}**")
        st.caption(f"Güven skoru: {confidence:.2f}")

        with st.expander("🔍 Detaylar"):
            st.write({
                "Olumsuz": float(probs[0][0]),
                "Olumlu": float(probs[0][1]),
                "Tarafsız": float(probs[0][2]),
            })
