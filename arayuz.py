import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import requests
import os

# RAG (LangChain & Chroma) Kütüphaneleri
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. NEURONEXT MİMARİSİ
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv1(x_cat))
        return x * attention_map

class NeuroNeXt(nn.Module):
    def __init__(self, num_classes=4):
        super(NeuroNeXt, self).__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = self.backbone.features
        self.attention = SpatialAttention()
        self.classifier = nn.Sequential(
            self.backbone.avgpool,
            nn.Flatten(1),
            nn.Dropout(p=0.4),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. YAPAY ZEKA MODELLERİNİ YÜKLE
# ==========================================
print("🚀 Görsel Zeka (NeuroNeXt) Yükleniyor...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuroNeXt(num_classes=4).to(device)
model.load_state_dict(torch.load('neuronext_v1.pth', map_location=device))
model.eval()
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

print("📚 RAG Veritabanı (Chroma) Yükleniyor...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vektor_veritabani = Chroma(persist_directory="rag_db", embedding_function=embedding_model)

# ==========================================
# 3. ERA & RAG (KİTAP OKUYAN KÖPRÜ)
# ==========================================
def era_rapor_yaz(hastalik_adi, eminlik_orani):
    url = "http://localhost:8080/completion"
    
    hastalik_sozlugu = {
        'glioma': 'GLİOMA (Kötü Huylu Beyin Tümörü)',
        'meningioma': 'MENİNGİOMA (İyi Huylu Beyin Zarı Tümörü)',
        'pituitary': 'PİTÜİTER (Hipofiz Bezi Tümörü)',
        'notumor': 'SAĞLIKLI BEYİN (Tümör veya lezyon bulgusu yok)'
    }
    gercek_hastalik = hastalik_sozlugu.get(hastalik_adi.lower(), hastalik_adi)
    
    # --- RAG: KİTAPLARDA ARAMA YAP ---
    okunan_kaynak = ""
    if hastalik_adi.lower() != 'notumor':
        arama_sorgusu = f"{gercek_hastalik} hastalığı MR bulguları ve tedavisi"
        bulunan_belgeler = vektor_veritabani.similarity_search(arama_sorgusu, k=2) 
        okunan_kaynak = "\n\n".join([doc.page_content for doc in bulunan_belgeler])
    else:
        okunan_kaynak = "Beyin MR'ı normal. Literatür taramasına gerek duyulmadı."

    # --- KURŞUN GEÇİRMEZ ZEKİ PROMPT ---
    prompt = f"""Sen uzman bir nöroradyoloji asistanısın. 
Görevin, aşağıdaki YAPAY ZEKA TESPİTİ ve TIBBİ LİTERATÜR'ü kullanarak profesyonel bir rapor yazmaktır.

[YAPAY ZEKA TESPİTİ]
Teşhis: {gercek_hastalik}
Güven Skoru: %{eminlik_orani:.2f}

[TIBBİ LİTERATÜR]
{okunan_kaynak}

ÖNEMLİ KURALLAR:
1. Eğer TIBBİ LİTERATÜR'de anlatılan hastalık ile YAPAY ZEKA TESPİTİ uyuşmuyorsa, literatürü görmezden gel ve sadece {gercek_hastalik} hakkında genel tıbbi bilgine dayanarak rapor yaz.
2. Aşağıdaki şablonu BİREBİR kullan. Şablondaki parantez içindeki açıklamaları SİL ve oraya kendi cümlelerini yaz.

📌 Yapay Zeka Bulgusu:
(Yapay zekanın koyduğu teşhisi ve güven skorunu belirt)

📖 Literatür Özeti:
(Hastalığın özelliklerini kısaca anlat)

🩺 Klinik Öneri:
(Bu hastalık için yapılması gerekenleri belirt)

Rapor:"""
    
    payload = {
        "prompt": prompt,
        "n_predict": 300, 
        "temperature": 0.1, 
        "stop": ["Doktor", "Teşekkür", "Merhaba", "[YAPAY"], 
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() 
        rapor = response.json()['content'].strip()
        return rapor, okunan_kaynak
    except Exception as e:
        return f"Era ile bağlantı kurulamadı. Hata: {e}", "Bağlantı Hatası"

# ==========================================
# 4. ARAYÜZ (GUI) İŞLEM FONKSİYONU
# ==========================================
def analiz_et(input_image):
    if input_image is None:
        return None, "Lütfen bir MR görüntüsü yükleyin!", ""

    rgb_img = cv2.resize(input_image, (224, 224))
    rgb_img_float = np.float32(rgb_img) / 255.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = outputs.argmax(1).item()
        predicted_class = classes[predicted_idx]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_idx].item() * 100
        
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    
    # Era raporu ve okuduğu PDF kaynağını al
    rapor_metni, kaynak_metni = era_rapor_yaz(predicted_class, confidence)
    
    return visualization, rapor_metni, kaynak_metni

# ==========================================
# 5. GRADIO ARAYÜZ TASARIMI
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🧠 NeuroNeXt & Era Klinik Asistanı (RAG Destekli)</h1>")
    gr.Markdown("<p style='text-align: center;'>Beyin MR görüntüsünü yükleyin, yapay zeka literatür taraması yaparak akademik rapor sunsun.</p>")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="MR Görüntüsü Yükle (Sürükle - Bırak)")
            analyze_btn = gr.Button("🔍 Taramayı Başlat", variant="primary")
        
        with gr.Column():
            img_output = gr.Image(label="Grad-CAM Isı Haritası (Odak Noktası)")
            text_output = gr.Textbox(label="Era (Llama 3.1) Klinik Değerlendirme Raporu", lines=8)
            kaynak_output = gr.Textbox(label="📖 Literatürden Çekilen Kaynak (RAG Kanıtı)", lines=4, interactive=False)

    analyze_btn.click(fn=analiz_et, inputs=img_input, outputs=[img_output, text_output, kaynak_output])

if __name__ == "__main__":
    demo.launch()