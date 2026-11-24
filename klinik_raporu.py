import os
import random
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import requests

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
# 2. ERA (LLAMA 3.1) KÖPRÜSÜ
# ==========================================
def era_rapor_yaz(hastalik_adi, eminlik_orani):
    url = "http://localhost:8080/completion"
    prompt = f"""Sen 'Era' adında, tamamen çevrimdışı çalışan uzman bir nöroradyoloji asistanısın. 
Geliştirdiğimiz NeuroNeXt modeli, hastanın beyin MR görüntüsünü analiz etti.
    
- Tespit Edilen Durum: {hastalik_adi.upper()}
- AI Güven Skoru: %{eminlik_orani:.2f}
    
Lütfen bu verileri kullanarak kısa, profesyonel bir 'Ön Değerlendirme Raporu' yaz. Hastalığın ne olduğuna dair 1-2 cümlelik anatomik bilgi ver ve biyopsi/klinik korelasyon gerektiğini ekle.
Rapor:"""
    
    payload = {
        "prompt": prompt,
        "n_predict": 150,
        "temperature": 0.1,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() 
        return response.json()['content'].strip()
    except Exception as e:
        return f"Era ile bağlantı kurulamadı. Hata: {e}"

# ==========================================
# 3. ANA OTOMASYON (PIPELINE)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Sistem Başlatılıyor... Donanım: {device}")
    
    # Modeli Yükle
    model = NeuroNeXt(num_classes=4).to(device)
    model.load_state_dict(torch.load('neuronext_v1.pth', map_location=device))
    model.eval()
    
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # 3.1 Rastgele Görüntü Seçimi
    test_dir = "dataset/test"
    secilen_sinif = random.choice(classes)
    sinif_yolu = os.path.join(test_dir, secilen_sinif)
    resimler = [f for f in os.listdir(sinif_yolu) if f.endswith(('.jpg', '.jpeg', '.png'))]
    secilen_resim = random.choice(resimler)
    resim_tam_yol = os.path.join(sinif_yolu, secilen_resim)
    
    print("\n" + "="*50)
    print(f"📸 ANALİZ EDİLEN DOSYA: {resim_tam_yol}")
    print("="*50)

    # 3.2 Görüntü İşleme
    rgb_img = cv2.imread(resim_tam_yol, 1)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img_float = np.float32(rgb_img) / 255.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)
    
    # 3.3 Tahmin
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = outputs.argmax(1).item()
        predicted_class = classes[predicted_idx]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_idx].item() * 100
        
    print(f"🤖 NeuroNeXt Teşhisi: {predicted_class.upper()} (Güven: %{confidence:.2f})")
    
    # 3.4 Grad-CAM Çizimi
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title(f"Gerçek Sınıf: {secilen_sinif.upper()}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Tahmin: {predicted_class.upper()} (%{confidence:.1f})")
    plt.axis('off')
    
    # Haritayı bilgisayara kaydet
    kayit_yolu = "son_klinik_vaka.png"
    plt.savefig(kayit_yolu, dpi=300, bbox_inches='tight')
    
    # 3.5 Era'ya Rapor Yazdırma
    print("📡 Veriler Era'ya iletiliyor...\n")
    rapor = era_rapor_yaz(predicted_class, confidence)
    
    print("🧠 ERA KLİNİK RAPORU:")
    print("-" * 50)
    print(rapor)
    print("-" * 50)
    
    print(f"✅ İşlem tamam! Görsel analiz '{kayit_yolu}' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    main()