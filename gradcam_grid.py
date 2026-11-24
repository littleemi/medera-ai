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
# 2. MODELİ HAZIRLA
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuroNeXt(num_classes=4)
model.load_state_dict(torch.load('neuronext_v1.pth', map_location=device))
model.to(device)
model.eval()

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
train_dir = "dataset/train"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# ==========================================
# 3. GENİŞLETİLMİŞ ÇİFTLİ IZGARA (4x4) OLUŞTURMA
# ==========================================
# 4 Satır (Sınıflar) ve 4 Sütun (Örnek1 Orijinal, Örnek1 CAM, Örnek2 Orijinal, Örnek2 CAM)
fig, axes = plt.subplots(4, 4, figsize=(18, 16))
fig.suptitle("NeuroNeXt - Çiftli (Double) Grad-CAM Doğrulama Analizi", fontsize=18, fontweight='bold', y=0.98)

for i, sinif in enumerate(classes):
    sinif_yolu = os.path.join(train_dir, sinif)
    resimler = [f for f in os.listdir(sinif_yolu) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    # Her sınıftan rastgele 2 FARKLI resim seç
    secilen_resimler = random.sample(resimler, 2)
    
    for j, secilen_resim in enumerate(secilen_resimler):
        resim_tam_yol = os.path.join(sinif_yolu, secilen_resim)
        
        # OpenCV ile oku
        rgb_img = cv2.imread(resim_tam_yol, 1)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img_float = np.float32(rgb_img) / 255.0
        
        input_tensor = transform(rgb_img).unsqueeze(0).to(device)
        
        # Tahmin Al
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_idx = outputs.argmax(1).item()
            predicted_class = classes[predicted_idx]
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_idx].item() * 100
        
        # Isı Haritası Üret
        targets = [ClassifierOutputTarget(predicted_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        
        # Sütun indeksini hesapla (1. örnek için 0 ve 1, 2. örnek için 2 ve 3)
        col_offset = j * 2
        
        # Orijinal Çizim
        axes[i, col_offset].imshow(rgb_img)
        axes[i, col_offset].set_title(f"Örnek {j+1}: {sinif.upper()}")
        axes[i, col_offset].axis('off')
        
        # Grad-CAM Çizim
        axes[i, col_offset + 1].imshow(visualization)
        color = "green" if sinif == predicted_class else "red"
        axes[i, col_offset + 1].set_title(f"Tahmin: {predicted_class.upper()} (%{confidence:.1f})", color=color)
        axes[i, col_offset + 1].axis('off')

plt.tight_layout()
kayit_yolu = "gradcam_ciftli_test.png"
plt.savefig(kayit_yolu, dpi=300, bbox_inches='tight')
print(f"🔥 Çiftli test tamamlandı! Yeni tablo '{kayit_yolu}' adıyla kaydedildi.")