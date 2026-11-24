import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. NEURONEXT MİMARİSİ (Train ile Birebir Aynı Olmalı)
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
# 2. MODEL VE TEST VERİSİNİ YÜKLEME
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Test Donanımı: {device}")

# Test klasörü yolunu belirt (Eğer farklıysa burayı kendi bilgisayarına göre güncelle)
test_dir = "dataset/test" 
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"❌ Test klasörü bulunamadı: {test_dir}")

# Eğitimdeki ile aynı kesin normalizasyon kuralları
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
# Windows için num_workers=0 yapıldı
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

classes = test_dataset.classes
num_classes = len(classes)

print(f"📂 Test Edilecek Sınıflar: {classes}")
print(f"📊 Toplam Test Görseli: {len(test_dataset)}")

model = NeuroNeXt(num_classes=num_classes).to(device)
# Eğittiğimiz yeni ve güçlü ağırlıkları yüklüyoruz
model.load_state_dict(torch.load('neuronext_v1.pth', map_location=device))
model.eval()

# ==========================================
# 3. TEST (INFERENCE) DÖNGÜSÜ
# ==========================================
all_preds = []
all_labels = []

print("⏳ Model test setini değerlendiriyor, lütfen bekleyin...")

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==========================================
# 4. AKADEMİK MATRİS VE RAPOR ÇİZİMİ
# ==========================================
# Terminale Raporu Bas
print("\n" + "="*50)
print("🏆 NEURONEXT SINIFLANDIRMA RAPORU")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

# Matrisi Çizdir
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 14, "weight": "bold"})

plt.title('NeuroNeXt - Güncel Karmaşıklık Matrisi (Confusion Matrix)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Gerçek Sınıf (Ground Truth)', fontsize=12, fontweight='bold')
plt.xlabel('Modelin Tahmini (Predicted)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('yeni_gercek_matris.png', dpi=300)
print("🔥 Yeni matris 'yeni_gercek_matris.png' olarak kaydedildi!")
plt.show()