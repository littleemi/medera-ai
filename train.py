import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.models as models

# ==========================================
# 0. AKADEMİK TEKRARLANABİLİRLİK (SEED SABİTLEME)
# Makalelerde modelin sonuçlarının şansa bağlı olmadığını 
# kanıtlamak için bu adım zorunludur.
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================
# 1. HİBRİT MİMARİ: NEURONEXT (DIAGNEXT UYARLAMASI)
# Şafak Hoca'nın DiagNeXt felsefesi: Güçlü bir omurga 
# ve spesifik detaylara odaklanan dikkat kapıları.
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
        # PyTorch'un en modern konvolüsyonel omurgası
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = self.backbone.features
        
        # DiagNeXt felsefesine uygun Spatial Attention modülü
        self.attention = SpatialAttention()
        
        self.classifier = nn.Sequential(
            self.backbone.avgpool,
            nn.Flatten(1),
            nn.Dropout(p=0.4), # Aşırı ezberlemeyi (overfitting) önlemek için Dropout
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x


# Windows'ta multiprocessing hatalarını önlemek için ana kod bloğu:
if __name__ == '__main__':
    # ==========================================
    # 2. DONANIM, VERİ SETİ VE VAL SPLIT STRATEJİSİ
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Donanım: {device}")

    train_dir = "dataset/train"
    if not os.path.exists(train_dir):
        print("HATA: dataset/train klasörü bulunamadı!")
        exit()

    # Eğitim verisi için augmentasyon (Rotasyon, yansıma)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Doğrulama (Validation) verisi için SADECE boyutlandırma ve normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Tüm veriyi belleğe çek (Dönüşümleri daha sonra uygulayacağız)
    full_dataset = datasets.DatasetFolder(
        root=train_dir,
        loader=datasets.folder.default_loader,
        extensions=('.jpg', '.jpeg', '.png', '.webp'),
        transform=None # Şimdilik None, split sonrası uygulayacağız
    )

    siniflar = full_dataset.classes
    num_classes = len(siniflar)

    # Makale Standartlarında Doğrulama: %80 Train, %20 Validation Split
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Transformasyonları atama (Train için augmentasyonlu, Val için temiz)
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    # Windows uyumlu num_workers=0 ayarı!
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Toplam Veri: {len(full_dataset)} | Eğitim: {train_size} | Doğrulama: {val_size} | Sınıf Sayısı: {num_classes}")

    # ==========================================
    # 3. AKADEMİK OPTİMİZASYON VE EĞİTİM DÖNGÜSÜ
    # ==========================================
    model = NeuroNeXt(num_classes=num_classes).to(device)

    # Hafif etiket gürültülerine karşı koruma kalkanı (Label Smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    epochs = 15
    best_val_acc = 0.0
    model_save_path = 'neuronext_v1.pth'

    print("\n🚀 Akademik Standartlarda NeuroNeXt Eğitimi Başlıyor...")

    for epoch in range(epochs):
        # -- EĞİTİM AŞAMASI --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total
        
        # -- DOĞRULAMA (VALIDATION) AŞAMASI --
        # Modelin ezberleyip ezberlemediğini (Overfitting) burada anlıyoruz.
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: %{epoch_train_acc:.2f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: %{epoch_val_acc:.2f}")
        
        # Akademik En İyi Model Kaydetme Kriteri (Validation Accuracy bazlı)
        # Sadece doğrulama verisinde en iyi sonucu veren ağırlıklar kaydedilir.
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 En İyi Model Güncellendi! (Val Acc: %{best_val_acc:.2f}) -> {model_save_path}")

    print(f"\n🎉 Eğitim Tamamlandı! En Yüksek Doğrulama Başarısı: %{best_val_acc:.2f}")