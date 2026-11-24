import os
from PIL import Image
from pathlib import Path

# Dataset ana klasör yolu (Kendi klasör yoluna göre düzenle)
veri_yolu = Path(r"C:\Users\emir\Desktop\archive\Training")

silinen_sayisi = 0

for dosya in veri_yolu.glob("**/*"):
    if dosya.is_file():
        if dosya.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            try:
                # Resmi açmayı dene, açılmıyorsa veya bozuksa hata verecektir
                with Image.open(dosya) as img:
                    img.verify()
            except Exception as e:
                print(f"Bozuk dosya siliniyor: {dosya.name} | Hata: {e}")
                dosya.unlink() # Bozuk dosyayı bilgisayardan kalıcı olarak siler
                silinen_sayisi += 1

print(f"\nTemizlik bitti! Toplam {silinen_sayisi} adet bozuk/hatalı dosya temizlendi.")