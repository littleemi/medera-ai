import requests
import json

def era_klinik_rapor_yaz(hastalik_adi, eminlik_orani):
    # llama.cpp Server varsayılan adresi (Port 8080)
    url = "http://localhost:8080/completion"
    
    # Era'ya vereceğimiz klinik direktif (Prompt)
    prompt = f"""Sen 'Era' adında, tamamen çevrimdışı ve gizlilik odaklı çalışan uzman bir nöroradyoloji asistanısın. 
Geliştirdiğimiz NeuroNeXt adlı yapay zeka modeli, hastanın beyin MR görüntüsünü analiz etti.
    
Tıbbi Analiz Sonucu:
- Tespit Edilen Durum: {hastalik_adi.upper()}
- Yapay Zeka Eminlik Oranı: %{eminlik_orani:.2f}
    
Lütfen bu verileri kullanarak doktorun önüne koyulacak çok kısa, profesyonel ve resmi bir 'Ön Değerlendirme Raporu' yaz. İçeriğinde hastalığın ne olduğuna dair 1-2 cümlelik anatomik bilgi ver ve kesin tanı için biyopsi gerektiğini ekle.
    
Rapor:"""
    
    # llama.cpp server'ın anladığı veri formatı
    payload = {
        "prompt": prompt,
        "n_predict": 150,  # Kelime sınırını biraz kıstık ki uzatıp döngüye girmesin
        "temperature": 0.1, # Daha kararlı, yaratıcılıktan uzak ve resmi konuşmasını sağlar
        "stream": False
    }
    
    print(f"📡 NeuroNeXt sonuçları Era'ya (Llama 3.1) iletiliyor... (Durum: {hastalik_adi.upper()}, Güven: %{eminlik_orani:.2f})")
    print("\n🧠 Era Raporu Hazırlıyor...\n" + "="*50)
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() 
        
        sonuc = response.json()
        # llama.cpp metni 'content' anahtarı içinde döndürür
        print(sonuc['content'].strip())
        print("\n" + "=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("❌ HATA: Era (llama-server) ile bağlantı kurulamadı!")
        print("Lütfen arka planda llama-server.exe'nin 8080 portunda çalıştığından emin ol.")
    except Exception as e:
        print(f"❌ Beklenmeyen bir hata oluştu: {e}")

if __name__ == "__main__":
    # Test verisi gönderiyoruz
    era_klinik_rapor_yaz("Meningioma", 94.30)