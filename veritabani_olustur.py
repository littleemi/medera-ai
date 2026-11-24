import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def veritabani_kur():
    pdf_klasoru = "tip_kitaplari"
    veritabani_yolu = "rag_db" # Veritabanının kaydedileceği klasör

    print(f"📂 '{pdf_klasoru}' klasöründeki PDF'ler taranıyor...")
    
    # 1. PDF'leri Yükle
    loader = PyPDFDirectoryLoader(pdf_klasoru)
    belgeler = loader.load()
    
    if not belgeler:
        print("❌ HATA: 'tip_kitaplari' klasöründe PDF bulunamadı!")
        return

    print(f"📄 Toplam {len(belgeler)} sayfa metin bulundu. Parçalanıyor...")

    # 2. Metinleri Parçala (Kitabı küçük paragraflara bölüyoruz)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Her parça yaklaşık 1000 karakter olacak
        chunk_overlap=200 # Anlam bütünlüğü kopmasın diye 200 karakter üst üste binecek
    )
    parcalar = text_splitter.split_documents(belgeler)
    print(f"✂️ Metinler {len(parcalar)} adet küçük anlam bloğuna bölündü.")

    # 3. Gömme (Embedding) Modeli Yükle - Tamamen Yerel ve İnternetsiz
    # Multilingual model kullanıyoruz ki hem Türkçe hem İngilizce kitapları anlasın
    print("🧠 Vektör dönüştürücü (Embedding) yapay zeka yükleniyor...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. Chroma Veritabanını Oluştur ve Kaydet
    print("💾 ChromaDB Vektör Veritabanı oluşturuluyor ve kaydediliyor (Bu işlem bilgisayar hızına göre sürebilir)...")
    Chroma.from_documents(
        documents=parcalar, 
        embedding=embedding_model, 
        persist_directory=veritabani_yolu
    )
    
    print(f"✅ MUAZZAM! Tıp kitapları başarıyla okundu ve '{veritabani_yolu}' klasörüne vektör olarak kaydedildi.")

if __name__ == "__main__":
    veritabani_kur()