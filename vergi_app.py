import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import hashlib
import json
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import io
import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from vergi_core import (tam_hesap, vergi_hesapla, gini_hesapla, ort_vergi_hesapla,
                        MEVCUT_ORANLAR, ESKI_SINIRLAR,
                        gelir_st_haric, gelir_dahil, hh_buyukluk, n_pay, n)
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint, LinearConstraint
import time

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(page_title="Vergi Optimizasyon Aracı", layout="wide")

# ============================================================
# VERİTABANI FONKSİYONLARI
# ============================================================
DB_PATH = "taxarch.db"

def db_baglanti():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def db_olustur():
    conn = db_baglanti()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS kullanicilar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eposta TEXT UNIQUE NOT NULL,
            sifre_hash TEXT NOT NULL,
            ad_soyad TEXT,
            kurum TEXT,
            kayit_tarihi TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS analiz_gecmisi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kullanici_id INTEGER NOT NULL,
            tarih TEXT NOT NULL,
            senaryo TEXT,
            opt_gini REAL,
            baseline_gini REAL,
            iyilesme_pct REAL,
            opt_oranlar TEXT,
            opt_sinirlar TEXT,
            parametreler TEXT,
            FOREIGN KEY (kullanici_id) REFERENCES kullanicilar(id)
        )
    """)
    conn.commit()
    conn.close()

def sifre_hashle(sifre):
    return hashlib.sha256(sifre.encode()).hexdigest()

def kullanici_kaydet(eposta, sifre, ad_soyad="", kurum=""):
    try:
        conn = db_baglanti()
        c = conn.cursor()
        c.execute("""INSERT INTO kullanicilar (eposta, sifre_hash, ad_soyad, kurum, kayit_tarihi)
                     VALUES (?, ?, ?, ?, ?)""",
                  (eposta.lower().strip(), sifre_hashle(sifre),
                   ad_soyad, kurum, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        return True, "Kayıt başarılı."
    except sqlite3.IntegrityError:
        return False, "Bu e-posta zaten kayıtlı."
    except Exception as e:
        return False, str(e)

def kullanici_dogrula(eposta, sifre):
    # Sabit hesaplar (demo)
    sabit = {
        "admin@itu.edu.tr": "admin123",
        "hazine@gov.tr":    "hazine2026",
        "gib@gov.tr":       "gib2026",
        "demo@demo.com":    "demo123",
    }
    if eposta in sabit and sifre == sabit[eposta]:
        return True, -1  # id=-1 sabit hesap
    # DB'den kontrol
    try:
        conn = db_baglanti()
        c = conn.cursor()
        c.execute("SELECT id FROM kullanicilar WHERE eposta=? AND sifre_hash=?",
                  (eposta.lower().strip(), sifre_hashle(sifre)))
        row = c.fetchone()
        conn.close()
        if row:
            return True, row[0]
        return False, None
    except:
        return False, None

def analiz_kaydet(kullanici_id, senaryo, opt_gini, baseline_gini,
                  opt_oranlar, opt_sinirlar, parametreler):
    if kullanici_id == -1:
        return  # demo hesap, kaydetme
    try:
        conn = db_baglanti()
        c = conn.cursor()
        iyilesme = (baseline_gini - opt_gini) / baseline_gini * 100
        c.execute("""INSERT INTO analiz_gecmisi
                     (kullanici_id, tarih, senaryo, opt_gini, baseline_gini,
                      iyilesme_pct, opt_oranlar, opt_sinirlar, parametreler)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (kullanici_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   senaryo, float(opt_gini), float(baseline_gini), float(iyilesme),
                   json.dumps(list(opt_oranlar)), json.dumps(list(opt_sinirlar)),
                   json.dumps(parametreler)))
        conn.commit()
        conn.close()
    except Exception as e:
        pass

def gecmis_getir(kullanici_id):
    if kullanici_id == -1:
        return []
    try:
        conn = db_baglanti()
        c = conn.cursor()
        c.execute("""SELECT id, tarih, senaryo, opt_gini, baseline_gini,
                            iyilesme_pct, opt_oranlar, opt_sinirlar, parametreler
                     FROM analiz_gecmisi WHERE kullanici_id=?
                     ORDER BY tarih DESC LIMIT 20""", (kullanici_id,))
        rows = c.fetchall()
        conn.close()
        return rows
    except:
        return []

# Veritabanını oluştur
db_olustur()

# ============================================================
# GİRİŞ SİSTEMİ
# ============================================================
if "giris_yapildi" not in st.session_state:
    st.session_state.giris_yapildi = False
if "kullanici_id" not in st.session_state:
    st.session_state.kullanici_id = None
if "kayit_modu" not in st.session_state:
    st.session_state.kayit_modu = False

if not st.session_state.giris_yapildi:
    # Tam ekran giris
    st.markdown("""
    <style>
    [data-testid="stHeader"]{display:none!important;}
    [data-testid="stSidebar"]{display:none!important;}
    .block-container{padding:0!important;max-width:100%!important;}
    footer{display:none!important;}
    #MainMenu{display:none!important;}
    iframe{border:none!important;}
    </style>
    """, unsafe_allow_html=True)

    giris_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
*{margin:0;padding:0;box-sizing:border-box;}
html,body{height:100%;font-family:'DM Sans',sans-serif;overflow:hidden;}
body{display:flex;background:#eef1f8;}

.left{
  width:58%;
  background:linear-gradient(150deg,#1e3060 0%,#2a4a8a 50%,#1e3870 100%);
  padding:32px 52px;
  position:relative;overflow:hidden;
  display:flex;flex-direction:column;justify-content:center;
  height:720px;
}
.grid{position:absolute;inset:0;pointer-events:none;
  background-image:linear-gradient(rgba(212,175,55,.06) 1px,transparent 1px),
  linear-gradient(90deg,rgba(212,175,55,.06) 1px,transparent 1px);
  background-size:56px 56px;}
.g1{position:absolute;top:-120px;left:-120px;width:420px;height:420px;border-radius:50%;
  background:radial-gradient(circle,rgba(212,175,55,.15) 0%,transparent 65%);pointer-events:none;}
.g2{position:absolute;bottom:-60px;right:40px;width:320px;height:320px;border-radius:50%;
  background:radial-gradient(circle,rgba(100,160,255,.15) 0%,transparent 65%);pointer-events:none;}
.inner{position:relative;z-index:1;width:100%;}
.logo{font-family:'Playfair Display',serif;font-size:58px;font-weight:900;color:#fff;
  line-height:1;letter-spacing:-2px;margin-bottom:12px;}
.logo span{color:#d4af37;}
.tagline{font-size:15px;color:#b0cce8;font-weight:300;margin-bottom:22px;line-height:1.5;}
.bar{width:52px;height:2px;background:linear-gradient(90deg,#d4af37,transparent);margin-bottom:24px;}
.feats{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:18px;}
.feat{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
  border-radius:10px;padding:14px 16px;}
.feat h4{color:#e8f0fc;font-size:13px;font-weight:500;margin-bottom:4px;}
.feat p{color:#7a9ec0;font-size:11.5px;line-height:1.5;}
.footer{padding-top:16px;border-top:1px solid rgba(255,255,255,.08);
  font-size:11px;color:#4a6a8a;letter-spacing:.5px;}

.right{width:42%;background:linear-gradient(160deg,#f5f7fc,#eaecf5);
  display:flex;align-items:center;justify-content:center;
  padding:24px 40px;height:720px;overflow-y:auto;}
.card{width:100%;max-width:400px;background:#fff;border-radius:20px;padding:32px 38px;
  box-shadow:0 12px 60px rgba(20,40,90,.12),0 2px 8px rgba(20,40,90,.06);position:relative;}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#d4af37,#f5d06a,#d4af37);border-radius:20px 20px 0 0;}

/* Tab switcher */
.tabs{display:flex;background:#f0f3fa;border-radius:10px;padding:4px;margin-bottom:20px;}
.tab-btn{flex:1;padding:9px;border:none;background:transparent;border-radius:7px;
  font-family:'DM Sans',sans-serif;font-size:13px;font-weight:500;
  color:#6080a0;cursor:pointer;transition:all .2s;}
.tab-btn.active{background:#fff;color:#1e3060;box-shadow:0 1px 6px rgba(20,40,90,.12);font-weight:600;}
.panel{display:none;} .panel.active{display:block;}

.card h2{font-family:'Playfair Display',serif;font-size:24px;
  font-weight:700;color:#0e1e40;margin-bottom:4px;}
.card .sub{font-size:12px;color:#90a8c0;margin-bottom:16px;}
.lbl{display:block;font-size:10px;font-weight:600;letter-spacing:2px;
  color:#6080a0;text-transform:uppercase;margin-bottom:5px;margin-top:12px;}
input,select{width:100%;padding:11px 14px;border:1.5px solid #dde4f0;border-radius:10px;
  font-size:13px;color:#0e1e40;font-family:'DM Sans',sans-serif;
  outline:none;background:#f8fafd;transition:border-color .2s,box-shadow .2s;
  -webkit-appearance:none;}
input:focus,select:focus{border-color:#d4af37;box-shadow:0 0 0 3px rgba(212,175,55,.13);}
.btn{margin-top:16px;width:100%;padding:13px;
  background:linear-gradient(135deg,#1e3060,#2a4a8a);
  color:#fff;border:none;border-radius:10px;font-size:14px;font-weight:500;
  font-family:'DM Sans',sans-serif;letter-spacing:.4px;cursor:pointer;
  transition:all .25s;box-shadow:0 4px 15px rgba(20,40,90,.2);}
.btn:hover{background:linear-gradient(135deg,#2a4a8a,#3a6aaa);transform:translateY(-1px);}
.err{color:#c0392b;font-size:12px;margin-top:8px;display:none;padding:8px 12px;
  background:#fff5f5;border-radius:6px;border-left:3px solid #e53e3e;}
.ok{color:#1a7a4a;font-size:12px;margin-top:8px;display:none;padding:8px 12px;
  background:#f0faf5;border-radius:6px;border-left:3px solid #27ae60;}
.hint{text-align:center;font-size:11px;color:#b0c0d4;margin-top:12px;}
select option{color:#0e1e40;}
</style>
</head>
<body>

<div class="left">
  <div class="grid"></div><div class="g1"></div><div class="g2"></div>
  <div style="position:absolute;inset:0;pointer-events:none;z-index:0;overflow:hidden;">
    <svg style="position:absolute;bottom:0;left:0;width:100%;height:75%;opacity:0.08;"
      viewBox="0 0 400 300" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
      <line x1="0" y1="300" x2="400" y2="0" stroke="white" stroke-width="1" stroke-dasharray="8,6"/>
      <path d="M0,300 C60,295 120,280 180,250 C240,220 300,175 360,120 C380,100 395,60 400,0"
        fill="none" stroke="#e74c3c" stroke-width="3"/>
      <path d="M0,300 C55,293 115,275 175,242 C235,209 298,162 358,106 C378,86 395,48 400,0"
        fill="rgba(46,204,113,0.15)" stroke="#2ecc71" stroke-width="3"/>
    </svg>
  </div>
  <div class="inner">
    <div class="logo">Tax<span>Arch</span></div>
    <div class="tagline">Türkiye için Optimal Vergi Sistemi Tasarım ve Analiz Platformu</div>
    <div class="bar"></div>
    <div class="feats">
      <div class="feat">
        <h4>Gini Optimizasyonu</h4>
        <p>Gelir eşitsizliğini minimize eden optimal vergi yapısını tasarlayın.</p>
      </div>
      <div class="feat">
        <h4>7 Senaryo</h4>
        <p>Oran, sınır ve dilim sayısını bağımsız veya birlikte optimize edin.</p>
      </div>
      <div class="feat">
        <h4>Duyarlılık Analizi</h4>
        <p>Parametre değişimlerinin Gini üzerindeki etkisini görselleştirin.</p>
      </div>
      <div class="feat">
        <h4>Hane Analizi</h4>
        <p>Kendi hane verinizle kişiselleştirilmiş analiz yapın.</p>
      </div>
    </div>
    <div class="footer">
      ISL4902E &nbsp;·&nbsp; İTÜ İşletme Mühendisliği &nbsp;·&nbsp; 2026
      &nbsp;|&nbsp; Selim Taşlıtarla &amp; Emir Miraç Yaman
    </div>
  </div>
</div>

<div class="right">
  <div class="card">
    <div class="tabs">
      <button class="tab-btn active" onclick="gosterPanel('giris',this)">Giriş Yap</button>
      <button class="tab-btn" onclick="gosterPanel('kayit',this)">Üye Ol</button>
    </div>

    <!-- GİRİŞ PANELİ -->
    <div class="panel active" id="panel-giris">
      <h2>Hoş Geldiniz</h2>
      <div class="sub">Devam etmek için giriş yapın</div>
      <label class="lbl">E-posta</label>
      <input type="text" id="g-em" placeholder="ornek@itu.edu.tr"
        onkeydown="if(event.key==='Enter')document.getElementById('g-pw').focus()">
      <label class="lbl">Şifre</label>
      <input type="password" id="g-pw" placeholder="••••••••"
        onkeydown="if(event.key==='Enter')giris()">
      <div class="err" id="g-err"></div>
      <button class="btn" onclick="giris()">Sisteme Giriş Yap &rarr;</button>
      <div class="hint">Demo: demo@demo.com &nbsp;/&nbsp; demo123</div>
    </div>

    <!-- KAYIT PANELİ -->
    <div class="panel" id="panel-kayit">
      <h2>Üye Ol</h2>
      <div class="sub">Yeni hesabınızı oluşturun</div>
      <label class="lbl">Ad Soyad</label>
      <input type="text" id="k-ad" placeholder="Ad Soyad">
      <label class="lbl">Kurum</label>
      <input type="text" id="k-kurum" placeholder="İTÜ / Hazine / vb.">
      <label class="lbl">E-posta</label>
      <input type="text" id="k-em" placeholder="ornek@itu.edu.tr">
      <label class="lbl">Şifre</label>
      <input type="password" id="k-pw" placeholder="En az 6 karakter">
      <label class="lbl">Şifre Tekrar</label>
      <input type="password" id="k-pw2" placeholder="••••••••">
      <label class="lbl">Kullanım Amacı</label>
      <input type="text" id="k-amac" placeholder="Akademik araştırma, politika analizi...">
      <div class="err" id="k-err"></div>
      <div class="ok" id="k-ok"></div>
      <button class="btn" onclick="kayitOl()">Hesap Oluştur &rarr;</button>
    </div>
  </div>
</div>

<script>
const sabit={"admin@itu.edu.tr":"admin123","hazine@gov.tr":"hazine2026","gib@gov.tr":"gib2026","demo@demo.com":"demo123"};

function gosterPanel(id,btn){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('panel-'+id).classList.add('active');
  btn.classList.add('active');
}

function giris(){
  const em=document.getElementById("g-em").value.trim().toLowerCase();
  const pw=document.getElementById("g-pw").value;
  const err=document.getElementById("g-err");
  err.style.display="none";
  if(!em||!pw){err.textContent="Lütfen tüm alanları doldurun.";err.style.display="block";return;}
  if(sabit[em]&&sabit[em]===pw){
    window.parent.location.href=window.parent.location.href.split('?')[0]+'?user='+encodeURIComponent(em)+'&ok=1';
  } else {
    window.parent.location.href=window.parent.location.href.split('?')[0]+'?user='+encodeURIComponent(em)+'&pw='+encodeURIComponent(pw)+'&ok=1';
  }
}

function kayitOl(){
  const ad=document.getElementById("k-ad").value.trim();
  const kurum=document.getElementById("k-kurum").value.trim();
  const em=document.getElementById("k-em").value.trim().toLowerCase();
  const pw=document.getElementById("k-pw").value;
  const pw2=document.getElementById("k-pw2").value;
  const amac=document.getElementById("k-amac").value.trim();
  const err=document.getElementById("k-err");
  const ok=document.getElementById("k-ok");
  err.style.display="none"; ok.style.display="none";
  if(!ad||!em||!pw||!pw2||!amac){err.textContent="Lütfen tüm alanları doldurun.";err.style.display="block";return;}
  if(pw.length<6){err.textContent="Şifre en az 6 karakter olmalıdır.";err.style.display="block";return;}
  if(pw!==pw2){err.textContent="Şifreler eşleşmiyor.";err.style.display="block";return;}
  if(!em.includes("@")){err.textContent="Geçerli bir e-posta girin.";err.style.display="block";return;}
  if(sabit[em]){err.textContent="Bu e-posta kullanılamaz.";err.style.display="block";return;}
  window.parent.location.href=window.parent.location.href.split('?')[0]
    +'?register=1&ad='+encodeURIComponent(ad)+'&kurum='+encodeURIComponent(kurum)
    +'&user='+encodeURIComponent(em)+'&pw='+encodeURIComponent(pw)+'&amac='+encodeURIComponent(amac);
}
</script>
</body></html>"""

    components.html(giris_html, height=720, scrolling=False)

    # Gizli ama calisan Streamlit formu
    st.markdown("""
    <style>
    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    div[data-testid="stForm"] > div {
        display: flex;
        gap: 8px;
        align-items: center;
        justify-content: center;
        padding: 8px 0 0 0;
    }
    div[data-testid="stForm"] input {
        padding: 8px 14px !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        max-width: 200px !important;
    }
    div[data-testid="stForm"] button {
        padding: 8px 20px !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        white-space: nowrap !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Giriş / Kayıt sekme seçimi
    st.markdown("<br>", unsafe_allow_html=True)
    col_tab1, col_tab2 = st.columns(2)
    with col_tab1:
        if st.button("🔑 Giriş Yap", use_container_width=True,
                     type="primary" if not st.session_state.kayit_modu else "secondary"):
            st.session_state.kayit_modu = False
            st.rerun()
    with col_tab2:
        if st.button("📝 Kayıt Ol", use_container_width=True,
                     type="primary" if st.session_state.kayit_modu else "secondary"):
            st.session_state.kayit_modu = True
            st.rerun()

    st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

    if not st.session_state.kayit_modu:
        with st.form("gf", clear_on_submit=False):
            em = st.text_input("E-posta", placeholder="ornek@itu.edu.tr")
            pw = st.text_input("Şifre", type="password", placeholder="••••••••")
            ok = st.form_submit_button("Giriş Yap →", use_container_width=True, type="primary")
            st.caption("Demo: demo@demo.com / demo123")
            if ok:
                basari, kid = kullanici_dogrula(em, pw)
                if basari:
                    st.session_state.giris_yapildi = True
                    st.session_state.kullanici = em
                    st.session_state.kullanici_id = kid
                    st.rerun()
                elif not em or not pw:
                    st.warning("Lütfen tüm alanları doldurun.")
                else:
                    st.error("Hatalı e-posta veya şifre.")
    else:
        with st.form("kayit_f", clear_on_submit=True):
            st.markdown("**Yeni Hesap Oluştur**")
            ad_soyad  = st.text_input("Ad Soyad", placeholder="Selim Taşlıtarla")
            kurum     = st.text_input("Kurum", placeholder="İTÜ / Hazine vb.")
            em_k      = st.text_input("E-posta", placeholder="ornek@itu.edu.tr")
            pw_k      = st.text_input("Şifre", type="password", placeholder="En az 6 karakter")
            pw_k2     = st.text_input("Şifre Tekrar", type="password", placeholder="••••••••")
            kayit_btn = st.form_submit_button("Kayıt Ol →", use_container_width=True, type="primary")
            if kayit_btn:
                if not em_k or not pw_k or not pw_k2:
                    st.warning("Lütfen tüm alanları doldurun.")
                elif len(pw_k) < 6:
                    st.error("Şifre en az 6 karakter olmalıdır.")
                elif pw_k != pw_k2:
                    st.error("Şifreler eşleşmiyor.")
                elif "@" not in em_k:
                    st.error("Geçerli bir e-posta girin.")
                else:
                    basari, mesaj = kullanici_kaydet(em_k, pw_k, ad_soyad, kurum)
                    if basari:
                        st.success("Kayıt başarılı! Giriş yapabilirsiniz.")
                        st.session_state.kayit_modu = False
                        st.rerun()
                    else:
                        st.error(f"{mesaj}")

    st.stop()

# Giriş yapıldıysa devam et

with st.expander("Bu araç hakkında — Amaç, Kapsam ve Kullanım Kılavuzu"):
    st.markdown("""
    ### Neden Bu Araç?

    Türkiye'de gelir eşitsizliği önemli bir sorun olmaya devam etmektedir.
    Mevcut gelir vergisi sistemi, artan oranlı yapısına rağmen eşitsizliği azaltmada yetersiz kalmaktadır.

    Bu araç, **İstanbul Teknik Üniversitesi Endüstri Mühendisliği** bitirme projesi kapsamında geliştirilmiştir.
    Temel amacı, politika yapıcılara farklı vergi yapılarının gelir dağılımı üzerindeki etkisini
    anlık olarak görebilecekleri bir **karar destek aracı** sunmaktır.

    ---

    ### Nasıl Çalışır?

    Araç, TÜİK'in 2025 Gelir ve Yaşam Koşulları Araştırması verilerini kullanır.
    20 gelir grubuna (%5'lik ventiller) ait hane geliri verileri üzerinde:

    1. **Mevcut vergi sistemi** uygulanarak baseline Gini hesaplanır
    2. Seçilen senaryoya göre **matematiksel optimizasyon** çalıştırılır
    3. Gini katsayısını minimize eden **optimal vergi yapısı** bulunur
    4. Bütçe gelir tarafsızlığı kısıtı ile vergi geliri korunur (±tolerans)

    ---

    ### Kimler Kullanabilir?

    | Kullanıcı | Amaç |
    |-----------|------|
    | Politika yapıcılar | Farklı vergi reformlarının etkisini test etmek |
    | Araştırmacılar | Vergi-eşitsizlik ilişkisini analiz etmek |
    | Akademisyenler | Optimal vergi teorisini uygulamalı görmek |
    """)

baseline_gini, baseline_ort = tam_hesap(MEVCUT_ORANLAR, ESKI_SINIRLAR)


# ============================================================
# SOL PANEL
# ============================================================
# Baslik satirinda cikis
col_baslik, col_cikis = st.columns([10, 1])
with col_baslik:
    st.markdown("### TaxArch — Türkiye Gelir Vergisi Optimizasyon Aracı")
with col_cikis:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Çıkış", help=f"Çıkış Yap ({st.session_state.get('kullanici','')})"):
        st.session_state.giris_yapildi = False
        st.session_state.kullanici = ""
        st.rerun()

with st.sidebar:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&display=swap');

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#1e3060 0%,#162548 100%) !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div { color: #d0dff5 !important; }
    section[data-testid="stSidebar"] .taxarch-logo-gold {
        color: #d4af37 !important;
        -webkit-text-fill-color: #d4af37 !important;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #d4af37 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg,#d4af37,#c49b2a) !important;
        color: #1e3060 !important; font-weight: 600 !important; border: none !important;
    }
    /* Checkbox sarı */
    section[data-testid="stSidebar"] [data-baseweb="checkbox"] [data-checked="true"] {
        background-color: #d4af37 !important; border-color: #d4af37 !important;
    }
    /* Checkbox işaret rengi — accent-color yöntemi */
    input[type="checkbox"] {
        accent-color: #d4af37 !important;
    }
    /* Streamlit checkbox kutucuğu — seçili durum */
    section[data-testid="stSidebar"] [data-baseweb="checkbox"] [data-checked="true"] > div,
    section[data-testid="stSidebar"] [data-baseweb="checkbox"] svg {
        background-color: #d4af37 !important;
        fill: #1e3060 !important;
        color: #1e3060 !important;
        border-color: #d4af37 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="checkbox"] [role="checkbox"][aria-checked="true"] {
        background-color: #d4af37 !important;
        border-color: #d4af37 !important;
    }
    /* Slider track ve thumb sarı */
    section[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {
        background: #d4af37 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: #d4af37 !important; border-color: #d4af37 !important;
    }
    /* Metric değerleri sarı */
    [data-testid="stMetricValue"] {
        color: #d4af37 !important; font-weight: 700 !important;
    }
    /* Seçili sekme sarı */
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #d4af37 !important;
        color: #d4af37 !important; font-weight: 700 !important;
    }
    button[data-baseweb="tab"] { color: #888 !important; font-weight: 500 !important; }
    /* Ana içerik primary buton sarı */
    section.main button[kind="primary"],
    [data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg,#d4af37,#c49b2a) !important;
        color: #1e3060 !important; border: none !important; font-weight: 600 !important;
    }
    section.main button[kind="primary"]:hover {
        background: linear-gradient(135deg,#c49b2a,#b8860b) !important;
    }
    /* Slider (genel) sarı */
    [data-testid="stSlider"] [role="slider"] {
        background: #d4af37 !important; border-color: #d4af37 !important;
    }
    [data-testid="stSlider"] > div > div > div > div {
        background: #d4af37 !important;
    }
    </style>
    <div style="background:linear-gradient(180deg,#1e3060,#162548);
                padding:24px 16px 18px 16px;
                border-bottom:1px solid rgba(212,175,55,0.3);
                margin-bottom:12px; margin-left:-16px; margin-right:-16px; margin-top:-16px;">
      <div style="font-family:'Playfair Display',serif; font-size:42px; font-weight:900;
                  line-height:1; letter-spacing:-2px; color:#ffffff;">
        Tax<span class="taxarch-logo-gold" style="color:#d4af37; -webkit-text-fill-color:#d4af37;">Arch</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<h3 style='color:#d4af37;font-size:16px;margin:0 0 8px 0;'>Parametreler</h3>", unsafe_allow_html=True)

    st.markdown("**Optimize edilecek değişkenler:**")
    opt_oranlar_sec   = st.checkbox("Vergi Oranları", value=True)
    opt_sinirlar_sec  = st.checkbox("Dilim Sınırları", value=False)
    opt_dilim_sec     = st.checkbox("Dilim Sayısı", value=False)

    # Aktif değişkenleri belirle — kullanıcıya senaryo adı gösterilmiyor
    if opt_oranlar_sec and not opt_sinirlar_sec and not opt_dilim_sec:
        senaryo = "Oranlar Optimize Edildi"
    elif opt_sinirlar_sec and not opt_oranlar_sec and not opt_dilim_sec:
        senaryo = "Sınırlar Optimize Edildi"
    elif opt_oranlar_sec and opt_sinirlar_sec and not opt_dilim_sec:
        senaryo = "Oranlar + Sınırlar Optimize Edildi"
    elif opt_dilim_sec and not opt_oranlar_sec and not opt_sinirlar_sec:
        senaryo = "Dilim Sayısı Optimize Edildi"
    elif opt_oranlar_sec and opt_dilim_sec and not opt_sinirlar_sec:
        senaryo = "Oranlar + Dilim Sayısı Optimize Edildi"
    elif opt_sinirlar_sec and opt_dilim_sec and not opt_oranlar_sec:
        senaryo = "Sınırlar + Dilim Sayısı Optimize Edildi"
    elif opt_oranlar_sec and opt_sinirlar_sec and opt_dilim_sec:
        senaryo = "Tam Optimizasyon"
    else:
        senaryo = None

    if not senaryo:
        st.warning("En az bir değişken seçin.")

    # Kısıt aktiflik durumları
    oran_aktif  = opt_oranlar_sec or opt_dilim_sec
    sinir_aktif = opt_sinirlar_sec or opt_dilim_sec
    s1 = senaryo == "Senaryo 1 - Sadece Oranlar"
    s2 = senaryo == "Senaryo 2 - Sadece Sınırlar"

    st.markdown("---")

    if opt_dilim_sec:
        st.subheader("Dilim Sayısı")
        dilim_sayisi = st.slider("Dilim sayısı", min_value=5, max_value=15, value=5)
        st.markdown("---")
    else:
        dilim_sayisi = 5

    if opt_oranlar_sec:
        st.subheader("Oran Kısıtları")
        min_oran      = st.slider("Min oran (%)", 0, 30, 10) / 100
        max_oran      = st.slider("Max oran (%)", 10, 80, 45) / 100
        min_oran_fark = st.slider("Oranlar arası min fark (%)", 0, 20, 3) / 100
        max_oran_fark = st.slider("Oranlar arası max fark (%)", 0, 40, 13) / 100
        st.markdown("---")
    else:
        min_oran = 0.10; max_oran = 0.45; min_oran_fark = 0.03; max_oran_fark = 0.13

    if opt_sinirlar_sec:
        st.subheader("Dilim Sınırı Kısıtları")
        min_sinir      = st.number_input("Min dilim sınırı (TL)", min_value=0,
                                          max_value=500000, value=50000, step=10000)
        son_dilim_min  = st.number_input("Son dilim minimum sınırı (TL)", min_value=100000,
                                          max_value=20000000, value=2000000, step=500000)
        min_sinir_fark_pct = st.slider("Sınırlar arası min fark (önceki dilimin %'si)", 0, 50, 10)
        min_sinir_fark = None  # yüzde tabanlı, optimizasyonda hesaplanır
        st.caption("Her sınır bir öncekinin en az bu kadar üzerinde olacak.")
        st.markdown("---")
    else:
        min_sinir = 50000; son_dilim_min = 2000000; min_sinir_fark = 50000; min_sinir_fark_pct = 10

    # opt_dilim_sec için de sınır kısıt varsayılanları gerekebilir
    if opt_dilim_sec and not opt_sinirlar_sec:
        min_sinir_fark_pct = 10

    st.markdown("---")
    st.subheader("Bütçe Toleransı")
    butce_tolerans = st.slider("Tolerans (±%)", 0.0, 10.0, 2.0, step=0.5) / 100

    st.markdown("---")
    optimize_et = st.button("Optimize Et", type="primary", use_container_width=True)


# ============================================================
# OPTİMİZASYON FONKSİYONU
# ============================================================
def calistir_optimizasyon(senaryo_no, dilim_sayisi, min_oran, max_oran,
                           min_oran_fark, max_oran_fark,
                           min_sinir, son_dilim_min, min_sinir_fark, butce_tolerans):
    tol_alt = 1 - butce_tolerans
    tol_ust = 1 + butce_tolerans

    def ort_oran(x):   return ort_vergi_hesapla(vergi_hesapla(x, ESKI_SINIRLAR))
    def ort_sinir(x):  return ort_vergi_hesapla(vergi_hesapla(MEVCUT_ORANLAR, x))
    def ort_tam(x, k): return ort_vergi_hesapla(vergi_hesapla(x[:k], x[k:]))

    if senaryo_no == 1:
        k = 5
        def hedef(x): return gini_hesapla(vergi_hesapla(x, ESKI_SINIRLAR))
        kisitlar = [
            {'type': 'ineq', 'fun': lambda x: ort_oran(x) - baseline_ort * tol_alt},
            {'type': 'ineq', 'fun': lambda x: baseline_ort * tol_ust - ort_oran(x)},
        ]
        for i in range(k - 1):
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1]-x[i]-min_oran_fark})
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: max_oran_fark-(x[i+1]-x[i])})
        bounds = [(min_oran, max_oran)] * k
        kisit_butce = NonlinearConstraint(ort_oran, baseline_ort*tol_alt, baseline_ort*tol_ust)
        A = np.zeros((k-1, k))
        for i in range(k-1): A[i,i]=-1; A[i,i+1]=1
        kisit_prog = LinearConstraint(A, min_oran_fark, max_oran_fark)
        sonuc = differential_evolution(hedef, bounds,
            constraints=[kisit_butce, kisit_prog],
            seed=42, maxiter=1000, tol=1e-8, popsize=20,
            mutation=(0.5,1.5), recombination=0.9, polish=True)
        sonuc2 = minimize(hedef, sonuc.x, method='SLSQP', bounds=bounds,
                          constraints=kisitlar, options={'maxiter':1000,'ftol':1e-10})
        return sonuc2.x, ESKI_SINIRLAR

    elif senaryo_no == 2:
        k = len(MEVCUT_ORANLAR)
        def hedef(x): return gini_hesapla(vergi_hesapla(MEVCUT_ORANLAR, x))
        kisitlar = [
            {'type': 'ineq', 'fun': lambda x: ort_sinir(x) - baseline_ort * tol_alt},
            {'type': 'ineq', 'fun': lambda x: baseline_ort * tol_ust - ort_sinir(x)},
        ]
        for i in range(k-2):
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1]-x[i]-min_sinir_fark})
        kisit_butce = NonlinearConstraint(ort_sinir, baseline_ort*tol_alt, baseline_ort*tol_ust)
        A = np.zeros((k-2, k-1))
        for i in range(k-2): A[i,i]=-1; A[i,i+1]=1
        kisit_sinir = LinearConstraint(A, min_sinir_fark, np.inf)
        bounds = [(50000,300000),(200000,600000),(600000,1500000),(son_dilim_min, son_dilim_min*4)]
        sonuc = differential_evolution(hedef, bounds,
            constraints=[kisit_butce, kisit_sinir],
            seed=42, maxiter=1000, tol=1e-8, popsize=20,
            mutation=(0.5,1.5), recombination=0.9, polish=True)
        sonuc2 = minimize(hedef, sonuc.x, method='SLSQP', bounds=bounds,
                          constraints=kisitlar, options={'maxiter':1000,'ftol':1e-10})
        return MEVCUT_ORANLAR, sonuc2.x

    elif senaryo_no == 3:
        k = 5
        def hedef(x): return gini_hesapla(vergi_hesapla(x[:k], x[k:]))
        def ort_fn(x): return ort_tam(x, k)
        kisitlar = [
            {'type': 'ineq', 'fun': lambda x: ort_fn(x) - baseline_ort * tol_alt},
            {'type': 'ineq', 'fun': lambda x: baseline_ort * tol_ust - ort_fn(x)},
        ]
        for i in range(k-1):
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1]-x[i]-min_oran_fark})
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: max_oran_fark-(x[i+1]-x[i])})
        for i in range(k-2):
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[k+i+1]-x[k+i]-min_sinir_fark})
        kisit_butce = NonlinearConstraint(ort_fn, baseline_ort*tol_alt, baseline_ort*tol_ust)
        toplam = (k-1)+(k-2)
        A = np.zeros((toplam, 2*k-1))
        for i in range(k-1): A[i,i]=-1; A[i,i+1]=1
        for i in range(k-2): A[k-1+i,k+i]=-1; A[k-1+i,k+i+1]=1
        alt = np.concatenate([np.full(k-1, min_oran_fark), np.full(k-2, min_sinir_fark)])
        ust = np.concatenate([np.full(k-1, max_oran_fark), np.full(k-2, np.inf)])
        kisit_yapi = LinearConstraint(A, alt, ust)
        bounds = [(min_oran,max_oran)]*k + [(50000,300000),(200000,600000),
                                             (600000,2000000),(son_dilim_min, son_dilim_min*4)]
        sonuc = differential_evolution(hedef, bounds,
            constraints=[kisit_butce, kisit_yapi],
            seed=42, maxiter=3000, tol=1e-9, popsize=30,
            mutation=(0.3, 1.8), recombination=0.95, polish=True,
            workers=1, updating='deferred')
        sonuc2 = minimize(hedef, sonuc.x, method='SLSQP', bounds=bounds,
                          constraints=kisitlar, options={'maxiter':1000,'ftol':1e-10})
        x = sonuc2.x
        return x[:k], x[k:]

    else:
        k = dilim_sayisi
        def hedef(x): return gini_hesapla(vergi_hesapla(x[:k], x[k:]))
        def ort_fn(x): return ort_tam(x, k)
        kisitlar = [
            {'type': 'ineq', 'fun': lambda x: ort_fn(x) - baseline_ort * tol_alt},
            {'type': 'ineq', 'fun': lambda x: baseline_ort * tol_ust - ort_fn(x)},
        ]
        for i in range(k-1):
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1]-x[i]-min_oran_fark})
        for i in range(k-2):
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[k+i+1]-x[k+i]-min_sinir_fark})
        bounds = [(min_oran, max_oran)]*k + [(min_sinir, son_dilim_min*4)]*(k-2) + [(son_dilim_min, son_dilim_min*4)]
        x0_listesi = [
            np.concatenate([np.linspace(min_oran, max_oran, k),
                            np.linspace(min_sinir*2, son_dilim_min*2, k-1)]),
            np.concatenate([np.linspace(min_oran, max_oran*0.9, k),
                            np.linspace(min_sinir*3, son_dilim_min*2, k-1)]),
        ]
        en_iyi = None
        en_iyi_gini = np.inf
        for x0 in x0_listesi:
            sonuc = minimize(hedef, x0, method='SLSQP', bounds=bounds,
                             constraints=kisitlar, options={'maxiter':2000,'ftol':1e-10})
            if sonuc.fun < en_iyi_gini:
                en_iyi_gini = sonuc.fun
                en_iyi = sonuc
        x = en_iyi.x
        return x[:k], x[k:]


# ============================================================
# LORENZ EĞRİSİ
# ============================================================
def lorenz_egri(odenen_vergi):
    net = (gelir_dahil - odenen_vergi) / hh_buyukluk
    y   = np.sort(net)
    cg  = np.cumsum(y * n_pay) / np.sum(y * n_pay)
    cp  = np.cumsum(n_pay)
    return np.concatenate([[0], cp]), np.concatenate([[0], cg])




# ============================================================
# YENİ OPTİMİZASYON — 7 KOMBİNASYON
# ============================================================
def calistir_optimizasyon_v2(opt_oran, opt_sinir, opt_dilim,
                              dilim_sayisi, min_oran, max_oran,
                              min_oran_fark, max_oran_fark,
                              min_sinir, son_dilim_min,
                              min_sinir_fark, butce_tolerans):

    tol_alt = 1 - butce_tolerans
    tol_ust = 1 + butce_tolerans

    # Dilim sayısı
    k = dilim_sayisi if opt_dilim else 5

    # Başlangıç oranları ve sınırları
    if k == 5:
        init_oranlar  = MEVCUT_ORANLAR.copy()
        init_sinirlar = ESKI_SINIRLAR.copy()
    else:
        init_oranlar  = np.linspace(min_oran, max_oran, k)
        init_sinirlar = np.linspace(min_sinir*2, son_dilim_min*2, k-1)

    def vergi_fn(oranlar, sinirlar):
        return vergi_hesapla(oranlar, sinirlar)

    def ort_fn(x):
        oranlar, sinirlar = unpack(x)
        return ort_vergi_hesapla(vergi_fn(oranlar, sinirlar))

    def hedef(x):
        oranlar, sinirlar = unpack(x)
        return gini_hesapla(vergi_fn(oranlar, sinirlar))

    def unpack(x):
        """x vektörünü oran ve sınırlara ayır"""
        idx = 0
        if opt_oran:
            oranlar  = x[idx:idx+k]; idx += k
        else:
            oranlar  = init_oranlar[:k] if k <= 5 else np.linspace(min_oran, max_oran, k)
        if opt_sinir or opt_dilim:
            sinirlar = x[idx:idx+k-1]
        else:
            sinirlar = init_sinirlar[:k-1] if k <= 5 else np.linspace(min_sinir*2, son_dilim_min, k-1)
        return oranlar, sinirlar

    # Karar değişkeni vektörü oluştur
    x0_parts = []
    bounds   = []

    if opt_oran:
        x0_parts.append(init_oranlar[:k] if k <= 5 else np.linspace(min_oran, max_oran, k))
        bounds  += [(min_oran, max_oran)] * k

    if opt_sinir or opt_dilim:
        if k <= 5:
            x0_parts.append(init_sinirlar[:k-1])
        else:
            x0_parts.append(np.linspace(min_sinir*2, son_dilim_min*2, k-1))
        bounds += ([(min_sinir, son_dilim_min*4)] * (k-2) +
                   [(son_dilim_min, son_dilim_min*4)])

    x0 = np.concatenate(x0_parts) if x0_parts else np.array([])

    # Hiç değişken yok (olmamalı ama güvenlik için)
    if len(x0) == 0:
        return init_oranlar, init_sinirlar

    # Kısıtlar
    kisitlar = [
        {'type': 'ineq', 'fun': lambda x: ort_fn(x) - baseline_ort * tol_alt},
        {'type': 'ineq', 'fun': lambda x: baseline_ort * tol_ust - ort_fn(x)},
    ]

    if opt_oran:
        for i in range(k-1):
            kisitlar.append({'type': 'ineq',
                'fun': lambda x, i=i: x[i+1] - x[i] - min_oran_fark})
            kisitlar.append({'type': 'ineq',
                'fun': lambda x, i=i: max_oran_fark - (x[i+1] - x[i])})

    if opt_sinir or opt_dilim:
        oran_offset = k if opt_oran else 0
        for i in range(k-2):
            kisitlar.append({'type': 'ineq',
                'fun': lambda x, i=i, o=oran_offset: x[o+i+1] - x[o+i] - min_sinir_fark})

    # NonlinearConstraint ve LinearConstraint (DE için)
    kisit_butce = NonlinearConstraint(ort_fn, baseline_ort*tol_alt, baseline_ort*tol_ust)

    nc = len(x0)
    if opt_oran:
        A = np.zeros((k-1, nc))
        for i in range(k-1): A[i,i]=-1; A[i,i+1]=1
        kisit_prog = LinearConstraint(A, min_oran_fark, max_oran_fark)
        de_constraints = [kisit_butce, kisit_prog]
    else:
        de_constraints = [kisit_butce]

    # Optimizasyon — guclendirilmis parametreler
    sonuc = differential_evolution(
        hedef, bounds,
        constraints=de_constraints,
        seed=42, maxiter=3000, tol=1e-9,
        popsize=30, mutation=(0.3, 1.8),
        recombination=0.95, polish=True,
        updating='deferred')

    # Coklu baslangic noktasiyla lokal iyilestirme
    en_iyi = minimize(hedef, sonuc.x, method='SLSQP',
                      bounds=bounds, constraints=kisitlar,
                      options={'maxiter':2000, 'ftol':1e-12})

    for _p in [0.95, 1.05, 0.90, 1.10]:
        try:
            _x0 = np.clip(sonuc.x * _p,
                          [b[0] for b in bounds],
                          [b[1] for b in bounds])
            _r = minimize(hedef, _x0, method='SLSQP',
                          bounds=bounds, constraints=kisitlar,
                          options={'maxiter':2000, 'ftol':1e-12})
            if _r.fun < en_iyi.fun:
                en_iyi = _r
        except:
            pass

    opt_o, opt_s = unpack(en_iyi.x)
    return opt_o, opt_s

# ============================================================
# OTOMATİK YORUM
# ============================================================

def policy_type_tespiti(senaryo_no, min_oran, max_oran, min_oran_fark,
                         butce_tolerans, iyilesme_pct, butce_fark, dilim_sayisi):
    """Reform tipini tespit et ve etiket döndür."""

    agresif_isaretler = 0
    muhafazakar_isaretler = 0

    if butce_tolerans >= 0.05:       agresif_isaretler += 2
    elif butce_tolerans <= 0.01:     muhafazakar_isaretler += 2

    if max_oran >= 0.50:             agresif_isaretler += 2
    elif max_oran <= 0.40:           muhafazakar_isaretler += 1

    if min_oran <= 0.05:             agresif_isaretler += 1
    if min_oran_fark <= 0.02:        muhafazakar_isaretler += 1

    if senaryo_no == 4 and dilim_sayisi >= 10:
        agresif_isaretler += 1

    yuksek_fiskal_risk = abs(butce_fark) > 3.0

    if yuksek_fiskal_risk and agresif_isaretler >= 3:
        return ("", "Agresif Yeniden Dağılım / Yüksek Fiskal Risk",
                "Yüksek yeniden dağılım kapasitesi ancak ciddi bütçe sapma riski "
                "tespit edildi. Seçilen parametre kombinasyonu bütçe dengesini "
                "önemli ölçüde etkileyebilir.")
    elif agresif_isaretler >= 3:
        return ("", "Agresif Yeniden Dağılım Modeli",
                "Güçlü bir yeniden dağılım yapısı tespit edildi. Yüksek oran "
                "esnekliği ve geniş bütçe toleransı sisteme büyük manevra alanı "
                "tanımaktadır.")
    elif muhafazakar_isaretler >= 3:
        return ("", "Muhafazakâr Reform Modeli",
                "Kısıtlayıcı bir parametre yapısı tespit edildi. Bütçe riski "
                "düşük olmakla birlikte yeniden dağılım kapasitesi sınırlı "
                "kalabilir.")
    else:
        return ("", "Dengeli Yeniden Dağılım Modeli",
                "Orta düzeyde bir reform yapısı tespit edildi. Yeniden dağılım "
                "kapasitesi ve bütçe riski dengeli bir seviyededir.")


def policy_summary_olustur(senaryo_no, iyilesme_pct, butce_fark,
                            opt_gini, baseline_gini, opt_ov, baseline_ov,
                            opt_oranlar, n):
    """2-3 cümlelik executive summary üret."""

    azalan = sum(1 for f in (opt_ov - baseline_ov) if f < -10)
    artan  = sum(1 for f in (opt_ov - baseline_ov) if f > 10)

    senaryo_acik = {
        1: "yalnızca vergi oranlarının optimize edilmesiyle",
        2: "yalnızca dilim sınırlarının yeniden düzenlenmesiyle",
        3: "vergi oranları ve dilim sınırlarının eş zamanlı optimizasyonuyla",
        4: "dilim sayısı artırılarak gerçekleştirilen optimizasyonla",
    }.get(senaryo_no, "optimizasyon sonucunda")

    if iyilesme_pct >= 5:
        etki = "anlamlı ölçüde azaltılmıştır"
    elif iyilesme_pct >= 2:
        etki = "ölçülebilir biçimde azaltılmıştır"
    else:
        etki = "sınırlı düzeyde azaltılmıştır"

    if butce_fark > 0.5:
        butce_c = f"vergi gelirleri %{butce_fark:.1f} oranında artarken"
    elif butce_fark < -0.5:
        butce_c = f"vergi gelirleri %{abs(butce_fark):.1f} oranında azalırken"
    else:
        butce_c = "vergi gelirleri büyük ölçüde sabit tutulurken"

    ozet = (
        f"Bu optimizasyon, {senaryo_acik} Gini katsayısı "
        f"%{iyilesme_pct:.2f} oranında {etki} ({baseline_gini:.4f} → {opt_gini:.4f}). "
        f"{butce_c.capitalize()} yeniden dağılım etkisi ağırlıklı olarak "
        f"üst {artan} gelir grubunun vergi yükünün artırılması ve "
        f"alt {azalan} gelir grubunun yükünün hafifletilmesi yoluyla sağlanmıştır. "
        f"Sonuçlar, mevcut kısıt yapısı çerçevesinde modelin gelir eşitsizliğini "
        f"azaltma kapasitesini ortaya koymaktadır."
    )
    return ozet


def otomatik_yorum(senaryo_no, opt_oranlar, opt_sinirlar, opt_gini, opt_ort,
                   baseline_gini, baseline_ort, opt_ov, baseline_ov,
                   min_oran, max_oran, min_oran_fark, max_oran_fark,
                   min_sinir, son_dilim_min, min_sinir_fark, butce_tolerans,
                   dilim_sayisi, sure):

    yorumlar = []
    iyilesme     = baseline_gini - opt_gini
    iyilesme_pct = (iyilesme / baseline_gini) * 100
    butce_fark   = (opt_ort - baseline_ort) / baseline_ort * 100

    # ── POLICY TYPE ─────────────────────────────────────────
    emoji, etiket, aciklama = policy_type_tespiti(
        senaryo_no, min_oran, max_oran, min_oran_fark,
        butce_tolerans, iyilesme_pct, butce_fark, dilim_sayisi)

    # ── POLICY SUMMARY ──────────────────────────────────────
    ozet = policy_summary_olustur(
        senaryo_no, iyilesme_pct, butce_fark,
        opt_gini, baseline_gini, opt_ov, baseline_ov, opt_oranlar, n)

    # ── PARAMETRE YORUMLARI ──────────────────────────────────
    if butce_tolerans == 0:
        yorumlar.append(("warning", "Bütçe Toleransı",
            "Sıfır tolerans seçildi; tam gelir tarafsızlığı zorunlu kılındı. "
            "Bu kısıt optimizasyonun yeniden dağılım kapasitesini önemli ölçüde "
            "daraltmaktadır."))
    elif butce_tolerans <= 0.02:
        yorumlar.append(("info", "Bütçe Toleransı",
            f"±%{butce_tolerans*100:.1f} tolerans, bütçe gelirini büyük ölçüde "
            "korurken optimizasyona yeterli esneklik tanıyan dengeli bir seçimdir."))
    else:
        yorumlar.append(("info", "Bütçe Toleransı",
            f"±%{butce_tolerans*100:.1f} tolerans, geniş bir manevra alanı "
            "sunmakta ve daha agresif bir yeniden dağılıma olanak tanımaktadır."))

    if senaryo_no != 2:
        if min_oran <= 0.05:
            yorumlar.append(("info", "Alt Oran Sınırı",
                f"Alt oran sınırı %{min_oran*100:.0f} düzeyi, yeniden dağılım "
                "etkisini artıracak ölçüde düşük olup alt gelir gruplarının "
                "vergi yükünü önemli ölçüde hafifletebilir."))
        if max_oran >= 0.50:
            yorumlar.append(("warning", "Üst Oran Sınırı",
                f"Üst oran sınırı %{max_oran*100:.0f}, uluslararası karşılaştırmalı "
                "açıdan yüksek bir düzeydedir. Bu modelde davranışsal tepkiler "
                "(vergi kaçınma, emek arzı azalması) dikkate alınmamaktadır; "
                "gerçek reforma yansıması daha sınırlı olabilir."))
        if min_oran_fark <= 0.02:
            yorumlar.append(("warning", "Oran Fark Kısıtı",
                "Oranlar arası minimum fark yeniden dağıtım etkisini zayıflatacak "
                "ölçüde küçüktür; dilimler birbirine yakın oranlar alarak "
                "progressiviteyi sınırlayabilir."))

    if senaryo_no in [2, 3]:
        if min_sinir_fark < 50000:
            yorumlar.append(("warning", "Dilim Sınır Farkı",
                f"Dilimler arası minimum fark ({min_sinir_fark:,.0f} TL) oldukça "
                "küçüktür. Bu yapı dilim sınırlarının birbirine çok yakınlaşması "
                "riskini beraberinde getirebilir."))

    if senaryo_no == 4:
        if dilim_sayisi >= 12:
            yorumlar.append(("warning", "Dilim Sayısı",
                f"{dilim_sayisi} dilimlik yapı, OECD ortalamasının (5-7 dilim) "
                "belirgin biçimde üzerindedir. Sistem daha hassas olmakla birlikte "
                "uygulamada karmaşıklık artışına yol açabilir."))
        elif dilim_sayisi <= 5:
            yorumlar.append(("info", "Dilim Sayısı",
                f"{dilim_sayisi} dilim, mevcut sistemle aynı düzeyde olup ek "
                "esneklik kazanımı sınırlı kalacaktır."))

    # ── SONUÇ YORUMLARI ──────────────────────────────────────
    if iyilesme_pct >= 5:
        yorumlar.append(("success", "Gini Katsayısı İyileşmesi",
            f"Gini katsayısı %{iyilesme_pct:.2f} oranında geriledi "
            f"({baseline_gini:.4f} → {opt_gini:.4f}). Bu sonuç, modelin "
            "gelir eşitsizliğini anlamlı ölçüde azaltabildiğini göstermektedir."))
    elif iyilesme_pct >= 2:
        yorumlar.append(("success", "Gini Katsayısı İyileşmesi",
            f"Gini katsayısı %{iyilesme_pct:.2f} oranında geriledi "
            f"({baseline_gini:.4f} → {opt_gini:.4f}). Sonuç politika açısından "
            "anlamlı bir iyileşmeye işaret etmektedir."))
    elif iyilesme_pct >= 0:
        yorumlar.append(("warning", "Gini Katsayısı İyileşmesi",
            f"Gini katsayısı %{iyilesme_pct:.2f} oranında geriledi "
            f"({baseline_gini:.4f} → {opt_gini:.4f}). Marjinal düzeyde kalan "
            "bu iyileşme, kısıtların gevşetilmesiyle artırılabilir."))
    else:
        yorumlar.append(("error", "Gini Katsayısı",
            "Optimizasyon sonucunda Gini katsayısı kötüleşti. Parametre "
            "kombinasyonu tutarsız olabilir; kısıtlar gözden geçirilmelidir."))

    azalan = sum(1 for f in (opt_ov - baseline_ov) if f < -10)
    artan  = sum(1 for f in (opt_ov - baseline_ov) if f > 10)
    if azalan > 0 and artan > 0:
        yorumlar.append(("success", "Yeniden Dağılım Etkisi",
            f"Vergi yükü alt {azalan} gelir grubunda azalırken üst {artan} "
            "gelir grubunda artmıştır. Reform, yük transferini başarıyla "
            "gerçekleştirmiştir."))
    elif azalan > 0:
        yorumlar.append(("warning", "Yeniden Dağılım Etkisi",
            f"Alt {azalan} grup lehine vergi hafiflemesi gözlemlenmiş ancak "
            "üst grupların yükü artmamıştır. Bütçe kısıtı yeniden dağılımı "
            "sınırlamış olabilir."))

    if abs(butce_fark) < 0.5:
        yorumlar.append(("success", "Bütçe Etkisi",
            f"Vergi geliri %{butce_fark:+.2f} değişimiyle bütçe tarafsızlığı "
            "büyük ölçüde sağlanmıştır."))
    elif butce_fark > 0:
        yorumlar.append(("info", "Bütçe Etkisi",
            f"Vergi geliri %{butce_fark:.2f} artmıştır. Reform eşitsizliği "
            "azaltırken aynı zamanda gelir artırıcı bir etki doğurmuştur."))
    else:
        yorumlar.append(("warning", "Bütçe Etkisi",
            f"Vergi geliri %{abs(butce_fark):.2f} azalmıştır. Tolerans sınırı "
            "içinde kalmakla birlikte bütçe açığı riski göz önünde "
            "bulundurulmalıdır."))

    if senaryo_no != 2 and opt_oranlar[-1] >= max_oran * 0.99:
        yorumlar.append(("warning", "Üst Dilim Oranı",
            f"En yüksek vergi oranı belirlenen üst sınıra (%{max_oran*100:.0f}) "
            "dayanmıştır. Üst sınır artırılması halinde ek iyileşme "
            "sağlanabilir."))

    return emoji, etiket, aciklama, ozet, yorumlar


# ============================================================
# PDF RAPORU — TaxArch Lacivert/Altın Tema
# ============================================================
def pdf_olustur(senaryo, opt_oranlar, opt_sinirlar, opt_gini, opt_ort,
                baseline_gini, baseline_ort, etki_df, dilim_df,
                cp_base, cg_base, cp_opt, cg_opt):

    def tr(metin):
        s = str(metin)
        for a, b in [("ı","i"),("İ","I"),("ş","s"),("Ş","S"),
                     ("ğ","g"),("Ğ","G"),("ü","u"),("Ü","U"),
                     ("ö","o"),("Ö","O"),("ç","c"),("Ç","C"),
                     ("–","-"),("→","->")]:
            s = s.replace(a, b)
        return s

    # Renk paleti — lacivert/altın
    LACIVERT  = (30,  48,  96)
    LACIVERT2 = (22,  37,  72)
    ALTIN     = (212, 175,  55)
    ALTIN_ACK = (252, 243, 207)
    YESIL     = (39,  174,  96)
    ACIK_YES  = (212, 239, 223)
    KIRMIZI   = (192,  57,  43)
    ACIK_KIR  = (250, 219, 216)
    GRI       = (120, 130, 140)
    ACIK_GRI  = (240, 243, 248)
    BEYAZ     = (255, 255, 255)
    KOYU      = (30,  48,  96)

    def header_bar(pdf, metin, h=10):
        """Lacivert arka plan, altın sol şerit, beyaz yazı başlık barı."""
        pdf.set_fill_color(*LACIVERT)
        pdf.set_x(15)
        pdf.rect(15, pdf.get_y(), 180, h, "F")
        # Altın sol şerit
        pdf.set_fill_color(*ALTIN)
        pdf.rect(15, pdf.get_y(), 3, h, "F")
        pdf.set_text_color(*BEYAZ)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_x(22)
        pdf.cell(173, h, tr(metin), ln=True)
        pdf.set_text_color(*KOYU)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── SAYFA 1: KAPAK + ÖZET ──────────────────────────────────
    pdf.add_page()

    # Üst lacivert banner
    pdf.set_fill_color(*LACIVERT)
    pdf.rect(0, 0, 210, 42, "F")
    # Altın alt çizgi
    pdf.set_fill_color(*ALTIN)
    pdf.rect(0, 42, 210, 2, "F")
    # Logo: Tax + Arch
    pdf.set_text_color(*BEYAZ)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(14, 10)
    pdf.cell(14, 10, "Tax", ln=False)
    pdf.set_text_color(*ALTIN)
    pdf.cell(20, 10, "Arch", ln=False)
    # Sağda slogan
    pdf.set_text_color(176, 204, 232)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(60, 12)
    pdf.cell(0, 6, "Turkiye Gelir Vergisi Optimizasyon Sistemi", ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(60, 20)
    pdf.cell(0, 5, "ISL4902E  |  ITU Isletme Muhendisligi  |  2026", ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(60, 27)
    pdf.set_text_color(176, 204, 232)
    pdf.cell(0, 5, "Selim Taslitarla  &  Emir Miras Yaman", ln=True)
    # Tarih sağ üst
    from datetime import datetime as _dt
    pdf.set_text_color(176, 204, 232)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(140, 26)
    pdf.cell(0, 5, _dt.now().strftime("%d.%m.%Y  %H:%M"), ln=True)

    pdf.set_text_color(*BEYAZ)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_xy(14, 28)
    pdf.cell(0, 8, tr(f"Rapor: {senaryo}"), ln=True)

    pdf.set_y(52)
    pdf.set_text_color(*KOYU)

    # Özet metrik kutuları (2 satır x 3 sütun)
    _iyilesme_pct = (baseline_gini - opt_gini) / baseline_gini * 100
    _butce_pct = (opt_ort - baseline_ort) / baseline_ort * 100
    metrikler = [
        ("Baseline Gini",       f"{baseline_gini:.4f}",                    ACIK_GRI,  KOYU),
        ("Optimal Gini",        f"{opt_gini:.4f}",                         ACIK_YES,  YESIL),
        ("Gini Iyilesme",       f"%{_iyilesme_pct:.2f}",                   ALTIN_ACK, LACIVERT),
        ("Baseline Ort. Vergi", f"{baseline_ort:,.0f} TL",                 ACIK_GRI,  KOYU),
        ("Yeni Ort. Vergi",     f"{opt_ort:,.0f} TL",                      ACIK_YES,  YESIL),
        ("Butce Degisimi",      f"{_butce_pct:+.2f}%",
         ACIK_KIR if _butce_pct > 0.5 else ACIK_YES,
         KIRMIZI  if _butce_pct > 0.5 else YESIL),
    ]
    kw, kh = 58, 22
    x0, y0 = 15, pdf.get_y()
    for idx, (baslik, deger, bg, fg) in enumerate(metrikler):
        col = idx % 3; row = idx // 3
        x = x0 + col * (kw + 3); y = y0 + row * (kh + 4)
        pdf.set_fill_color(*bg)
        pdf.set_draw_color(*LACIVERT)
        pdf.rect(x, y, kw, kh, "FD")
        # Altın üst şerit
        pdf.set_fill_color(*ALTIN)
        pdf.rect(x, y, kw, 2, "F")
        pdf.set_text_color(*GRI)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_xy(x+2, y+4)
        pdf.cell(kw-4, 4, tr(baslik))
        pdf.set_text_color(*fg)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_xy(x+2, y+10)
        pdf.cell(kw-4, 8, tr(deger))
    pdf.set_text_color(*KOYU)
    pdf.set_y(y0 + 2*(kh+4) + 8)

    # Lorenz grafiği
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.set_facecolor("#f8fafd")
    fig.patch.set_facecolor("#f8fafd")
    ax.plot([0,1],[0,1], color='#aaaaaa', linewidth=1, linestyle='--', label='Tam Esitlik')
    ax.plot(cp_base, cg_base, color='#e74c3c', linewidth=2,
            label=f'Mevcut (Gini={baseline_gini:.4f})')
    cp_int = np.linspace(0,1,len(cp_opt))
    cg_bi  = np.interp(cp_int, cp_base, cg_base)
    ax.fill_between(cp_opt, cg_bi, cg_opt, alpha=0.2, color='#27ae60')
    ax.plot(cp_opt, cg_opt, color='#27ae60', linewidth=2,
            label=f'Optimal (Gini={opt_gini:.4f})')
    ax.set_xlabel("Nufus Payi", fontsize=9)
    ax.set_ylabel("Gelir Payi", fontsize=9)
    ax.set_title("Lorenz Egrisi Karsilastirmasi", fontsize=10, fontweight='bold',
                 color='#1e3060')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig); buf.seek(0)

    if pdf.get_y() > 175: pdf.add_page()
    header_bar(pdf, "Lorenz Egrisi", h=9)
    pdf.image(buf, x=15, y=pdf.get_y()+2, w=175)
    pdf.set_y(pdf.get_y() + 88)

    # ── SAYFA 2: VERGİ DİLİM TABLOSU ──────────────────────────
    pdf.add_page()
    header_bar(pdf, "Vergi Dilim Karsilastirmasi", h=10)
    pdf.ln(4)

    _has_fark = "Fark" in dilim_df.columns
    _has_eski_oran = "Eski Oran" in dilim_df.columns
    _has_yeni_oran = "Yeni Oran" in dilim_df.columns
    _has_eski_aralik = "Eski Aralik" in dilim_df.columns
    _has_yeni_aralik = "Yeni Aralik" in dilim_df.columns

    if _has_fark:
        cols = [("Dilim",15),("Aralik",70),("Eski Oran",24),("Yeni Oran",24),("Fark",22)]
    elif _has_eski_aralik and _has_yeni_aralik:
        cols = [("Dilim",15),("Eski Aralik",77),("Yeni Aralik",78),("Oran",20)]
    else:
        cols = [("Dilim",15),("Aralik",120),("Oran",25)]

    pdf.set_fill_color(*LACIVERT)
    pdf.set_text_color(*BEYAZ)
    pdf.set_font("Helvetica","B",9)
    pdf.set_x(15)
    for baslik, w in cols:
        pdf.cell(w, 8, tr(baslik), border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica","",9)
    for i, (_, row) in enumerate(dilim_df.iterrows()):
        fill_color = ACIK_GRI if i%2==0 else BEYAZ
        fark_color = KOYU
        if _has_fark:
            fark_str = tr(row.get("Fark",""))
            try:
                fark_val = float(fark_str.replace("%","").replace("+",""))
                fill_color = ACIK_YES if fark_val < 0 else (ACIK_KIR if fark_val > 0 else BEYAZ)
                fark_color = YESIL if fark_val < 0 else (KIRMIZI if fark_val > 0 else KOYU)
            except: pass
        pdf.set_fill_color(*fill_color); pdf.set_text_color(*KOYU); pdf.set_x(15)
        pdf.cell(15, 7, tr(str(row.get("Dilim",""))), border=1, fill=True, align="C")
        if _has_fark:
            pdf.cell(70, 7, tr(row.get("Yeni Aralik", row.get("Aralik",""))), border=1, fill=True)
            pdf.cell(24, 7, tr(row.get("Eski Oran","")), border=1, fill=True, align="C")
            pdf.cell(24, 7, tr(row.get("Yeni Oran","")), border=1, fill=True, align="C")
            pdf.set_text_color(*fark_color)
            pdf.cell(22, 7, fark_str, border=1, fill=True, align="C")
        elif _has_eski_aralik and _has_yeni_aralik:
            pdf.cell(77, 7, tr(row.get("Eski Aralik","")), border=1, fill=True)
            pdf.cell(78, 7, tr(row.get("Yeni Aralik","")), border=1, fill=True)
            pdf.cell(20, 7, tr(row.get("Oran","")), border=1, fill=True, align="C")
        else:
            pdf.cell(120, 7, tr(row.get("Aralik","")), border=1, fill=True)
            pdf.cell(25, 7, tr(row.get("Oran","")), border=1, fill=True, align="C")
        pdf.set_text_color(*KOYU); pdf.ln()

    # ── SAYFA 3: ETKİ ANALİZİ ─────────────────────────────────
    pdf.add_page()
    header_bar(pdf, "Gelir Grubu Etki Analizi", h=10)
    pdf.ln(2)
    pdf.set_font("Helvetica","I",8); pdf.set_text_color(*GRI); pdf.set_x(15)
    pdf.cell(0, 5, "Yesil: vergi azaldi  |  Kirmizi: vergi artti", ln=True)
    pdf.set_text_color(*KOYU); pdf.ln(2)

    ecols = [("Grup",15),("Hane Geliri",40),("Eski Vergi",38),("Yeni Vergi",38),("Fark",38)]
    pdf.set_fill_color(*LACIVERT); pdf.set_text_color(*BEYAZ)
    pdf.set_font("Helvetica","B",9); pdf.set_x(15)
    for baslik, w in ecols:
        pdf.cell(w, 8, tr(baslik), border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica","",8)
    for i, (_, row) in enumerate(etki_df.iterrows()):
        fark_str = tr(row["Fark"])
        try:
            fark_val = int(fark_str.replace("+","").replace(" TL","").replace(",","").replace(".",""))
            fill_color = ACIK_YES if fark_val <= 0 else ACIK_KIR
            fark_color = YESIL if fark_val <= 0 else KIRMIZI
        except:
            fill_color = ACIK_GRI if i%2==0 else BEYAZ; fark_color = KOYU
        pdf.set_fill_color(*fill_color); pdf.set_text_color(*KOYU); pdf.set_x(15)
        pdf.cell(15, 6, tr(row["Grup"]),        border=1, fill=True, align="C")
        pdf.cell(40, 6, tr(row["Hane Geliri"]), border=1, fill=True, align="R")
        pdf.cell(38, 6, tr(row["Eski Vergi"]),  border=1, fill=True, align="R")
        pdf.cell(38, 6, tr(row["Yeni Vergi"]),  border=1, fill=True, align="R")
        pdf.set_text_color(*fark_color)
        pdf.cell(38, 6, fark_str, border=1, fill=True, align="R")
        pdf.set_text_color(*KOYU); pdf.ln()

    # Alt footer — her sayfaya
    for pg in range(1, pdf.page + 1):
        pdf.page = pg
        pdf.set_y(-14)
        pdf.set_fill_color(*LACIVERT)
        pdf.rect(0, pdf.get_y()-2, 210, 16, "F")
        pdf.set_fill_color(*ALTIN)
        pdf.rect(0, pdf.get_y()-2, 210, 2, "F")
        pdf.set_text_color(*BEYAZ)
        pdf.set_font("Helvetica","",7)
        pdf.set_x(15)
        pdf.cell(100, 6, "TaxArch  |  ITU ISL4902E  |  2026", ln=False)
        pdf.set_x(110)
        pdf.cell(85, 6, f"Sayfa {pg}/{pdf.page}", align="R")
    pdf.page = pdf.page  # reset

    return bytes(pdf.output())



# ============================================================
# SEKMELER
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["Optimizasyon", "Duyarlılık Analizi", "Kendi Verinizle Analiz", "Geçmiş Analizlerim"])

# ============================================================
# TAB 1: OPTİMİZASYON
# ============================================================
with tab1:
    if not optimize_et:
        st.info("Sol panelden parametreleri ayarlayın ve **Optimize Et** butonuna basın.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Gini", f"{baseline_gini:.4f}")
        col2.metric("Baseline Ort. Vergi", f"{baseline_ort:,.0f} TL")
        col3.metric("Mevcut Dilim Sayısı", "5")

        st.subheader("Mevcut Vergi Sistemi")
        tam = np.concatenate([[0], ESKI_SINIRLAR, [np.inf]])
        rows = []
        for i in range(5):
            aralik = (f"{tam[i]:,.0f} TL ve üzeri" if np.isinf(tam[i+1])
                      else f"{tam[i]:,.0f} – {tam[i+1]:,.0f} TL")
            rows.append({"Dilim": i+1, "Gelir Aralığı": aralik,
                         "Oran": f"%{MEVCUT_ORANLAR[i]*100:.0f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        if not senaryo:
            st.error("Lütfen en az bir değişken seçin.")
            st.stop()

        t0 = time.time()
        # Yüzde tabanlı sınır farkı: her sınır için önceki sınırın %'si kadar boşluk
        if min_sinir_fark is None:
            # İlk dilim için min_sinir'in yüzdesi; sonraki dilimler optimizasyonda dinamik hesaplanır
            min_sinir_fark = int(min_sinir * min_sinir_fark_pct / 100) if min_sinir > 0 else 10000
        with st.spinner("Optimizasyon çalışıyor, lütfen bekleyin..."):
            opt_oranlar, opt_sinirlar = calistir_optimizasyon_v2(
                opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
                dilim_sayisi, min_oran, max_oran,
                min_oran_fark, max_oran_fark,
                min_sinir, son_dilim_min, min_sinir_fark, butce_tolerans)
        sure = time.time() - t0

        baseline_ov = vergi_hesapla(MEVCUT_ORANLAR, ESKI_SINIRLAR)
        opt_ov      = vergi_hesapla(opt_oranlar, opt_sinirlar)
        opt_gini    = gini_hesapla(opt_ov)
        opt_ort     = ort_vergi_hesapla(opt_ov)
        iyilesme    = baseline_gini - opt_gini

        st.caption(f"Optimizasyon {sure:.1f} saniyede tamamlandı.")

        # Analiz sonucunu veritabanına kaydet
        analiz_kaydet(
            kullanici_id  = st.session_state.get("kullanici_id", -1),
            senaryo       = senaryo if senaryo else "Özel",
            opt_gini      = opt_gini,
            baseline_gini = baseline_gini,
            opt_oranlar   = opt_oranlar,
            opt_sinirlar  = opt_sinirlar,
            parametreler  = {
                "butce_tolerans":     butce_tolerans,
                "dilim_sayisi":       dilim_sayisi,
                "min_oran":           min_oran,
                "max_oran":           max_oran,
                "min_oran_fark":      min_oran_fark,
                "max_oran_fark":      max_oran_fark,
                "min_sinir":          min_sinir         if (opt_sinirlar_sec or opt_dilim_sec) else None,
                "son_dilim_min":      son_dilim_min     if (opt_sinirlar_sec or opt_dilim_sec) else None,
                "min_sinir_fark_pct": min_sinir_fark_pct if (opt_sinirlar_sec or opt_dilim_sec) else None,
                "opt_sinirlar":       list(opt_sinirlar),
                "opt_oranlar_full":   list(opt_oranlar),
                "optimize_oranlar":   opt_oranlar_sec,
                "optimize_sinirlar":  opt_sinirlar_sec,
                "optimize_dilim":     opt_dilim_sec,
            }
        )

        _butce_pct = (opt_ort - baseline_ort) / baseline_ort * 100

        st.subheader("Sonuçlar")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Baseline Gini", f"{baseline_gini:.4f}")
        with col2:
            st.metric("Optimal Gini", f"{opt_gini:.4f}",
                      delta=f"{-iyilesme:.4f}", delta_color="inverse")
        with col3:
            _gini_pct = (opt_gini - baseline_gini) / baseline_gini * 100
            st.metric("Gini Değişimi", f"{_gini_pct:+.2f}%")
        with col4:
            st.metric("Mevcut Ort. Vergi", f"{baseline_ort:,.0f} TL")
        with col5:
            st.metric("Yeni Ort. Vergi", f"{opt_ort:,.0f} TL",
                      delta=f"{opt_ort - baseline_ort:+,.0f} TL")
        with col6:
            st.metric("Bütçe Değişimi", f"{_butce_pct:+.2f}%")

        st.markdown("---")
        col_tablo, col_grafik = st.columns([1, 1])

        with col_tablo:
            st.subheader("Vergi Dilim Karşılaştırması")
            tam_eski = np.concatenate([[0], ESKI_SINIRLAR, [np.inf]])
            tam_yeni = np.concatenate([[0], opt_sinirlar, [np.inf]])
            k = len(opt_oranlar)
            dilim_rows = []

            if not opt_sinirlar_sec and not opt_oranlar_sec:
                _dilim_mod = "sadece_sinir"
            elif opt_sinirlar_sec and not opt_oranlar_sec:
                _dilim_mod = "sadece_sinir"
            else:
                _dilim_mod = "diger"

            if _dilim_mod == "sadece_sinir":
                for i in range(k):
                    eski_aralik = (f"{tam_eski[i]:,.0f} TL ve üzeri" if np.isinf(tam_eski[i+1])
                                   else f"{tam_eski[i]:,.0f} – {tam_eski[i+1]:,.0f} TL") if i+1 < len(tam_eski) else "—"
                    yeni_aralik = (f"{tam_yeni[i]:,.0f} TL ve üzeri" if np.isinf(tam_yeni[i+1])
                                   else f"{tam_yeni[i]:,.0f} – {tam_yeni[i+1]:,.0f} TL")
                    dilim_rows.append({
                        "Dilim": i+1,
                        "Eski Aralik": eski_aralik,
                        "Yeni Aralik": yeni_aralik,
                        "Oran": f"%{opt_oranlar[i]*100:.0f}",
                    })
            else:
                for i in range(k):
                    # Eski aralık: sadece mevcut 5 dilim için, fazlası için — göster
                    if i+1 < len(tam_eski):
                        eski_aralik = (f"{tam_eski[i]:,.0f} TL ve üzeri" if np.isinf(tam_eski[i+1])
                                       else f"{tam_eski[i]:,.0f} – {tam_eski[i+1]:,.0f} TL")
                    else:
                        eski_aralik = "—"
                    yeni_aralik = (f"{tam_yeni[i]:,.0f} TL ve üzeri" if np.isinf(tam_yeni[i+1])
                                   else f"{tam_yeni[i]:,.0f} – {tam_yeni[i+1]:,.0f} TL")
                    eski_oran = f"%{MEVCUT_ORANLAR[i]*100:.0f}" if i < len(MEVCUT_ORANLAR) else "—"
                    fark = ((opt_oranlar[i] - MEVCUT_ORANLAR[i]) * 100
                            if i < len(MEVCUT_ORANLAR) else 0)
                    dilim_rows.append({
                        "Dilim": i+1,
                        "Eski Aralik": eski_aralik,
                        "Eski Oran": eski_oran,
                        "Yeni Aralik": yeni_aralik,
                        "Yeni Oran": f"%{opt_oranlar[i]*100:.1f}",
                        "Fark": f"{fark:+.1f}%"
                    })

            dilim_df = pd.DataFrame(dilim_rows)
            st.dataframe(dilim_df, use_container_width=True, hide_index=True)

            st.subheader("Gelir Grubu Etki Analizi")
            etki_rows = []
            for i in range(n):
                etki_rows.append({
                    "Grup": f"%{(i+1)*5}",
                    "Hane Geliri": f"{gelir_st_haric[i]:,.0f} TL",
                    "Eski Vergi": f"{baseline_ov[i]:,.0f} TL",
                    "Yeni Vergi": f"{opt_ov[i]:,.0f} TL",
                    "Fark": f"{opt_ov[i]-baseline_ov[i]:+,.0f} TL"
                })
            etki_df = pd.DataFrame(etki_rows)
            st.dataframe(etki_df, use_container_width=True, hide_index=True)

        with col_grafik:
            st.subheader("Lorenz Eğrisi")
            cp_base, cg_base = lorenz_egri(baseline_ov)
            cp_opt,  cg_opt  = lorenz_egri(opt_ov)

            fig_lorenz = go.Figure()
            fig_lorenz.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                name='Tam Eşitlik', line=dict(dash='dash', color='gray', width=1)))
            fig_lorenz.add_trace(go.Scatter(x=cp_base, y=cg_base, mode='lines',
                name=f'Mevcut (Gini={baseline_gini:.4f})',
                line=dict(color='red', width=2)))
            fig_lorenz.add_trace(go.Scatter(x=cp_opt, y=cg_opt, mode='lines',
                name=f'Optimal (Gini={opt_gini:.4f})',
                line=dict(color='green', width=2),
                fill='tonexty', fillcolor='rgba(0,200,0,0.15)'))
            fig_lorenz.update_layout(
                xaxis_title="Nüfus Payı", yaxis_title="Gelir Payı",
                height=380, margin=dict(t=10),
                legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig_lorenz, use_container_width=True)

            st.subheader("Gelir Grubuna Göre Vergi Farkı")
            farklar = opt_ov - baseline_ov
            renkler = ['#2ecc71' if f <= 0 else '#e74c3c' for f in farklar]
            gruplar = [f"%{(i+1)*5}" for i in range(n)]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=gruplar, y=farklar, marker_color=renkler))
            fig_bar.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
            fig_bar.update_layout(
                xaxis_title="Gelir Grubu", yaxis_title="Vergi Farkı (TL)",
                height=350, margin=dict(t=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        if opt_oranlar_sec and not opt_sinirlar_sec and not opt_dilim_sec:
            _sno = 1
        elif opt_sinirlar_sec and not opt_oranlar_sec and not opt_dilim_sec:
            _sno = 2
        elif opt_oranlar_sec and opt_sinirlar_sec:
            _sno = 3
        else:
            _sno = 4

        # Otomatik yorum
        _emoji, _etiket, _aciklama, _ozet, _yorumlar = otomatik_yorum(
            _sno, opt_oranlar, opt_sinirlar, opt_gini, opt_ort,
            baseline_gini, baseline_ort, opt_ov, baseline_ov,
            min_oran, max_oran, min_oran_fark, max_oran_fark,
            min_sinir, son_dilim_min, min_sinir_fark, butce_tolerans,
            dilim_sayisi, sure)

        st.subheader("Değerlendirme")
        st.markdown(f"""
        <div style="border-left:4px solid #1e3060; padding:12px 16px;
                    background:#eef2f8; border-radius:0 8px 8px 0; margin-bottom:12px;">
          <div style="font-size:11px; color:#1e3060; font-weight:700;
                      letter-spacing:1px; margin-bottom:6px; text-transform:uppercase;">
            Reform Özeti
          </div>
          <div style="font-size:13px; color:#2c3e50; line-height:1.6;">{_ozet}</div>
        </div>
        """, unsafe_allow_html=True)

        _renk_map = {
            "success": ("#1a472a", "#d4edda", "#27ae60"),
            "warning": ("#7d5a00", "#fff3cd", "#d4af37"),
            "error":   ("#6b1a1a", "#fde8e8", "#c0392b"),
            "info":    ("#1a3a5c", "#dde8f5", "#2980b9"),
        }
        for tip, baslik, metin in _yorumlar:
            tc, bg, border = _renk_map.get(tip, _renk_map["info"])
            st.markdown(f"""
            <div style="background:{bg};border-left:3px solid {border};
                        border-radius:0 6px 6px 0;padding:8px 14px;margin-bottom:6px;">
              <span style="font-size:13px;font-weight:700;color:{border};">{baslik}:</span>
              <span style="font-size:13px;color:{tc};"> {metin}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        pdf_bytes = pdf_olustur(
            senaryo, opt_oranlar, opt_sinirlar,
            opt_gini, opt_ort, baseline_gini, baseline_ort,
            etki_df, dilim_df, cp_base, cg_base, cp_opt, cg_opt)
        st.download_button(
            label="PDF Raporu İndir",
            data=pdf_bytes,
            file_name=f"vergi_optimizasyon_{_sno}.pdf",
            mime="application/pdf")


# ============================================================
# TAB 2: DUYARLILIK ANALİZİ
# ============================================================
with tab2:
    st.subheader("Duyarlılık Analizi")
    st.markdown("""
    Seçilen parametrenin belirli bir aralıkta değiştirilmesiyle optimal Gini'nin
    nasıl değiştiğini gösterir. Her nokta için ayrı optimizasyon çalışır.
    """)

    col_ayar, col_sonuc = st.columns([1, 2])

    with col_ayar:
        st.markdown("**Analiz Ayarları**")

        parametre = st.selectbox("Analiz edilecek parametre", [
            "Bütçe Toleransı (%)",
            "Min Oran (%)",
            "Max Oran (%)",
            "Oranlar Arası Min Fark (%)",
            "Sınırlar Arası Min Fark (TL)",
        ])

        col_a, col_b = st.columns(2)
        with col_a:
            p_min = st.number_input("Başlangıç değeri", value=0.0 if "Tolerans" in parametre or "Fark" in parametre and "TL" not in parametre else (50000.0 if "TL" in parametre else 5.0))
        with col_b:
            p_max = st.number_input("Bitiş değeri", value=5.0 if "Tolerans" in parametre or "Fark" in parametre and "TL" not in parametre else (300000.0 if "TL" in parametre else 45.0))

        n_nokta = st.slider("Nokta sayısı", min_value=3, max_value=10, value=5)
        st.info(f"{n_nokta} optimizasyon çalışacak, biraz zaman alabilir.")

        duy_calistir = st.button("Analizi Çalıştır", type="primary", use_container_width=True)

    with col_sonuc:
        if duy_calistir:
            degerler = np.linspace(p_min, p_max, n_nokta)
            gini_sonuclari = []
            progress = st.progress(0, text="Hesaplanıyor...")

            for idx, deger in enumerate(degerler):
                # Sabit varsayılan parametreler
                p = {
                    "butce_tolerans": 0.02,
                    "min_oran": 0.10,
                    "max_oran": 0.45,
                    "min_oran_fark": 0.03,
                    "max_oran_fark": 0.13,
                    "min_sinir": 50000,
                    "son_dilim_min": 2000000,
                    "min_sinir_fark": 100000,
                    "dilim_sayisi": 5,
                }

                # Seçilen parametreyi değiştir
                if parametre == "Bütçe Toleransı (%)":
                    p["butce_tolerans"] = deger / 100
                elif parametre == "Min Oran (%)":
                    p["min_oran"] = deger / 100
                elif parametre == "Max Oran (%)":
                    p["max_oran"] = deger / 100
                elif parametre == "Oranlar Arası Min Fark (%)":
                    p["min_oran_fark"] = deger / 100
                elif parametre == "Sınırlar Arası Min Fark (TL)":
                    p["min_sinir_fark"] = int(deger)

                try:
                    opt_o, opt_s = calistir_optimizasyon(
                        1, p["dilim_sayisi"],
                        p["min_oran"], p["max_oran"],
                        p["min_oran_fark"], p["max_oran_fark"],
                        p["min_sinir"], p["son_dilim_min"],
                        p["min_sinir_fark"], p["butce_tolerans"])
                    g = gini_hesapla(vergi_hesapla(opt_o, opt_s))
                except:
                    g = np.nan

                gini_sonuclari.append(g)
                progress.progress((idx+1)/n_nokta,
                                  text=f"Hesaplanıyor... {idx+1}/{n_nokta}")

            progress.empty()

            # Grafik
            gecerli = [(d, g) for d, g in zip(degerler, gini_sonuclari)
                       if not np.isnan(g)]
            if gecerli:
                x_vals = [v[0] for v in gecerli]
                y_vals = [v[1] for v in gecerli]

                fig_duy = go.Figure()
                fig_duy.add_hline(y=baseline_gini, line_dash="dash",
                                  line_color="red", annotation_text="Baseline Gini")
                fig_duy.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='lines+markers',
                    line=dict(color='#2980b9', width=2),
                    marker=dict(size=8, color='#2980b9'),
                    name="Optimal Gini"))
                fig_duy.update_layout(
                    xaxis_title=parametre,
                    yaxis_title="Optimal Gini Katsayısı",
                    height=400,
                    title=f"{parametre} → Optimal Gini İlişkisi ({duy_senaryo})")
                st.plotly_chart(fig_duy, use_container_width=True)

                # Sonuç tablosu
                tablo = pd.DataFrame({
                    parametre: [f"{d:.2f}" for d in x_vals],
                    "Optimal Gini": [f"{g:.4f}" for g in y_vals],
                    "İyileşme": [f"{baseline_gini-g:.4f}" for g in y_vals],
                    "İyileşme (%)": [f"%{(baseline_gini-g)/baseline_gini*100:.2f}" for g in y_vals],
                })
                st.dataframe(tablo, use_container_width=True, hide_index=True)

                # Duyarlılık yorumu
                st.markdown("---")
                st.markdown("**📝 Duyarlılık Değerlendirmesi**")

                min_g   = min(y_vals)
                max_g   = max(y_vals)
                aralik  = max_g - min_g
                min_idx = y_vals.index(min_g)
                min_x   = x_vals[min_idx]

                # Eğri şeklini tespit et
                if aralik < 0.001:
                    egri_tipi = "duz"
                elif y_vals[-1] < y_vals[0]:
                    egri_tipi = "azalan"
                elif y_vals[-1] > y_vals[0]:
                    egri_tipi = "artan"
                elif min_idx not in [0, len(y_vals)-1]:
                    egri_tipi = "u_sekli"
                else:
                    egri_tipi = "diger"

                if egri_tipi == "duz":
                    st.info(
                        f"**Sınırlı Etki:** Seçilen parametre aralığında "
                        f"({p_min:.1f} – {p_max:.1f}) Gini katsayısı "
                        f"yalnızca {aralik:.4f} değişmiştir. "
                        f"Bu parametre, mevcut kısıt yapısı altında "
                        f"eşitsizlik üzerinde sınırlı belirleyiciliğe sahiptir.")

                elif egri_tipi == "azalan":
                    st.success(
                        f"**Tutarlı İyileşme:** Parametre artışıyla birlikte "
                        f"Gini katsayısı {max_g:.4f}'ten {min_g:.4f}'e gerilemiştir "
                        f"(%{(max_g-min_g)/max_g*100:.2f} iyileşme). "
                        f"En düşük eşitsizlik {min_x:.2f} değerinde elde edilmiştir. "
                        f"Parametre artışının yeniden dağılım kapasitesini "
                        f"artırdığı gözlemlenmektedir.")

                elif egri_tipi == "artan":
                    st.warning(
                        f"**Ters Etki:** Parametre artışıyla birlikte "
                        f"Gini katsayısı {min_g:.4f}'ten {max_g:.4f}'e yükselmiştir. "
                        f"Bu parametre için daha düşük değerler tercih edilmelidir. "
                        f"En iyi sonuç {min_x:.2f} değerinde elde edilmiştir.")

                elif egri_tipi == "u_sekli":
                    st.info(
                        f"**Optimal Nokta Tespit Edildi:** Gini katsayısı "
                        f"parametre değeri {min_x:.2f} iken minimum düzeye "
                        f"({min_g:.4f}) ulaşmıştır. Bu noktanın ötesinde "
                        f"ya da gerisinde eşitsizlik artış eğilimi göstermektedir. "
                        f"Politika tasarımında {min_x:.2f} civarındaki değerler "
                        f"tercih edilmelidir.")

                else:
                    st.info(
                        f"**Değerlendirme:** En düşük Gini katsayısı "
                        f"({min_g:.4f}), parametre değeri {min_x:.2f} iken "
                        f"elde edilmiştir. Analiz aralığındaki toplam "
                        f"Gini değişimi {aralik:.4f} olarak hesaplanmıştır.")

            else:
                st.error("Hiçbir nokta için geçerli sonuç bulunamadı. Parametreleri kontrol edin.")
        else:
            st.info("Sol taraftan parametreleri ayarlayın ve **Analizi Çalıştır** butonuna basın.")


# ============================================================
# TAB 3: KENDİ VERİNİZLE ANALİZ
# ============================================================
with tab3:
    st.subheader("Kendi Verinizle Analiz")
    st.markdown("""
    Hane bazlı gelir verinizi CSV formatında yükleyin.
    Her satır bir hane, her kolon bir bireyin brüt yıllık geliri (TL).
    Son kolon **eşdeğer hane büyüklüğü** — OECD ölçeğine göre kendiniz hesaplayın.

    **OECD Eşdeğerlik Ölçeği:**
    - 1. yetişkin → **1.0**
    - Her ek yetişkin → **+0.5**
    - Her çocuk (18 yaş altı) → **+0.3**

    *Örnek: 2 yetişkin + 1 çocuk → 1.0 + 0.5 + 0.3 = **1.8***

    **Metodoloji:**
    1. Her bireyin brüt geliri mevcut/optimal vergi sistemine sokulur → bireysel net gelir
    2. Hanedeki tüm bireylerin net gelirleri toplanır → hane net geliri
    3. Eşdeğer hane büyüklüğüne bölünür → eşdeğer kişi başı gelir
    4. Bu değerler üzerinden Gini hesaplanır

    > Gelir getirmeyen hane üyeleri (çocuklar, çalışmayan yetişkinler) eşdeğer büyüklüğe dahil edilir ancak gelir kolonlarına **0** girilir.
    """)

    # Örnek dosya
    ornek = ("gelir_1,gelir_2,gelir_3,esit_hane_buyuklugu\n"
             "150000,80000,0,1.5\n"
             "280000,0,0,1.0\n"
             "450000,120000,95000,2.0\n"
             "600000,350000,0,1.5\n"
             "1200000,0,0,1.0\n"
             "380000,210000,0,1.5\n"
             "750000,420000,180000,2.0\n"
             "95000,0,0,1.0\n"
             "520000,310000,0,1.5\n"
             "2500000,0,0,1.0")
    st.download_button("📥 Örnek CSV İndir", data=ornek,
                       file_name="ornek_hane_verisi.csv", mime="text/csv")

    st.markdown("---")

    yuklu_dosya = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"],
        help="Kolonlar: gelir_1, gelir_2, ... (0 = o birey yok), esit_hane_buyuklugu")

    if yuklu_dosya is not None:
        try:
            df = pd.read_csv(yuklu_dosya)

            # Format kontrolü
            if "esit_hane_buyuklugu" not in df.columns:
                st.error("'esit_hane_buyuklugu' kolonu bulunamadı.")
                st.stop()

            gelir_kolonlari = [c for c in df.columns if c.startswith("gelir_")]
            if not gelir_kolonlari:
                st.error("Gelir kolonları bulunamadı. 'gelir_1', 'gelir_2' formatında olmalı.")
                st.stop()

            n_hane = len(df)
            st.success(f"{n_hane:,} haneli veri yüklendi.")

            # İstatistikler
            hane_brut = df[gelir_kolonlari].sum(axis=1)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Hane Sayısı", f"{n_hane:,}")
            col2.metric("Ort. Hane Geliri", f"{hane_brut.mean():,.0f} TL")
            col3.metric("Medyan Hane Geliri", f"{hane_brut.median():,.0f} TL")
            col4.metric("Maks Hane Geliri", f"{hane_brut.max():,.0f} TL")

            st.markdown("---")

            # Vergi hesaplama fonksiyonu
            def vergi_birey(gelir, oranlar, sinirlar, istisna=45151):
                if gelir <= 0:
                    return 0.0
                tam = np.concatenate([[0], sinirlar, [np.inf]])
                matrah = gelir * 0.85
                vh = 0.0
                for j in range(len(oranlar)):
                    alt = tam[j]; ust = tam[j+1]
                    if matrah <= alt: break
                    vh += (min(matrah, ust) - alt) * oranlar[j]
                return max(vh - istisna, 0)

            def esit_gelir_hesapla(df_hane, gelir_kols, oranlar, sinirlar):
                """Her hane için eşdeğer kişi başı net gelir hesapla"""
                esit_gelirler = np.zeros(len(df_hane))
                for i, (_, row) in enumerate(df_hane.iterrows()):
                    hane_net = 0.0
                    for kol in gelir_kols:
                        g = float(row[kol])
                        v = vergi_birey(g, oranlar, sinirlar)
                        hane_net += (g - v)
                    esit_gelirler[i] = max(hane_net, 0) / float(row["esit_hane_buyuklugu"])
                return esit_gelirler

            def gini_hesapla_hane(esit_gelirler):
                y = np.sort(esit_gelirler)
                n = len(y)
                npay = np.ones(n) / n
                cg = np.cumsum(y * npay) / np.sum(y * npay)
                cp = np.cumsum(npay)
                return 1 - 2 * np.trapezoid(
                    np.concatenate([[0], cg]),
                    np.concatenate([[0], cp]))

            # Baseline
            esit_base = esit_gelir_hesapla(df, gelir_kolonlari,
                                            MEVCUT_ORANLAR, ESKI_SINIRLAR)
            baseline_gini_h = gini_hesapla_hane(esit_base)
            baseline_ort_h  = np.mean(esit_base)

            st.info(f"**Mevcut sistemde Gini:** {baseline_gini_h:.4f}  |  "
                    f"**Ort. Eşdeğer Gelir:** {baseline_ort_h:,.0f} TL")

            if st.button("Bu Veriyle Optimize Et", type="primary"):

                # Sol paneldeki secimlere gore karar degiskenleri
                k = dilim_sayisi if opt_dilim_sec else 5

                def unpack_hane(x):
                    idx = 0
                    if opt_oranlar_sec or opt_dilim_sec:
                        _o = x[idx:idx+k]; idx += k
                    else:
                        _o = np.array(MEVCUT_ORANLAR[:k]) if k<=5 else np.linspace(min_oran, max_oran, k)
                    if opt_sinirlar_sec or opt_dilim_sec:
                        _s = x[idx:idx+k-1]
                    else:
                        _s = np.array(ESKI_SINIRLAR[:k-1]) if k<=5 else np.linspace(50000, son_dilim_min, k-1)
                    return _o, _s

                def hedef_hane(x):
                    _o, _s = unpack_hane(x)
                    eg = esit_gelir_hesapla(df, gelir_kolonlari, _o, _s)
                    return gini_hesapla_hane(eg)

                def ort_fn_hane(x):
                    _o, _s = unpack_hane(x)
                    eg = esit_gelir_hesapla(df, gelir_kolonlari, _o, _s)
                    return np.mean(eg)

                # Bounds ve x0
                bounds_h = []; x0_h = []
                if opt_oranlar_sec or opt_dilim_sec:
                    init_o = np.array(MEVCUT_ORANLAR[:k]) if k<=5 else np.linspace(min_oran, max_oran, k)
                    bounds_h += [(min_oran, max_oran)] * k
                    x0_h.append(init_o)
                if opt_sinirlar_sec or opt_dilim_sec:
                    init_s = np.array(ESKI_SINIRLAR[:k-1]) if k<=5 else np.linspace(50000, son_dilim_min, k-1)
                    bounds_h += [(min_sinir, son_dilim_min*4)] * (k-2) + [(son_dilim_min, son_dilim_min*4)]
                    x0_h.append(init_s)

                if not bounds_h:
                    bounds_h = [(min_oran, max_oran)] * k
                    x0_h = [np.array(MEVCUT_ORANLAR[:k])]

                x0_h = np.concatenate(x0_h)
                tol  = butce_tolerans

                kisitlar_h = [
                    {'type':'ineq','fun': lambda x: ort_fn_hane(x) - baseline_ort_h*(1-tol)},
                    {'type':'ineq','fun': lambda x: baseline_ort_h*(1+tol) - ort_fn_hane(x)},
                ]
                if opt_oranlar_sec or opt_dilim_sec:
                    for i in range(k-1):
                        kisitlar_h.append({'type':'ineq',
                            'fun': lambda x,i=i: x[i+1]-x[i]-min_oran_fark})
                        kisitlar_h.append({'type':'ineq',
                            'fun': lambda x,i=i: max_oran_fark-(x[i+1]-x[i])})
                if opt_sinirlar_sec or opt_dilim_sec:
                    oo = k if (opt_oranlar_sec or opt_dilim_sec) else 0
                    for i in range(k-2):
                        kisitlar_h.append({'type':'ineq',
                            'fun': lambda x,i=i,o=oo: x[o+i+1]-x[o+i]-min_sinir_fark})

                kisit_butce_h = NonlinearConstraint(
                    ort_fn_hane, baseline_ort_h*(1-tol), baseline_ort_h*(1+tol))
                de_cons_h = [kisit_butce_h]
                if opt_oranlar_sec or opt_dilim_sec:
                    A = np.zeros((k-1, len(bounds_h)))
                    for i in range(k-1): A[i,i]=-1; A[i,i+1]=1
                    de_cons_h.append(LinearConstraint(A, min_oran_fark, max_oran_fark))

                with st.spinner("Optimizasyon çalışıyor..."):
                    # Coklu baslangic noktasi ile SLSQP
                    en_iyi = None
                    en_iyi_gini = np.inf
                    for _x0 in [x0_h,
                                x0_h * np.random.uniform(0.9, 1.1, len(x0_h)),
                                x0_h * np.random.uniform(0.85, 1.15, len(x0_h))]:
                        _x0 = np.clip(_x0, [b[0] for b in bounds_h], [b[1] for b in bounds_h])
                        try:
                            _r = minimize(hedef_hane, _x0, method='SLSQP',
                                         bounds=bounds_h, constraints=kisitlar_h,
                                         options={'maxiter':1000,'ftol':1e-8})
                            if _r.fun < en_iyi_gini:
                                en_iyi_gini = _r.fun
                                en_iyi = _r
                        except:
                            pass
                    sonuc2 = en_iyi

                opt_oranlar_h, opt_sinirlar_h = unpack_hane(sonuc2.x)
                esit_opt = esit_gelir_hesapla(df, gelir_kolonlari,
                                               opt_oranlar_h, opt_sinirlar_h)
                opt_gini_h = gini_hesapla_hane(esit_opt)
                opt_ort_h  = np.mean(esit_opt)
                iyilesme_h = baseline_gini_h - opt_gini_h

                # Metrikler
                col1, col2, col3 = st.columns(3)
                col1.metric("Baseline Gini", f"{baseline_gini_h:.4f}")
                col2.metric("Optimal Gini", f"{opt_gini_h:.4f}",
                            delta=f"{-iyilesme_h:.4f}", delta_color="inverse")
                col3.metric("Gini İyileşme (%)",
                            f"%{(iyilesme_h/baseline_gini_h)*100:.2f}")

                # Dilim tablosu
                st.subheader("Optimal Vergi Oranları")
                tam = np.concatenate([[0], ESKI_SINIRLAR, [np.inf]])
                rows = []
                for i in range(k):
                    aralik = (f"{tam[i]:,.0f} TL ve üzeri" if np.isinf(tam[i+1])
                              else f"{tam[i]:,.0f} – {tam[i+1]:,.0f} TL")
                    rows.append({
                        "Dilim": i+1,
                        "Aralık": aralik,
                        "Mevcut Oran": f"%{MEVCUT_ORANLAR[i]*100:.0f}",
                        "Optimal Oran": f"%{opt_oranlar_h[i]*100:.1f}",
                        "Fark": f"{(opt_oranlar_h[i]-MEVCUT_ORANLAR[i])*100:+.1f}%"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Lorenz
                st.subheader("Lorenz Eğrisi")
                def lorenz_hane(eg):
                    y = np.sort(eg)
                    n = len(y)
                    npay = np.ones(n) / n
                    cg = np.cumsum(y * npay) / np.sum(y * npay)
                    cp = np.cumsum(npay)
                    return np.concatenate([[0], cp]), np.concatenate([[0], cg])

                cp_b, cg_b = lorenz_hane(esit_base)
                cp_o, cg_o = lorenz_hane(esit_opt)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                    name='Tam Eşitlik', line=dict(dash='dash', color='gray')))
                fig.add_trace(go.Scatter(x=cp_b, y=cg_b, mode='lines',
                    name=f'Mevcut (Gini={baseline_gini_h:.4f})',
                    line=dict(color='red', width=2)))
                fig.add_trace(go.Scatter(x=cp_o, y=cg_o, mode='lines',
                    name=f'Optimal (Gini={opt_gini_h:.4f})',
                    line=dict(color='green', width=2),
                    fill='tonexty', fillcolor='rgba(0,200,0,0.15)'))
                fig.update_layout(xaxis_title="Nüfus Payı",
                                  yaxis_title="Gelir Payı",
                                  height=400, margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Hata: {e}")
    else:
        st.info("CSV dosyanızı yükleyerek başlayın.")


# ============================================================
# TAB 4: GEÇMİŞ ANALİZLERİM
# ============================================================
with tab4:
    st.subheader("Geçmiş Analizlerim")

    kid = st.session_state.get("kullanici_id", -1)
    if kid == -1:
        st.info("Demo hesapla giriş yaptınız. Geçmiş analiz kaydı için kayıt olun.")
    else:
        gecmis = gecmis_getir(kid)
        if not gecmis:
            st.info("Henüz kayıtlı analiz yok. Optimizasyon sekmesinden analiz yapın.")
        else:
            st.success(f"Toplam {len(gecmis)} analiz bulundu.")
            st.markdown("---")

            for kayit in gecmis:
                aid, tarih, senaryo, opt_gini, base_gini, iyilesme, \
                    opt_oranlar_json, opt_sinirlar_json, param_json = kayit

                try:
                    params   = json.loads(param_json)        if param_json         else {}
                    oranlar  = json.loads(opt_oranlar_json)  if opt_oranlar_json   else []
                    sinirlar = json.loads(opt_sinirlar_json) if opt_sinirlar_json  else []
                except:
                    params = {}; oranlar = []; sinirlar = []

                opt_oran_sec  = params.get("optimize_oranlar",  False)
                opt_sinir_sec = params.get("optimize_sinirlar", False)
                opt_dilim_sc  = params.get("optimize_dilim",    False)
                # Gerçek dilim sayısı: kaydedilen oran sayısı
                dilim_k = len(oranlar) if oranlar else params.get("dilim_sayisi", 5)

                try:
                    param_parts = [f"Bütçe toleransı: %{params.get('butce_tolerans', 0.02)*100:.1f}",
                                   f"Dilim sayısı: {dilim_k}"]
                    if opt_oran_sec or opt_dilim_sc:
                        param_parts += [
                            f"Min oran: %{params.get('min_oran', 0.10)*100:.0f}",
                            f"Max oran: %{params.get('max_oran', 0.45)*100:.0f}",
                            f"Min oran farkı: %{params.get('min_oran_fark', 0.03)*100:.0f}",
                            f"Max oran farkı: %{params.get('max_oran_fark', 0.13)*100:.0f}",
                        ]
                    if opt_sinir_sec or opt_dilim_sc:
                        if params.get("min_sinir") is not None:
                            param_parts += [
                                f"Min dilim sınırı: {int(params['min_sinir']):,} TL",
                                f"Son dilim min: {int(params.get('son_dilim_min', 2000000)):,} TL",
                                f"Sınır fark oranı: %{params.get('min_sinir_fark_pct', 10)}",
                            ]
                    param_str = "  |  ".join(param_parts)
                except:
                    param_str = "-"

                with st.expander(
                    f"{tarih}  |  {senaryo}  |  Gini: {opt_gini:.4f}  (%{iyilesme:.2f} iyileşme)"):

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Baseline Gini", f"{base_gini:.4f}")
                    c2.metric("Optimal Gini",  f"{opt_gini:.4f}")
                    c3.metric("İyileşme",      f"%{iyilesme:.2f}")

                    st.markdown("**Kullanılan Parametreler:**")
                    st.code(param_str, language=None)

                    # Sınır optimizasyonu
                    if opt_sinir_sec or opt_dilim_sc:
                        if sinirlar:
                            st.markdown("**Optimize Edilen Dilim Sınırları:**")
                            tam_s = np.concatenate([[0], sinirlar, [np.inf]])
                            sinir_rows = []
                            for i in range(len(sinirlar) + 1):
                                aralik = (f"{tam_s[i]:,.0f} TL ve üzeri" if np.isinf(tam_s[i+1])
                                          else f"{tam_s[i]:,.0f} – {tam_s[i+1]:,.0f} TL")
                                oran = f"%{oranlar[i]*100:.1f}" if i < len(oranlar) else "—"
                                sinir_rows.append({"Dilim": i+1, "Yeni Aralık": aralik, "Oran": oran})
                            st.dataframe(pd.DataFrame(sinir_rows), use_container_width=True, hide_index=True)

                    # Oran optimizasyonu
                    if opt_oran_sec or opt_dilim_sc:
                        if oranlar:
                            st.markdown("**Optimize Edilen Vergi Oranları:**")
                            tam_g = (np.concatenate([[0], sinirlar, [np.inf]]) if sinirlar
                                     else np.concatenate([[0], ESKI_SINIRLAR[:max(len(oranlar)-1,0)], [np.inf]]))
                            oran_rows = []
                            for i, o in enumerate(oranlar):
                                if i < len(tam_g) - 1:
                                    aralik = (f"{tam_g[i]:,.0f} TL ve üzeri" if np.isinf(tam_g[i+1])
                                              else f"{tam_g[i]:,.0f} – {tam_g[i+1]:,.0f} TL")
                                else:
                                    aralik = "—"
                                oran_rows.append({"Dilim": i+1, "Aralık": aralik, "Optimal Oran": f"%{o*100:.1f}"})
                            st.dataframe(pd.DataFrame(oran_rows), use_container_width=True, hide_index=True)

                    # Ne optimize edildiği bilinmiyorsa (eski kayıt) ham göster
                    if not opt_oran_sec and not opt_sinir_sec and not opt_dilim_sc:
                        if oranlar:
                            st.markdown("**Kaydedilen Vergi Oranları:**")
                            st.code("  ".join([f"%{o*100:.1f}" for o in oranlar]), language=None)
                        if sinirlar:
                            st.markdown("**Kaydedilen Dilim Sınırları:**")
                            st.code("  ".join([f"{int(s):,} TL" for s in sinirlar]), language=None)

            st.markdown("---")
            st.caption("Son 20 analiz gösterilmektedir. En yeni analiz üstte.")
