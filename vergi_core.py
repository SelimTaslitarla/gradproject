import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

# ============================================================
# VERİ
# ============================================================

hh_buyukluk = np.array([1.86, 1.87, 1.90, 1.90, 1.92, 1.93, 1.95, 1.96, 1.99, 2.01,
                         2.02, 2.03, 2.03, 2.04, 2.03, 2.04, 2.00, 2.00, 2.00, 1.97])

gelir_st_haric = np.array([110726, 182255, 226063, 263127, 300957, 338835, 377922, 418028,
                            464878, 515349, 565170, 618133, 675611, 742842, 817470, 931496,
                            1047661, 1242515, 1556276, 3438954], dtype=float)

sosyal_transfer = np.array([13570, 9907, 9095, 7480, 6378, 5475, 4790, 5000, 5255, 4767,
                             4322, 4139, 3889, 4327, 5074, 5319, 5852, 5366, 6176, 10706], dtype=float)

gelir_dahil = gelir_st_haric + sosyal_transfer

SABIT_ISTISNA  = 45151
MEVCUT_ORANLAR = np.array([0.15, 0.20, 0.27, 0.35, 0.40])
ESKI_SINIRLAR  = np.array([158000, 330000, 800000, 4300000], dtype=float)

n     = len(hh_buyukluk)
n_pay = np.ones(n) / n


# ============================================================
# ÇEKIRDEK FONKSİYONLAR
# ============================================================

def vergi_hesapla(oranlar, sinirlar, istisna=SABIT_ISTISNA):
    """
    Verilen oran ve sınırlara göre her grup için vergi hesaplar.
    Döndürür: odenen_vergi (array, n)
    """
    oranlar  = np.array(oranlar)
    sinirlar = np.array(sinirlar)
    tam      = np.concatenate([[0], sinirlar, [np.inf]])
    
    odenen_vergi = np.zeros(n)
    for i in range(n):
        matrah    = gelir_st_haric[i] * 0.85
        vergi_ham = 0.0
        for j in range(len(oranlar)):
            alt = tam[j]
            ust = tam[j + 1]
            if matrah <= alt:
                break
            vergi_ham += (min(matrah, ust) - alt) * oranlar[j]
        odened = max(vergi_ham - istisna, 0)
        odenen_vergi[i] = odened
    return odenen_vergi


def gini_hesapla(odenen_vergi):
    """
    Ödenen vergiye göre net geliri hesaplar ve Gini katsayısını döndürür.
    """
    net_gelir_kisi = (gelir_dahil - odenen_vergi) / hh_buyukluk
    y  = np.sort(net_gelir_kisi)
    cg = np.cumsum(y * n_pay) / np.sum(y * n_pay)
    cp = np.cumsum(n_pay)
    gini = 1 - 2 * np.trapezoid(np.concatenate([[0], cg]),
                                np.concatenate([[0], cp]))
    return gini


def ort_vergi_hesapla(odenen_vergi):
    """Kişi başı ortalama vergi."""
    return np.mean(odenen_vergi / hh_buyukluk)


def tam_hesap(oranlar, sinirlar, istisna=SABIT_ISTISNA):
    """Gini + ortalama vergi birlikte döndürür."""
    ov   = vergi_hesapla(oranlar, sinirlar, istisna)
    gini = gini_hesapla(ov)
    ort  = ort_vergi_hesapla(ov)
    return gini, ort


# ============================================================
# BASELINE
# ============================================================

if __name__ == "__main__":
    baseline_gini, baseline_ort = tam_hesap(MEVCUT_ORANLAR, ESKI_SINIRLAR)
    print(f"Baseline Gini       : {baseline_gini:.4f}")
    print(f"Baseline Ort. Vergi : {baseline_ort:.0f} TL")
    print()
    print("MATLAB beklenen → Gini: 0.3725")
    print(f"Python sonuç    → Gini: {baseline_gini:.4f}")
    eslesme = "✓ EŞLEŞIYOR" if abs(baseline_gini - 0.3725) < 0.001 else "✗ FARKI VAR"
    print(f"Durum: {eslesme}")
