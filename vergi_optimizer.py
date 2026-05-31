import numpy as np
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint, LinearConstraint
from vergi_core import (tam_hesap, vergi_hesapla, gini_hesapla, ort_vergi_hesapla,
                        MEVCUT_ORANLAR, ESKI_SINIRLAR)

baseline_gini, baseline_ort = tam_hesap(MEVCUT_ORANLAR, ESKI_SINIRLAR)


# ============================================================
# SENARYO 1: SADECE ORANLAR
# ============================================================
def optimize_senaryo1(sabit_sinirlar=ESKI_SINIRLAR):
    k = 5

    def hedef(x):
        return gini_hesapla(vergi_hesapla(x, sabit_sinirlar))

    def ort_fn(x):
        return ort_vergi_hesapla(vergi_hesapla(x, sabit_sinirlar))

    kisit_butce = NonlinearConstraint(ort_fn, baseline_ort * 0.985, baseline_ort * 1.015)
    A = np.zeros((k - 1, k))
    for i in range(k - 1):
        A[i, i] = -1; A[i, i + 1] = 1
    kisit_prog = LinearConstraint(A, 0.03, 0.13)
    bounds = [(0.10, 0.45)] * k

    sonuc = differential_evolution(hedef, bounds,
        constraints=[kisit_butce, kisit_prog],
        seed=42, maxiter=1000, tol=1e-8,
        popsize=20, mutation=(0.5, 1.5), recombination=0.9, polish=True)

    kisitlar = [
        {'type': 'ineq', 'fun': lambda x: ort_fn(x) - baseline_ort * 0.985},
        {'type': 'ineq', 'fun': lambda x: baseline_ort * 1.015 - ort_fn(x)},
    ]
    for i in range(k - 1):
        kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i + 1] - x[i] - 0.03})
        kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: 0.13 - (x[i + 1] - x[i])})

    sonuc2 = minimize(hedef, sonuc.x, method='SLSQP', bounds=bounds,
                      constraints=kisitlar, options={'maxiter': 1000, 'ftol': 1e-10})

    ov = vergi_hesapla(sonuc2.x, sabit_sinirlar)
    return {
        'mod': 'Senaryo 1 - Sadece Oranlar',
        'opt_oranlar': sonuc2.x,
        'opt_sinirlar': sabit_sinirlar,
        'opt_gini': sonuc2.fun,
        'opt_ort': ort_vergi_hesapla(ov),
        'baseline_gini': baseline_gini,
        'iyilesme': baseline_gini - sonuc2.fun,
        'odenen_vergi': ov
    }


# ============================================================
# SENARYO 2: SADECE SINIRLAR
# ============================================================
def optimize_senaryo2(sabit_oranlar=MEVCUT_ORANLAR):
    k = len(sabit_oranlar)

    def hedef(x):
        return gini_hesapla(vergi_hesapla(sabit_oranlar, x))

    def ort_fn(x):
        return ort_vergi_hesapla(vergi_hesapla(sabit_oranlar, x))

    kisit_butce = NonlinearConstraint(ort_fn, baseline_ort * 0.985, baseline_ort * 1.015)
    A = np.zeros((k - 2, k - 1))
    for i in range(k - 2):
        A[i, i] = -1; A[i, i + 1] = 1
    kisit_sinir = LinearConstraint(A, 100000, np.inf)
    bounds = [(50000, 300000), (200000, 600000), (600000, 1500000), (1000000, 8000000)]

    sonuc = differential_evolution(hedef, bounds,
        constraints=[kisit_butce, kisit_sinir],
        seed=42, maxiter=1000, tol=1e-8,
        popsize=20, mutation=(0.5, 1.5), recombination=0.9, polish=True)

    kisitlar = [
        {'type': 'ineq', 'fun': lambda x: ort_fn(x) - baseline_ort * 0.985},
        {'type': 'ineq', 'fun': lambda x: baseline_ort * 1.015 - ort_fn(x)},
    ]
    for i in range(k - 2):
        kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i + 1] - x[i] - 100000})

    sonuc2 = minimize(hedef, sonuc.x, method='SLSQP', bounds=bounds,
                      constraints=kisitlar, options={'maxiter': 1000, 'ftol': 1e-10})

    ov = vergi_hesapla(sabit_oranlar, sonuc2.x)
    return {
        'mod': 'Senaryo 2 - Sadece Sinirlar',
        'opt_oranlar': sabit_oranlar,
        'opt_sinirlar': sonuc2.x,
        'opt_gini': sonuc2.fun,
        'opt_ort': ort_vergi_hesapla(ov),
        'baseline_gini': baseline_gini,
        'iyilesme': baseline_gini - sonuc2.fun,
        'odenen_vergi': ov
    }


# ============================================================
# SENARYO 3: JOINT (ORANLAR + SINIRLAR)
# ============================================================
def optimize_senaryo3():
    k = 5

    def hedef(x):
        return gini_hesapla(vergi_hesapla(x[:k], x[k:]))

    def ort_fn(x):
        return ort_vergi_hesapla(vergi_hesapla(x[:k], x[k:]))

    kisit_butce = NonlinearConstraint(ort_fn, baseline_ort * 0.985, baseline_ort * 1.015)
    toplam = (k - 1) + (k - 2)
    A = np.zeros((toplam, 2 * k - 1))
    for i in range(k - 1):
        A[i, i] = -1; A[i, i + 1] = 1
    for i in range(k - 2):
        A[k - 1 + i, k + i] = -1; A[k - 1 + i, k + i + 1] = 1
    alt = np.concatenate([np.full(k - 1, 0.03), np.full(k - 2, 100000)])
    ust = np.concatenate([np.full(k - 1, 0.13), np.full(k - 2, np.inf)])
    kisit_yapi = LinearConstraint(A, alt, ust)
    bounds = [(0.10, 0.45)] * k + [(50000, 300000), (200000, 600000),
                                    (600000, 2000000), (1000000, 4000000)]

    sonuc = differential_evolution(hedef, bounds,
        constraints=[kisit_butce, kisit_yapi],
        seed=42, maxiter=1500, tol=1e-8,
        popsize=25, mutation=(0.5, 1.5), recombination=0.9, polish=True)

    kisitlar = [
        {'type': 'ineq', 'fun': lambda x: ort_fn(x) - baseline_ort * 0.985},
        {'type': 'ineq', 'fun': lambda x: baseline_ort * 1.015 - ort_fn(x)},
    ]
    for i in range(k - 1):
        kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i + 1] - x[i] - 0.03})
        kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: 0.13 - (x[i + 1] - x[i])})
    for i in range(k - 2):
        kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[k + i + 1] - x[k + i] - 100000})

    sonuc2 = minimize(hedef, sonuc.x, method='SLSQP', bounds=bounds,
                      constraints=kisitlar, options={'maxiter': 1000, 'ftol': 1e-10})

    x = sonuc2.x
    ov = vergi_hesapla(x[:k], x[k:])
    return {
        'mod': 'Senaryo 3 - Joint (Oranlar + Sinirlar)',
        'opt_oranlar': x[:k],
        'opt_sinirlar': x[k:],
        'opt_gini': sonuc2.fun,
        'opt_ort': ort_vergi_hesapla(ov),
        'baseline_gini': baseline_gini,
        'iyilesme': baseline_gini - sonuc2.fun,
        'odenen_vergi': ov
    }


# ============================================================
# SENARYO 4: DILIM SAYISI OPTIMIZASYONU (SLSQP, coklu baslangic)
# ============================================================
def optimize_senaryo4(dilim_sayisi=9):
    k = dilim_sayisi

    def hedef(x):
        return gini_hesapla(vergi_hesapla(x[:k], x[k:]))

    def ort_fn(x):
        return ort_vergi_hesapla(vergi_hesapla(x[:k], x[k:]))

    kisitlar = [
        {'type': 'ineq', 'fun': lambda x: ort_fn(x) - baseline_ort * 0.985},
        {'type': 'ineq', 'fun': lambda x: baseline_ort * 1.015 - ort_fn(x)},
    ]
    for i in range(k - 1):
        kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i + 1] - x[i] - 0.03})
    for i in range(k - 2):
        kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[k + i + 1] - x[k + i] - 50000})

    bounds = [(0.10, 0.45)] * k + [(50000, 8000000)] * (k - 1)

    # Birden fazla baslangic noktasi - en iyisini al
    x0_listesi = [
        np.concatenate([np.array([0.10, 0.13, 0.17, 0.21, 0.25, 0.30, 0.35, 0.40, 0.45]),
                        np.array([120000, 220000, 370000, 520000, 720000, 1020000, 1520000, 2520000])]),
        np.concatenate([np.array([0.10, 0.14, 0.18, 0.22, 0.26, 0.31, 0.36, 0.41, 0.45]),
                        np.array([150000, 300000, 500000, 700000, 1000000, 1500000, 2500000, 4000000])]),
        np.concatenate([np.array([0.10, 0.13, 0.16, 0.20, 0.24, 0.28, 0.33, 0.38, 0.44]),
                        np.array([200000, 400000, 600000, 900000, 1200000, 1800000, 2800000, 4500000])]),
    ]

    en_iyi = None
    en_iyi_gini = np.inf
    for x0 in x0_listesi:
        sonuc = minimize(hedef, x0, method='SLSQP', bounds=bounds,
                         constraints=kisitlar, options={'maxiter': 2000, 'ftol': 1e-10})
        if sonuc.fun < en_iyi_gini:
            en_iyi_gini = sonuc.fun
            en_iyi = sonuc

    x = en_iyi.x
    ov = vergi_hesapla(x[:k], x[k:])
    return {
        'mod': f'Senaryo 4 - {k} Dilim Optimizasyonu',
        'opt_oranlar': x[:k],
        'opt_sinirlar': x[k:],
        'opt_gini': en_iyi_gini,
        'opt_ort': ort_vergi_hesapla(ov),
        'baseline_gini': baseline_gini,
        'iyilesme': baseline_gini - en_iyi_gini,
        'odenen_vergi': ov
    }


# ============================================================
# YAZDIR
# ============================================================
def yazdir(sonuc):
    print(f"\n{'=' * 55}")
    print(f"{sonuc['mod']}")
    print(f"{'=' * 55}")
    print(f"Eski Gini  : {sonuc['baseline_gini']:.4f}")
    print(f"YENİ Gini  : {sonuc['opt_gini']:.4f}  (İyilesme: {sonuc['iyilesme']:+.4f})")
    print(f"Ort. Vergi : {sonuc['opt_ort']:.0f} TL  (Baseline: {baseline_ort:.0f} TL)")

    oranlar  = sonuc['opt_oranlar']
    sinirlar = sonuc['opt_sinirlar']
    tam = np.concatenate([[0], sinirlar, [np.inf]])

    print(f"\n{'Dilim':<6} | {'Aralik':<40} | Oran")
    print("-" * 55)
    for i in range(len(oranlar)):
        if np.isinf(tam[i + 1]):
            aralik = f"{tam[i]:>12,.0f} TL ve uzeri"
        else:
            aralik = f"{tam[i]:>12,.0f} - {tam[i + 1]:>12,.0f} TL"
        print(f"{i + 1:<6} | {aralik:<40} | %{oranlar[i] * 100:.1f}")


# ============================================================
# CALISTIR
# ============================================================
if __name__ == "__main__":
    print(f"Baseline Gini: {baseline_gini:.4f}  |  Baseline Ort. Vergi: {baseline_ort:.0f} TL\n")

    print("Senaryo 1 calisıyor...")
    s1 = optimize_senaryo1()
    yazdir(s1)

    print("\nSenaryo 2 calisıyor...")
    s2 = optimize_senaryo2()
    yazdir(s2)

    print("\nSenaryo 3 calisıyor...")
    s3 = optimize_senaryo3()
    yazdir(s3)

    print("\nSenaryo 4 calisıyor...")
    s4 = optimize_senaryo4(dilim_sayisi=9)
    yazdir(s4)

    print("\n\n=== OZET ===")
    print(f"{'Senaryo':<42} | {'Gini':>8} | {'İyilesme':>10}")
    print("-" * 65)
    print(f"{'Baseline':<42} | {baseline_gini:>8.4f} | {'':>10}")
    for s in [s1, s2, s3, s4]:
        print(f"{s['mod']:<42} | {s['opt_gini']:>8.4f} | {s['iyilesme']:>+10.4f}")
