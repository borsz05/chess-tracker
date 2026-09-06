# Vision pipeline — zajszűrő rétegek, küszöbök és a mérések (10. szakasz)

Állapot: 2026-09-06. Ez a dokumentum rögzíti, **miért** azok a stabilizer és a
tracker küszöbei, amik, és **hogyan kell újramérni** őket, ha a modell, a
kamera vagy a fényviszonyok változnak. A küszöbök EGY helyen élnek:
`vision/app/config.py` (`STABILIZER_PARAMS` + `AppConfig`), a mezőnkénti
indoklás a `StabilizerParams` definíciójában (`vision/pipeline/stabilizer.py`).

## 1. Mérőeszközök

| Eszköz | Mit mér | Mit hangol |
|---|---|---|
| `python -m tools.measure_pipeline_noise video <felvétel>` | statikus tábla: mezőnkénti villódzás, konfidencia-eloszlás, mozgás-alapvonal (négyzetenkénti mean\|diff\|) | `motion_threshold`, `min_changed_conf`, `frame_reject_mean_conf`, kell-e szavazópuffer |
| `python -m tools.measure_pipeline_noise frames` | a FEN-gyűjtő 184 mentett képe az éles kivágási úton: pontosság sessionönként, hibás vs helyes predikciók konfidenciája, a detektor orientáció-instabilitása | `min_changed_conf`, az orientáció-igazítás szükségessége, adatminőség (ismétlődő hibás mezők = rossz FEN) |
| `python -m tools.replay_frames sessions` | a mentett képek sorozatai játszmaként, szimulált kéz-fázissal, 30 fps, determinisztikus órával: latencia, hamis elfogadás, beragadás | `min_stable_s`, `min_static_s`, `prefix_ambiguity_extra_s` |
| `python -m tools.record_camera` + `python -m tools.replay_frames video` | VALÓDI játék felvétele és offline visszajátszása időbélyegekkel | ugyanazok valódi kézzel — ez a végső mérce |
| `vision/app/run_live.py` → `timing_output/<run>/moves.csv` | élesben: `latency_from_state_ms` (a végállapot első frame-jétől), `latency_from_static_ms` (az utolsó mozgástól), `latency_ms` (a kéz megjelenésétől, benne a fizikai lépés ideje) | elfogadási kritérium (~300 ms) |

## 2. Mért tények (MobileNetV3-Small @128 px, ONNX, 1080p, i7-1255U)

### 2.1 Statikus tábla (élő kamera, félhomály, 333 frame)

- Villódzó mező: **0 / 64**. 21 312 mezőosztályozás, **0** eltérés a módusz-rácstól.
  Két szomszédos frame között címkét váltó mezők száma: mindig 0.
- Helyes predikciók konfidenciája: p1 = 0,920, p5 = 0,948, p50 = 0,980, min = 0,595.
- Mozgás-alapvonal (négyzetenkénti mean|diff|, 0–255): p50 5,6, p99,9 6,9, **max 7,1**;
  képkockánkénti max a 64 mezőn: p95 6,95, max 7,11.

Következmény: a **többségi szavazás 5 frame-es pufferen tiszta redundancia** —
statikus jelenetben nincs mit kisimítania, csak +2–3 frame késleltetést adott.
Elhagyva. A **mozgás-küszöb 12** a zaj-maximum (7,1) felett, a részleges
újraklasszifikálás küszöbe (18) alatt.

### 2.2 A gyűjtő 184 képe (6 session, 3 fényviszony, 720p + 1080p)

FIGYELEM: a képek a tanítóhalmaz részei / val_new near-duplicate → felső becslés.

- **A detektor orientációja**: ugyanarra a jelenetre (a mentett JPEG az élő
  frame helyett) **18 / 184** képen 270°-kal elforgatott rácsot adott. Ez a
  10. szakasz "orientáció törékeny" megjegyzésének közvetlen mérése → az
  újradetektálás utáni orientáció-igazítás nem opció, hanem szükséglet.
- Igazítás után: 11 776 mező, **47 hiba (0,40 %)**, ebből 30 = egy
  session (`sotetben_1080`) 15 képén ugyanaz a mezőpár (r0c5/r0c6, 0,98
  konfidenciával) → **rossz FEN/felrakás a gyűjtéskor**, nem modellhiba; 11 =
  `r0c0 black→empty` a `..._es_kislampa_1080` sessionben (tartós, egy mező).
  Tényleges modellhiba: **≤ 0,15 %**, és **tartós** (ugyanaz a mező képről
  képre), nem villódzás. Black recall (igazítva): 0,990; a két 720p session 1,000.
- **A hibás predikciók konfidenciája ugyanolyan magas, mint a helyeseké**
  (hibás: p50 0,976; helyes: p50 0,977). Egy 0,9-es kapu a hibák 74 %-át
  átengedné és a helyes cellák 1,75 %-át elutasítaná → **a konfidencia-kapu nem
  hibaszűrő**. Ezért: nincs tábla-átlag kapu; a változott mezőkre 0,40-es padló
  (a helyes predikciók minimuma 0,47; 0,60-nal a visszajátszásban 2/142 helyes
  lépés beragadt egy 0,48-as célmezőn), és 0,60-as frame-elvetés a tábla átlagán.

### 2.3 Költségek képkockánként (p50)

| Komponens | régi (ResNet18 torch, 10 fps) | most |
|---|---|---|
| board_detect (1080p) | 1971 ms (max 10 s) | **165 ms** (vektorizált `findGoodPoints` + `nonmax_sup`, bitre azonos eredmény) |
| warp | 2,9 ms | 2,3 ms |
| square_diff | 3,4 ms | 1,1 ms (+1,1 a 100 ms-os referencia-frame-hez) |
| classifier_full (64) | 300–350 ms | **30 ms** (preprocess 5,7 + ORT 21) |
| classifier_partial (20 / 28) | 40–55 ms | 6,9 / 10,7 ms |
| resolve | 6,4 ms | **0,14 ms** (inkrementális foglaltság) |
| stabilizer | 0,17 ms | ~0,2 ms |
| **részleges út összesen** | ~60 ms | **~15 ms** (30 fps-en 2x tartalék) |
| **teljes út összesen** | ~310 ms | ~35 ms (90 frame-enként + ébresztéskor) |

## 3. Az öt zajszűrő réteg — döntés számokkal

| Réteg | Régen | Most | Miért |
|---|---|---|---|
| 1. frame throttle | minden 3. frame (10 fps) | **minden frame** (30 fps) | 15 ms < 33 ms; a stabilizer fali időben mér, a ráta csak a granularitást javítja (100 → 33 ms) |
| 2. többségi szavazás | 5 frame-es puffer, ≥ 0,65 | **elhagyva** | 0 villódzás 21 312 mezőn (2.1); +200–300 ms késleltetést adott |
| 3. perzisztencia | 4 frame (400 ms @10 fps; a puffer miatt effektíve ~600 ms) | **250 ms fali idő + ≥ 3 frame** | ez az egyetlen réteg, ami a "jelenet változatlan" információt hordozza; a hover-védelmet átveszi a mozgás-kapu |
| 4. konfidencia-kapuk | tábla-átlag ≥ 0,5, változott ≥ 0,5, hold < 0,35 | **változott mezők min ≥ 0,40; frame-elvetés < 0,60; NINCS tábla-átlag kapu** | 2.2: a kapu nem hibaszűrő; a tábla-átlag egy távoli bizonytalan mező miatt blokkolt biztos lépést |
| 5. resolver zajtűrés | noise ≤ 1, cost ≤ 0,9 | ugyanaz, **de**: ≥ 2 mezős eltérés kötelező, a fuzzy célmezőt foglaltnak kell látni | az 1 mezős fuzzy ág "eltűnt bábu → tippelt célmező" volt: hamis elfogadási út; félkész ütés (mindkét mező üres) sem fogadható el |
| ÚJ: mozgás-kapu | — | négyzetenkénti mean\|diff\| > 12 = mozgás; 200 ms nyugalom kell | valódi információ: "a kéz elment"; ez engedi a rövidebb ablakot a hamis elfogadás növelése nélkül |
| ÚJ: előtag-várakozás | — | +1,0 s, ha a lépés egy másik legális lépés előtagja (Rf1 ↔ O-O) | bástyával kezdett sánc |
| ÚJ: promóció-várakozás | mindig vezér | a típus-fej választ; amíg gyalogot lát a 8. soron, max 5 s várakozás | a bábucsere a lépés része |

### Milyen hibamódot engedünk meg cserébe (amit elhagytunk)

- **Szavazópuffer nélkül** egy egyetlen frame-es, **≥ 2 mezős** egyidejű
  téves osztályozás új jelöltet indít (az ablak újraindul → +250 ms
  késleltetés, NEM hamis elfogadás, mert a jelöltnek 250 ms-ig kell
  fennállnia). Mért gyakorisága statikus táblán: 0 / 333 frame.
- **Rövidebb ablak (600 → 250 ms)**: egy olyan félkész állapot, amely
  pontosan egy legális lépésnek látszik ÉS a tábla pixelszinten 200 ms-ig
  nyugalomban van (a kéz elhagyta a mezőket) ÉS 250 ms-ig változatlan —
  ilyen a bástyával kezdett sánc (külön kezelve) és a "lerakom, majd
  mégis máshová teszem" 250 ms után (ez a régi rendszerben 600 ms volt). A
  kézben tartott, mezők fölött lebegtetett bábu a mozgás-kapun akad fenn.
- **Tábla-átlag kapu nélkül**: egy fény-összeomlás, amely a tábla nagy részét
  bizonytalanná teszi, de a két változott mezőt nem, elfogadható — a
  frame-elvetés (átlag < 0,60) és a ≥ 2 / ≤ 6 mezős korlát ezt fogja meg.
- **1 mezős fuzzy nélkül**: ha a lépés egyik mezőjét a modell TARTÓSAN
  rosszul látja, a lépés nem kerül elfogadásra (régen tippelt); a tracker
  ébresztő teljes átosztályozást és gördülő frissítést futtat, a modell
  tartós hibaaránya ≤ 0,15 %.

## 4. Visszajátszás és élő ellenőrzés — a 10. szakasz mérése

`python -m tools.replay_frames sessions` (142 lépés, 6 session, 18 futam,
szintetikus 0,8 s-os kéz-fázis, 2 s tartás, 30 fps, ±2 pixelzaj):

| | |
|---|---|
| helyes elfogadás | 125 |
| **hamis elfogadás** | **0** |
| beragadás | 17 → ebből 15 a `sotetben_1080` rossz FEN-címkéinek műterméke (a fizikai tábla két mezőben eltér a FEN-től, a resolver a Game állásához képest sosem talál illeszkedést), 2 a 0,60-as konfidencia-padló (0,48-as HELYES célmező) → padló 0,40; ezzel a `jo_fenyviszony_1080` 38/38, 0 beragadás |
| latencia a végállapot első frame-jétől | átlag 409, **p50 367, p95 567**, max 1100 ms; frames-to-accept módusz 11 frame |
| a kéz eltűnésétől | ≈ 300 ms = min_static_s 200 + motion_ref_age_s 100 (a 100 ms-os referencia-frame még "látja" a kezet) + frame-granularitás |

Élő ellenőrzés (kamera, statikus középjáték-állás, félhomály, 30 s, az ÚJ
trackerrel): init elsőre (a detektor rácsa rot90-ben állt, az igazítás
megoldotta), 445/445 frame `no_change`, **0 elfogadás**, 0 újradetektálás,
képkockánként p50 19,7 ms / p95 27,6 ms (warp 6,7, square_diff 5,6,
classifier_partial 6,0, stabilizer 0,08), mozgás-jel p50 6,4 / max 7,4.
A feldolgozott 14,9 fps a kamera félhomályos hosszú exponálásából jön
(auto-exposure 312 → a kamera ~15 fps-t ad), nem a pipeline-ból.

Amit ez NEM mér: a valódi kéz (lebegtetés, félkész lépés). Ehhez:
`tools/record_camera.py` + `tools/replay_frames.py video`.

## 5. Újrahangolás — recept

1. Új modell/kamera után: `python -m tools.record_camera --out rec/static --seconds 30`
   (statikus tábla), majd `python -m tools.measure_pipeline_noise video rec/static/…`
   → ha a mozgás-alapvonal maximuma > 10, emeld `motion_threshold`-ot (a
   `partial_diff_threshold` alatt maradjon); ha megjelenik villódzás
   (P(≥2) > 0,01), emeld `min_stable_frames`-t 4-re.
2. `python -m tools.measure_pipeline_noise frames` → black recall, hibás
   predikciók konfidenciája; ha a hibás predikciók p50-je 0,8 alá megy, a
   `min_changed_conf` érdemben szűr (akkor 0,7–0,8).
3. Valódi játék: `record_camera` + `replay_frames video --moves …` → hamis
   elfogadás = 0 kötelező; a `latency_from_static_ms` p50 a cél (~300 ms).
   Ha hamis elfogadás jelenik meg lebegtetésnél, először `min_static_s`-t
   emeld (0,20 → 0,35), utána `min_stable_s`-t.
4. A stabilizer és a tracker döntési útjai tesztekkel fedettek:
   `tests/test_stabilizer.py`, `tests/test_tracker_state_machine.py`,
   `tests/test_resolver_incremental.py`, `tests/test_orientation.py`,
   `tests/test_board_detect_equivalence.py`.
