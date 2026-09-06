# Feladat: sakk-mezőosztályozó újratervezése + vision pipeline átszervezése

---

## ÁLLAPOT — 2026-09-06 este (olvasd el ELŐSZÖR, felülírja az alábbi szakaszok elavult számait)

**MIND A NÉGY FELADAT KÉSZ (7., 8., 9., 10.).** A 10. szakasz részletes mérési
dokumentuma: `docs/pipeline_tuning.md`. A küszöbök EGY helyen:
`vision/app/config.py` (`STABILIZER_PARAMS` + `AppConfig`).

### A 10. szakasz — mi készült el

- **Stabilizer újraírva** (`vision/pipeline/stabilizer.py`): fali-idő alapú
  perzisztencia (250 ms + ≥3 frame), **mozgás-kapu** (négyzetenkénti |diff| > 12
  = mozgás; 200 ms nyugalom kell), a **≥2 mezős invariáns** kikényszerítve
  (1 mezős eltérés SOHA nem kerül kiadásra), flicker-tolerancia változatlan.
  Elhagyva: frame-throttle, 5 frame-es többségi szavazás, tábla-átlag
  konfidencia-kapu, HOLD-időzítő, cooldown. A referencia az ELFOGADOTT állás
  (a resolver expected_occ-ja), így "no_change" = a kamera azt látja, amit a
  sakklogika hisz.
- **Tracker** (`vision/pipeline/tracker.py`): minden frame feldolgozása (30 fps),
  gördülő frissítés (+8 mező/frame → a 64 mező ~270 ms alatt frissül),
  eseményvezérelt teljes átosztályozás feloldhatatlan eltérésnél, injektálható
  óra (visszajátszáshoz), latencia-számvitel (`accept_info`), promóció a
  típus-fejjel (a bábucseréig várva), bástyával kezdett sánc hosszabb
  megerősítése, **orientáció-igazítás** init-kor és újradetektáláskor (a
  bbox-rács átrendezésével; `raw_to_standard` érintetlen).
- **Resolver** (`chess_logic/resolver.py`): inkrementális foglaltság (6,4 →
  0,14 ms), `min_changed_cells=2`, a fuzzy célmezőt foglaltnak kell látni,
  `prefix_ambiguities()` (Rf1 ↔ O-O), promóciós típus-tipp + `use_type_hint_for_moves`
  kapcsoló (alapból ki). A sakkszabály-logika (python-chess) érintetlen.
- **board_detect** 1971 → **165 ms** (1080p): `findGoodPoints` és `nonmax_sup`
  vektorizálva, bitre azonos eredmény (tests/test_board_detect_equivalence.py).
- **run_live**: `process_every_nth_captured_frame=1`; moves.csv új oszlopok:
  `latency_from_state_ms`, `latency_from_static_ms`; a robot lépése után a
  tracker a backend aktuális FEN-jéről indul újra (eddig az alapállásról →
  init-too-far).
- **Eszközök**: `tools/measure_pipeline_noise.py` (zaj/konfidencia/mozgás
  mérése), `tools/replay_frames.py` (offline visszajátszás valódi modellel,
  determinisztikus órával), `tools/record_camera.py` (valódi játék felvétele
  offline méréshez). Tesztek: 126 (stabilizer, tracker-állapotgép
  szintetikus képekkel, resolver-ekvivalencia, orientáció, detektor-ekvivalencia,
  run_live).

### Mért eredmények

| Mérés | Eredmény |
|---|---|
| Statikus tábla, élő kamera, 333 frame | 0/64 villódzó mező, 0 címkeváltás → a szavazópuffer redundáns volt |
| Mozgás-alapvonal statikus táblán (mean\|diff\| / mező) | felvételen max 7,1; élő pipeline-ban max 7,4 → küszöb 12 |
| 184 mentett kép (éles kivágási út) | igazítás után 47/11 776 hiba, ebből 41 rossz FEN/felrakás a gyűjtéskor; tényleges modellhiba ≤ 0,15 %, black recall 0,990 |
| Hibás predikciók konfidenciája | p50 0,976 = a helyeseké → a konfidencia-kapu nem hibaszűrő; padló 0,40 |
| Detektor orientáció | ugyanarra a jelenetre 18/184 újradetektálás rot270; élőben két detektálás 1 s-on belül rot0 vs rot90 → az igazítás mindkettőt megoldotta |
| Képkockánkénti költség (részleges út) | **~15 ms** benchmarkon, **19,7 ms p50** élőben (30 fps-hez elég; félhomályban a kamera maga esik ~15 fps-re a hosszú exponálás miatt) |
| Visszajátszás (142 lépés, 6 session, szintetikus kéz) | **0 hamis elfogadás**, 125 helyes; a 17 beragadásból 15 a `sotetben_1080` hibás FEN-címkéinek műterméke, 2 a 0,60-as padló volt (0,40-nel: 38/38 a jó fényű sessionben, 0 beragadás) |
| Latencia a végállapot első frame-jétől (szimulált 30 fps) | **p50 367 ms, p95 567 ms** (a kéz eltűnésétől ≈ 300 ms: 200 ms nyugalom + 100 ms mozgás-referencia + frame-granularitás) |
| Élő statikus futás 30 s | 445/445 frame `no_change`, 0 elfogadás, 0 újradetektálás, init elsőre (rot90 igazítással) |

### Ami NEM mérhető offline, és mit kell tenni

A **lebegtetett kéz** (a bábu a célmező felett, még nem letéve) elleni védelem
most a mozgás-kapu + 250 ms ablak (régen ~600 ms ablak, mozgás-kapu nélkül).
Ezt valódi játékon kell mérni: `python -m tools.record_camera --out rec/j1`,
majd `python -m tools.replay_frames video rec/j1 --moves "..."`. Ha hamis
elfogadás jelenik meg, először `min_static_s`-t emeld (0,20 → 0,35), aztán
`min_stable_s`-t — mindkettő `vision/app/config.py`-ban.

**Adatminőség**: a `sotetben_1080` session 15 fotóján a FEN két mezőben eltér a
fizikai táblától (r0c5 ↔ r0c6), a `maxra_allitott_lampaval_es_kislampa_1080`
sessionben r0c0 11 fotón, a `maxra_allitott_lampaval_720` egy fotóján két bábu
egy oszloppal arrébb — ezek a tanítóhalmazban rossz címkék; a
`measure_pipeline_noise frames` "ISMÉTLŐDŐ hibás mezők" listája mutatja őket.

---

Egy működő, éles kamerás sakk-követő rendszeren dolgozol, amely egy Franka Research 3
robotkart vezérel. A rendszer működik, de **túl lassan fogadja el a lépéseket**, és a
**fekete bábukat gyakran tévedésből "empty"-nek vagy "white"-nak** osztályozza.

A feladatod három részből áll:
1. Új tanítószkript írása (Google Colab) — te döntöd el a modellarchitektúrát, **mérés alapján**.
2. Új tanítóadat-gyűjtő szkript, ismert FEN-ből automatikusan címkézve.
3. A vision pipeline érdemi átszervezése, ONNX Runtime inferenciára.

---

## 1. Hardveres kényszer — ez a legfontosabb korlát

Az inferencia **ezen a gépen** fut, nem szerveren:

- CPU: Intel Alder Lake-UP3 (laptop/NUC osztály), **12 mag**, torch 10 szálat használ
- GPU: **Intel Iris Xe integrált grafika — NINCS NVIDIA GPU**
- `torch.cuda.is_available() == False` — ez hardveres tény, nem konfigurációs hiba
- Python 3.12, a repo gyökerében `.venv/` virtualenv

A tanítás Google Colabban történik (ott van GPU), az inferencia itt, CPU-n.
**Minden modellválasztási döntést a CPU-inferencia sebessége dominál.**

---

## 2. Mért teljesítményadatok (valós profiler-export a rendszerből)

| Komponens | mean | p50 | p95 | megjegyzés |
|---|---|---|---|---|
| `board_detect` | **1450 ms** | 1453 | 1550 | homográfia megoldás, csak init-kor + beragadáskor fut |
| `classifier_full` | **300 ms** | 310 | 345 | mind a 64 mező, ResNet18 @ 100px, CPU |
| `classifier_partial` | 40–55 ms | 55 | 61 | csak a változott mezők (~12–20 db) |
| `warp` | 2.9 ms | 2.7 | 3.9 | |
| `square_diff` | 3.4 ms | 3.2 | 5.4 | |
| `stabilizer` | 0.17 ms | 0.15 | 0.25 | elhanyagolható |
| `resolve` | 6.4 ms | 6.3 | 14.1 | legális lépések végigpróbálása |
| `frame_total` | 28 ms | **8 ms** | **66 ms** | |

**Következtetés:** a `classifier_full` 300 ms-a szinte teljesen a ResNet18 forward pass CPU-n.
A preprocessing (64 crop + resize + normalizálás) ebből csak ~20–30 ms.

---

## 3. Jelenlegi architektúra

```
vision/
  app/
    config.py              AppConfig dataclass + make_stabilizer()  (69 sor)
    run_live.py            kamera, worker thread, backend sync, timing export  (699 sor)
    debug_draw.py          OpenCV overlay
  board/
    find_chessboard.py     saddle-point alapú tábladetektálás  (291 sor)
    squares.py             mezők bbox-ainak kinyerése a warpolt képből
  pipeline/
    board_detector.py      detect_board_on_frame() -> DetectionResult(M, bbox_warp, centers)
    batch_classifier.py    crop_with_context(), preprocess_roi_for_batch(), batch inferencia
    tracker.py             ChessVisionTracker — a fő állapotgép  (511 sor)
    stabilizer.py          StateStabilizer — mikor elég stabil egy állás  (296 sor)
    profiler.py            PipelineProfiler — komponens-időmérés
  models/
    occupancy_color_model.py   OccupancyColorModel — checkpoint betöltés + predikció
chess_logic/
    resolver.py            resolve_move_from_occupancy() — foglaltság -> legális lépés
    game.py                Game — a sakkállapot egyetlen igaz forrása
    move_types.py          MoveGuess, MoveRecord, OccupancyResolveResult
tools/
    collect_training_data.py   régi gyűjtő (modell-predikció alapú címkézés)
    split_train_val.py         80/20 split
    train_colab.py             jelenlegi tanítószkript
    dump_live_rois.py          élő ROI-mentés összehasonlításhoz
```

### Adatfolyam

```
kamera frame (1280x720)
  -> detect_board_on_frame(gray)         # homográfia M + 8x8 bbox rács, CSAK init-kor
  -> warpPerspective                     # 1632x1632 warpolt tábla
  -> crop_with_context(bbox, 0.50)       # mezőnként ~1.5x méretű ROI
  -> preprocess_roi_for_batch            # resize 100x100, BGR->RGB, ImageNet norm
  -> batch forward (64 db)               # -> labels[8][8], confs[8][8]
  -> raw_to_standard                     # rot90 + fliplr; kamera: bal-felső = A1
  -> StateStabilizer.update              # mikor tekintjük stabilnak
  -> Game.resolve_from_occupancy         # foglaltság -> legális lépés
  -> Game.apply_uci                      # állapot frissítés
  -> backend POST /api/move              # majd a robot végrehajtja
```

---

## 4. Már meghozott döntések — ezeket NE kérdőjelezd meg

| Döntés | Érték |
|---|---|
| Inferencia runtime | **ONNX Runtime** (tanítás marad PyTorch/Colab, export ONNX-be) |
| Osztályszerkezet | **Hibrid multi-task**: elsődleges 3-osztályos fej (empty/white/black) + másodlagos bábutípus-fej |
| A típus-fej bekötése | **Egyelőre CSAK promóciónál** használjuk. A 3-osztályos útvonal érintetlen marad. |
| Címkézés | **Automatikus, ismert FEN-ből** |
| Refaktor mélysége | **Jelentős átszervezés** — a stabilizer és a klasszifikációs útvonal újraírható, a pipeline általános alakja marad |
| Tanítóadat | **Mindkét forrás**: meglévő GitHub dataset + új gyűjtés |
| Latencia cél | **~300 ms** lépés-elfogadás (jelenleg ~700 ms) |

### A hibrid fej indoklása és jövőbeli bővítése

A második fej az inferenciát nem lassítja: a számítás ~99%-a a backbone-ban történik,
a fejek csak a lepoolozott feature-vektoron dolgoznak (512→3 vs 512→13 lineáris réteg).

**Most:** a típus-fej kimenete csak promóciónál használatos. A `resolver.py`-ban a
`_promotion_priority()` jelenleg mindig vezért tippel, mert a puszta foglaltságból nem
lehet eldönteni a promotált bábu típusát. Itt kell bekötni a típus-fejet.

**Készítsd elő** (de NE kösd be) azt, hogy később a típus-információ az egész
lépésdetektálásban aktívan használható legyen. Az architektúra tegye ezt egy kapcsolóval
elérhetővé, ne kelljen újraírni hozzá a pipeline-t.

---

## 5. A megoldandó problémák

### 5.1 A fekete bábuk osztályozása megbízhatatlan
Sötét mezőn álló fekete bábut gyakran "empty"-nek, néha "white"-nak osztályoz.
Ez a gyökérok, ami minden más tünetet felerősít.

Ellenőrzött tény: a tanítóképek kivágási geometriája **megegyezik** az éles pipeline-éval
(ugyanaz a `crop_with_context` + `context=0.50`), tehát ez **nem** train/inference
domain mismatch. A probléma az adat lefedettségében és/vagy a modellkapacitásban van.

### 5.2 Túl lassú lépés-elfogadás
Öt egymásra épülő zajszűrő réteg van ugyanarra a problémára:
1. frame throttle (`process_every_nth_captured_frame`)
2. mezőnkénti többségi szavazás egy 5 frame-es bufferen
3. perzisztencia-számláló (N egymást követő azonos jelölt)
4. konfidencia-kapuk halmaza (`min_votes_ratio`, `min_mean_conf`, `min_changed_conf`, …)
5. a resolver saját zajtűrése (`max_noise_cells`, `max_weighted_cost`)

Bármelyik réteg önmagában is új várakozási kört indíthat. **Értékeld ki, melyik réteg
hordoz valódi információt, és melyik puszta redundancia.**

### 5.3 A homográfia befagy
`detect_board_on_frame` csak init-kor fut. Ha a tábla/kamera elmozdul, minden későbbi
klasszifikáció romlik. (Részleges javítás már bekerült: beragadás esetén újradetektál.)

---

## 6. Már elvégzett javítások — ezekre építs, ne írd vissza őket

Ezek benne vannak a kódban, a viselkedésük teszteltés bizonyított:

**A `StateStabilizer` flicker-toleranciája.** Korábban bármilyen eltérés nullázta a
perzisztencia-számlálót, így egyetlen villódzó mező a végtelenségig újraindította a
számlálást (`candidate_not_persistent` beragadás). A javítás egy sakk-invariánsra épül:

> **Minden legális sakklépés legalább 2 mezőt változtat** (honnan + hova; en passant 3,
> sánc 4). Ezért az 1 mezős eltérés **sosem** lehet lépés — az mindig klasszifikátor-zaj.

A `_classify_candidate_change()` ezért három ágra bont: `SAME` / `FLICKER` (≤1 mező:
megtartja a jelöltet ÉS a számlálót) / `CHANGED` (≥2 mező: valódi állásváltás).
3 egymás utáni flicker után mégis adoptálja az új rácsot, hogy ne ragadhasson be.

**Ez az invariáns a rendszer egyik legerősebb eszköze — használd ki a refaktorban is.**

Egyéb már beállított értékek: `process_every_nth_captured_frame=3`,
`partial_max_squares=20`, `full_reclassify_interval=45`, valamint beragadás esetén
board-újradetektálás (`redetect_after_stuck_s=4.0`, `redetect_min_interval_s=10.0`).

---

## 7. FELADAT 1 — Tanítószkript (Google Colab)

Írj új tanítószkriptet a `tools/` mappába.

### 7.1 A modellarchitektúrát MÉRÉS alapján válaszd ki

**Ne találgass és ne hivatkozz általános népszerűségre.** A szkript tartalmazzon egy
benchmark szakaszt, amely a jelöltmodelleket **CPU-n, ONNX Runtime alatt, batch=64,
100–128px bemenettel** megméri, és a döntést a mért latencia + pontosság alapján hozza meg.

Mérlegelendő jelöltek (a listát bővítheted):
- MobileNetV3-Small — CPU-n nagyon gyors
- EfficientNet-B0 — kevés FLOP, de a depthwise konvolúciók CPU-n kevésbé hatékonyak
- ShuffleNetV2 x1.0
- ResNet18 — a jelenlegi, referencia-alapvonalnak
- Saját, kicsi CNN (4–6 konv réteg) — egy 3-osztályos, 100px-es feladathoz bőven elég lehet,
  és ez lehet a leggyorsabb

**Számszerű cél:** a `classifier_full` (64 mező) menjen **300 ms-ról 60 ms alá** ONNX
Runtime alatt, a jelenleginél jobb `black` recall mellett.

Írd le a döntés indoklását a szkript fejlécében, a mért számokkal együtt.

### 7.2 Multi-task fej

- Közös backbone
- 1. fej: 3 osztály (empty / white / black) — **ez az elsődleges, ez nem romolhat**
- 2. fej: bábutípus (6 típus + "nincs bábu", vagy 13 osztály — te döntsd el, indokold)
- A loss legyen súlyozott kombináció; a típus-fej **ne ronthassa el** a 3-osztályos fejet
- Ha negatív transzfert mérsz, a típus-fej gradiense legyen skálázható/leválasztható

### 7.3 A fekete bábu problémára célzott technikák

Kötelezően értékeld ki és ahol indokolt, alkalmazd:
- Erős megvilágítás-augmentáció (brightness/contrast, random gamma)
- `RandomAutocontrast`, `RandomEqualize` — a sötét mezőn álló fekete bábu kontrasztjához
- Focal loss (nehéz példákra fókuszál) és/vagy osztálysúlyozás
- `WeightedRandomSampler` az osztály-egyensúlytalanságra
- Árnyék-szimuláció (a robotkar árnyékot vet a táblára)
- `RandomErasing` — részleges takarás

### 7.4 Modellkiválasztás és mentés

- A legjobb checkpoint kiválasztása **osztályonkénti pontosságok átlaga (macro)** alapján,
  **ne** a nyers accuracy alapján — azt a domináns osztályok elnyomják
- Early stopping
- Konfúziós mátrix + osztályonkénti report mentése
- **Külön riportáld a `black` osztály recall-ját** — ez a projekt kulcsmetrikája

### 7.5 Checkpoint-formátum — kötelező szerződés

Az `OccupancyColorModel` ezeket a kulcsokat olvassa. **Ha ezt elrontod, a rendszer némán
rossz eredményt ad:**

```python
{
    "variant":     "<architektúra neve>",     # ez választja ki a betöltendő architektúrát
    "model_state": state_dict,
    "class_names": [...],   # LÁSD A FIGYELMEZTETÉST LENT
    "img_size":    int,
    "normalize":   {"mean": [...], "std": [...]},
}
```

> ### ⚠️ KRITIKUS CSAPDA — osztálysorrend
>
> A `torchvision.datasets.ImageFolder` **ábécésorrendben** adja az osztályokat:
> `['black', 'empty', 'white']` → indexek 0, 1, 2.
>
> A pipeline belső foglaltság-kódolása viszont **más**: `empty=0, white=1, black=2`.
>
> A fordítást az `OccupancyColorModel.idx_to_label` végzi, amely a checkpoint
> `class_names` mezőjéből épül. **A `class_names`-t pontosan abban a sorrendben kell
> menteni, ahogy a modell kimeneti indexei jelentik** (tehát ImageFolder esetén
> ábécésorrendben). Ha ezt felcseréled, a fehér és a fekete bábuk **némán felcserélődnek**,
> és a rendszer működni látszik, csak rossz lépéseket ad ki. Írj tesztet erre.

---

## 8. FELADAT 2 — Tanítóadat-gyűjtő, ismert FEN-ből

Írj új gyűjtő szkriptet a `tools/` mappába.

**Elv:** a felhasználó felállítja a táblát egy ismert állásra és megadja a FEN-t.
A szkript ebből **automatikusan, hibátlanul** címkézi mind a 64 mezőt — foglaltsággal
**és** bábutípussal együtt. Nincs kézi címkézés és nincs címkézési zaj.

Követelmények:
- Használja **pontosan ugyanazt** a kivágási útvonalat, mint az éles pipeline
  (`detect_board_on_frame` → `warpPerspective` → `crop_with_context(context=0.50)`).
  Minta erre: `tools/dump_live_rois.py`.
- A FEN-ből a `chess_logic.resolver.board_to_occupancy()` és a `python-chess` adja a címkéket.
- Mentse el a bábutípust is (a hibrid fejhez).
- Legyen könnyű sorozatban gyűjteni: több állás, több fényviszony, több napszak.
- Emlékeztesse a felhasználót a **fix kamera-exponálásra és fehéregyensúlyra** (auto-exposure
  esetén a fekete bábu látszó fényereje frame-ről frame-re változik).
- Legyen kompatibilis a meglévő `train_new/val_new/{black,empty,white}` struktúrával,
  hogy a régi és az új adat együtt használható legyen.

Meglévő dataset: `https://github.com/borsz05/sakk_modelltanitas`
(`train_new/` és `val_new/`, mindkettőben `black/`, `empty/`, `white/` almappák).

---

## 9. FELADAT 3 — ONNX Runtime inferencia

- Export szkript: PyTorch checkpoint → ONNX (dinamikus batch méret, mert a partial út
  változó számú mezőt osztályoz)
- Értékeld ki az **INT8 kvantálást**; ha a `black` recall nem romlik érdemben, használd
- Az `OccupancyColorModel` (vagy utódja) támogassa **mindkét backendet**: PyTorch és ONNX
  Runtime. Az ONNX legyen az alapértelmezett, a PyTorch maradjon fallbacknek.
- A `.pt` és a `.onnx` fájl adjon **numerikusan egyező** predikciókat — erre írj tesztet
- A preprocessing (64 crop + resize + normalizálás) jelenleg 64 iterációs Python ciklus;
  vektorizáld

Az ONNX Runtime legyen felvéve a `requirements.txt`-be.

---

## 10. FELADAT 4 — A vision pipeline átszervezése

Cél: **~300 ms lépés-elfogadási latencia**, beragadás nélkül.

### Amit vizsgálj felül

1. **A 4. szakaszban felsorolt öt zajszűrő réteg.** Melyik hordoz valódi információt?
   A többségi szavazás és a perzisztencia-számláló ugyanazt a zajt szűri kétszer.
   Ha egy réteg elhagyható a hamis elfogadások növelése nélkül, hagyd el — de **indokold
   számokkal**, ne érzésre.

2. **A `resolve_move_from_occupancy`** minden legális lépésre `board.copy()` + `push()` +
   `board_to_occupancy()` hívást végez (~6 ms). Nyitóállásban 20 legális lépés, középjátékban
   30–40. Ez gyorsítható inkrementális foglaltság-számítással.

3. **A partial reclassify útvonal.** Csak a képileg változott mezőket osztályozza újra.
   Ha egy mező címkéje rossz, de a pixelei nem változtak, a hibás címke megmarad a következő
   teljes újraklasszifikálásig. Van jobb ébresztési feltétel?

4. **A `board_detect` 1450 ms.** Beragadáskor újrafut, és ilyenkor a rendszer ennyi ideig
   nem dolgoz fel frame-et. Gyorsítható? Futhat háttérszálon?

5. **Konfidencia-kapuk.** A `min_mean_conf` a **teljes tábla** átlagkonfidenciáján dönt,
   miközben egy lépés 2–4 mezőt érint. Egyetlen bizonytalan mező a tábla másik végén
   blokkolhat egy egyébként biztos lépést. Ez helyes viselkedés?

### Sérthetetlen invariánsok

- **A hamis elfogadás valószínűsége nem nőhet.** Egy téves lépésre a robotkar fizikailag
  mozdul — ez a legsúlyosabb hibamód, súlyosabb, mint a lassúság.
- Minden legális lépés ≥2 mezőt változtat — az 1 mezős eltérés zaj.
- A sakklogika a `chess_logic/` csomagban marad, nem szivárog a vision trackerbe.
- A tábla orientációja: a kamera képén bal-felső = A1 (`raw_to_standard`: rot90 + fliplr).
- Foglaltság-kódolás: `empty=0, white=1, black=2`.
- A backend HTTP szerződése (`POST /api/move`, `GET /api/robot/busy`) nem változhat.
- A `--no-robot` kapcsoló és a kalibrációs kapu működése maradjon.

---

## 11. Elfogadási kritériumok

A munka akkor kész, ha:

1. `classifier_full` (64 mező) **< 60 ms** ONNX Runtime alatt, CPU-n
2. A lépés-elfogadási latencia **~300 ms** (mérd a `PipelineProfiler`-rel és a `moves.csv`-vel)
3. A `black` osztály recall-ja **mérhetően jobb** a jelenleginél — számot közölj
4. **Nincs `candidate_not_persistent` beragadás** stabil táblán
5. A hamis elfogadások száma nem nőtt
6. A `.pt` és `.onnx` predikciók numerikusan egyeznek — teszttel bizonyítva
7. Az osztálysorrend-fordítás (`class_names` → `idx_to_label`) teszttel fedett
8. A promóciós típus-felismerés bekötve, a többi útvonal érintetlen

---

## 12. Munkamódszer

- **Mérj, ne tippelj.** Minden teljesítményállítást a `PipelineProfiler` adataival támassz alá.
  A repo `timing_output/` mappájában vannak korábbi mérések referenciának.
- A rendszer **most is működik**. Ne törd el menet közben — a refaktor legyen lépésenként
  tesztelhető, és minden lépés után maradjon futtatható állapotban.
- Ahol egy védőréteget elhagysz, **írd le, milyen hibamódot engedsz meg cserébe**.
- A kód kommentstílusa kevert magyar/angol — illeszkedj a szerkesztett fájl stílusához.
- Ahol a döntésed nem egyértelmű, írd le az alternatívát és azt, miért ezt választottad.

## 13. Amit NE csinálj

- Ne írd át a `chess_logic/` sakkszabály-logikáját — az helyes és tesztelt
- Ne vezesd be a 13 osztályos típusfelismerést a fő lépésdetektálási útvonalba (csak promóció)
- Ne írd vissza a stabilizer flicker-toleranciáját reset-alapú logikára
- Ne feltételezz CUDA-t sehol az inferencia útvonalon
- Ne változtass a backend API-n vagy a robot-vezérlés szerződésén
- Ne emeld a bemeneti felbontást pusztán a pontosság reményében — mérd meg a latencia-árát
