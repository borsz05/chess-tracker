# Kameraalapú sakkparti-követő rendszer robotkaros végrehajtással — tudományos leírás

## Absztrakt

Jelen munka egy olyan integrált rendszer tervezését és megvalósítását ismerteti, amely valós időben követi a fizikai sakktáblán zajló játékot egy felülnézeti kamera segítségével, és a sakkmotoron alapuló gépi válaszlépéseket egy Franka Research 3 robotkarral hajtja végre fizikailag. A rendszer nem kísérli meg a bábuk típusát optikailag azonosítani; ehelyett egy háromoszályos (üres/fehér/fekete) ResNet18-alapú foglaltsági osztályozót kombinál a sakkszabályok mint determinisztikus szűrő alkalmazásával. A tábladetektáláshoz Hessian-determináns-alapú nyeregpont-detektálást és RANSAC homográfiát, a lépésfelismeréshez occupancy-különbség alapú exact és fuzzy illesztést, az időbeli stabilizáláshoz szavazásos puffert alkalmaz a rendszer. A robotvezérlés MoveIt2-alapú Cartesian path tervezéssel és erő alapú fogással valósul meg.

---

## 1. Bevezetés

A fizikai sakktábla automatikus digitalizálása látszólag egyszerű feladat, valójában azonban számos nehézséget rejt.

**Optikai kihívások.** A bábuk sziluettje felülnézeti kameraképen erősen hasonlít egymáshoz — különösen azonos színű készletek esetén a huszár és a futó, vagy a vezér és a bástya könnyen összetéveszthető. Emellett a perspektivikus torzítás, az egyenetlen megvilágítás és az árnyékok tovább nehezítik a megbízható osztályozást.

**Takarás és perturbáció.** Lépés közben a játékos keze takarja a tábla egy részét; a bábu ideiglenesen "lebeg" — sem forrás-, sem célmezőn nincs. Ez rövid, de határozott perturbációt okoz a vizuális kimenetben.

**Zajos osztályozói kimenet.** Még egy jól betanított osztályozó is képkockánként változó eredményeket produkálhat enyhe megvilágításváltozás, kameraszenzor-zaj vagy részleges takarás miatt.

**Azonosítás típus nélkül.** Ha a rendszer nem tudja, hogy egy adott mezőn huszár vagy futó áll, a lépésfelismerést az occupancy-változások és a sakkszabályok kombinációjával kell elvégezni — ezek együttesen általában egyértelműen azonosítják a lépést.

**Robotvezérlési pontosság.** A robotnak milliméteres pontossággal kell a bábuk körülbelüli helyzetét megtalálnia; a fogás erőalapú visszacsatolással kell hogy biztonsági leállást kerüljön el.

A rendszer a fenti kihívásokra moduláris megközelítéssel válaszol: a látás, a játéklogika, a backend és a robotvezérlés rétegei jól definiált interfészeken keresztül kommunikálnak, és minden réteg függetlenül fejleszthető és tesztelhető.

---

## 2. Tábladetektálás módszertana

### 2.1 Hessian-determináns alapú nyeregpont-detektálás

A sakktábla fekete-fehér mintázata jellegzetes nyeregpontokat (saddle point) hoz létre a négyzethatárokon: az intenzitás egy irányban növekszik, a merőlegesben csökken. Ezek a pontok a Hessian-mátrix determinánsa (gxx·gyy − gxy²) alapján detektálhatók, amelynek értéke ezeken a helyeken jellemzően negatív és nagy abszolútértékű.

A `getSaddle(gray_img)` függvény Sobel-szűrőkkel számítja a másodrendű deriváltakat, majd pixel-szinten értékeli a determinánst. A lokális maximumok kinyeréséhez 10×10 ablakos non-maximal suppression (`nonmax_sup`) és adaptív küszöbözés (`pruneSaddle`, kezdeti küszöb: 128, duplázva amíg kezelhető szám marad) kerül alkalmazásra.

**Miért ez a módszer?** A Hessian-alapú nyeregpont-detektálás robusztusabb a Harris-saroksdetektornál a sakktábla belső rácspontjain, ahol a sarokmintázat nem szükségképpen éles — a nyeregpont-jellemző megbízhatóbban reagál az intenzitás-átmenetekre változó megvilágítás és kameratávolság mellett is.

### 2.2 Homográfia és perspektíva-korrekció

A detektált belső rácspontokból (7×7 belső metszéspont) a `generateNewBestFit` RANSAC-alapú homográfiát számít a kép síkjából egy kanonikus top-down nézetbe. A transzformáció 1632×1632 pixeles kimenetet produkál (17 × 96 pixel/mező). A mező bounding boxok belső padding-gel (`inner_pad_ratio=0.06`) szűkíthetők, csökkentve a szomszédos mezők közötti "szivárgást".

### 2.3 Lucas-Kanade optikai áramlás alapú követés

A `ChessVisionTracker` — miután egyszer megtalálta a táblát — opcionalisan Lucas-Kanade optikai áramlással követi a nyeregpontokat a következő képkockákon, elkerülve a teljes újradetektálást minden frame-en. Ez csökkenti a számítási terhelést és stabilizálja a belső rácspontok pozícióját enyhe kameramozgás esetén. Periodikusan (és mindig sikertelen követés esetén) a rendszer teljes újradetektálást végez.

---

## 3. Bábuosztályozás

### 3.1 Miért háromoszályos és nem 12 osztályos megközelítés?

Naív megközelítésként kézenfekvő lenne mind a 12 bábu-típust (6 fehér + 6 fekete) közvetlenül osztályozni. Ezt a megközelítést a rendszer tudatosan elveti a következő okokból:

- **Adatmennyiség:** 12 osztályos osztályozóhoz lényegesen több, és változatosabb tanítóadat szükséges. A 3 osztályos megközelítés kisebb adatmennyiséggel is jó generalizációt ér el.
- **Robusztusság:** Felülnézeti képen a bábu típusát sokszor a szín is elfedi — a fehér vezér és a fehér futó csúcskörvonala hasonlít. A foglaltsági osztályozás (üres/fehér/fekete) sokkal megbízhatóbb jelzéseket kap.
- **Redundancia elkerülése:** A bábu típusát a játéklogika pontosan nyilvántartja — az osztályozónak nem kell megismételnie azt az információt, ami már eleve rendelkezésre áll.

### 3.2 ResNet18 transfer learning

Az `OccupancyColorModel` egy előtanított ResNet18 architektúrára épül, amelynek utolsó teljesen kapcsolt rétege 3 kimeneti neuronra van cserélve. A bemeneti képméretek: 100×100 pixel; normalizálás: mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25].

**Miért ResNet18?** A ResNet18 elegendő kapacitással rendelkezik a foglaltság/szín megkülönböztetéséhez, miközben inferenciája gyors (CPU-n is alkalmazható batch üzemmódban). Mélyebb architektúrák (ResNet50, EfficientNet) nem hoznak érdemi javulást ennél a háromoszályos feladatnál, de lassítanák a valós idejű feldolgozást.

A klasszifikáció minden mezőre 50%-os kontextus-paddingel történik (`context=0.50`): a kivágott kép tartalmazza a szomszédos mezők széleit is, ami segít a szín összehasonlításában és az árnyékhatások kezelésében.

### 3.3 Partial reclassify optimalizáció

Minden képkockán teljes batch osztályozás futtatása pazarló, mivel egy lépés között a 64 mező többsége nem változik. A `compute_square_diffs()` függvény pixelszintű különbséget számít az előző és aktuális képkocka warped verziója között; csak azok a mezők kerülnek újraosztályozásra, ahol a különbség meghaladja a `partial_diff_threshold=18.0` küszöböt, maximum `partial_max_squares=12` mező/képkocka. Teljes reklasszifikáció minden 30. képkockán, illetve tábla elvesztése/visszatalálás esetén hajtódik végre.

---

## 4. Lépésfelismerés

### 4.1 Occupancy-különbség alapú resolver

A `resolve_move_from_occupancy(current_board, observed_occ, observed_conf)` függvény (a `chess_logic/resolver.py`-ban) az alábbi lépéseken megy végig:

**Exact matching:** A python-chess könyvtár `board.legal_moves` generátorán iterálva minden legális lépésre kiszámítja az elvárt occupancy térképet (`board_to_occupancy`), és összehasonlítja a megfigyelttel. Ha nulla eltérés (`occupancy_distance = 0`) → a lépés egyértelműen azonosítva.

**Fuzzy matching:** Ha az exact illesztés sikertelen (pl. az osztályozó 1-2 mezőn tévedett), a resolver megvizsgálja az összes olyan legális lépést, amelyre az occupancy-különbség (Hamming-távolság) legfeljebb `max_noise_cells=1`. A jelöltek közül a legkisebb "súlyozott különbség" (`weighted_diff`) kerül kiválasztásra: ez az eltérő cellákon lévő osztályozói konfidenciák összege — alacsony konfidenciájú eltérések kisebb büntetést kapnak. A küszöb: `max_weighted_cost=0.9`.

**Sakklogika mint szűrő.** A resolver kizárólag legális lépéseket vizsgál: az illegális (pl. saját királyt sakkba helyező) lépések automatikusan kiesnek. Ez drasztikusan szűkíti a keresési teret és megakadályozza a téves felismeréseket.

**Lépésrendezés:** A resolver a gyalogpromotálásokat (mindig vezérré konvertálva) előnyben részesíti a normál lépésekkel szemben, biztosítva a helyes feldolgozást a ritka esetekben.

### 4.2 Koordináta-leképezés

A vision réteg sor-oszlop koordinátái (`raw_to_standard`: 90° CW forgatás + vízszintes tükrözés) a `coords_to_chess_square(row, col)` segítségével konvertálódnak python-chess szimbólumokká (pl. "e2"), és vissza (`chess_square_to_coords`).

---

## 5. Időbeli stabilizálás

### 5.1 A nyers kimenet zajossága

Valós idejű kamerafeldolgozásnál az osztályozó kimenete képkockánként fluktuálhat: egy határ-közelben lévő mező hol "fehér", hol "üres" besorolást kaphat. Különösen perturbált helyzetben (lépés közben a kéz takarja a táblát) a nyers occupancy térképek kaotikusak.

### 5.2 StateStabilizer: szavazásos puffer és hisztézis

A `StateStabilizer` egy körkörös puffert tart fenn az utolsó `buffer_size=5` képkocka occupancy térképéből. Minden frissítésnél cell-wise majority vote-ot végez (küszöb: `min_votes_ratio=0.65`): csak akkor fogad el egy állást, ha az utolsó 5 képkockából legalább 3,25-ben ugyanazt az értéket kapja az adott cellára.

**Emissziós feltételek:**
1. Az átlagos osztályozói konfidencia ≥ 0.50 a teljes táblán (`min_mean_conf`)
2. A megváltozott mezők konfidenciája ≥ 0.50 (`min_changed_conf`)
3. Legfeljebb 6 mező változhat egy érvényes lépésnél (`max_changed_for_move`)
4. Két emisszió között legalább 0.20 másodperc (`emit_cooldown_s`)
5. A kandidátus állapotnak legalább `stable_frames=4` egymást követő friss képkockán kell megjelennie

**HOLD mód:** Ha egyszerre több mint 10 mező változik (`hold_changed_threshold`), vagy az átlagos konfidencia 0.35 alá esik (`hold_low_conf_threshold`), a rendszer HOLD módba vált: minimum 0.50 másodpercig (`hold_min_duration_s`) nem ad ki emisszióst, és legalább 2 megerősítő frame kell a kilépéshez (`recovery_stable_frames`).

**Miért ezek a paraméterek?** A `buffer_size=5` és `min_votes_ratio=0.65` elegendő robusztusságot biztosít a pillanatnyi osztályozói zajjal szemben, miközben a késés 5 × (1/5 FPS-nek megfelelő feldolgozási ráta esetén) ≈ 1 másodpercen belül marad. Az `emit_cooldown_s=0.20` megakadályozza a dupla-emissziókat gyors egymás utáni hasonló állapotoknál.

---

## 6. Robotvezérlés

### 6.1 Kalibrációs matematika

A tábla és a robot koordináta-rendszerének összekapcsolásához az A1 és H8 mezők közepe kerül mérésre robot-koordinátákban (mm). A `Calibration` osztály ebből vektoros interpolációval számítja a 64 mező pozícióját:

Legyen d = H8 − A1 a tábla átlós vektora. Ekkor:
- Oszloplépés: file_x = (dx + dy) / 14, file_y = (dy − dx) / 14
- Sorlépés (90° CCW forgatás): rank_step = (−file_y, file_x)

Az `square_to_xy("e4")` metódus az oszlop- és sorlépések lineáris kombinációjaként adja vissza a keresett koordinátát. Ez a megközelítés egyetlen kalibrációs mérési lépést igényel (két sarok), és a rácsszerkezet regularitásán alapul — nem igényel mind a 64 mező egyedi kalibrálását.

### 6.2 Cartesian path tervezés MoveIt2-vel

A `chess_executor.py` ROS2 node MoveIt2-t használ mozgástervezéshez:
- Kartéziusi (Cartesian) úttervezés: az end-effector pályája a feladatérben terveződik, joint-térbe konvertálva
- Csoportnév: `fr3_arm`, 7 szabadsági fok (fr3_joint1..7)
- Orientáció: állandóan lefelé mutató fogó (`_DOWN_QUAT = [1.0, 0.0, 0.0, 0.0]`)
- Sebesség: 0.3 m/s, gyorsulás: 0.3 m/s²

A vertikális mozgások (`_vertical_path`) fix XY koordinátán fel-le pályát terveznek; a mezők közötti átvitel (`_arc_path`) parabolikus Z-profillal valósul meg:

```
z(t) = Z_LIFT + 4·(Z_TRAVEL − Z_LIFT)·t·(1−t),  t ∈ [0, 1]
```

ahol Z_LIFT = 0.12 m és Z_TRAVEL = 0.20 m. A 8 waypoint egyenletes t-paraméterezéssel kerül kiszámításra, biztosítva a sima tangenciális átmenetet az ív végpontjain.

**Miért parabolikus ív és nem egyenes pálya?** Az egyenes pálya a mezők közötti áthaladáskor „lesöpörné" az útba eső bábukat. A parabolikus ív garantálja, hogy az end-effector legalább Z_LIFT = 12 cm magasságban haladjon az ív végpontjain, miközben az ív tetőpontján 20 cm-ig emelkedik. Ez praktikusan minden esetben elkerüli az ütközést a szomszédos bábukon.

### 6.3 Erő alapú fogás

A Franka Research 3 gripper action interfészen keresztül vezérelhető (`/fr3_gripper/grasp`). A fogásnál alkalmazott erő: 20.0 N, sebesség: 0.05 m/s, pozíció-tűrés: ±0.015 m. Az erőalapú fogás két előnnyel jár:
- Nem igényli a bábu pontos méretének ismeretét: különböző méretű bábukhoz automatikusan alkalmazkodik.
- Ha a fogó akadályba ütközik (pl. bábu váratlan pozíciójú), a gripper vezérlő biztonsági leállást generál, megakadályozva a mechanikai sérülést.

---

## 7. Korlátok és jövőbeli fejlesztési irányok

### 7.1 Jelenlegi korlátok

**Típus-felismerés hiánya.** A rendszer kizárólag foglaltságot és színt érzékel; a bábu típusa a játéktörténetből rekonstruálódik. Ha a játékot félbeszakítják és nem a kezdő állásból indítják, a játéklogika megzavarodik. Tetszőleges kezdőállás (`start_fen` paraméterrel) beállítható, de manuálisan.

**Megvilágítás érzékenység.** Erős oldalfény vagy gyors megvilágításváltás a stabilizátor HOLD modjába kergetheti a rendszert. A kameraalapú tábladetektálás robusztussága csökken alacsony kontrasztú körülmények között.

**Egy kameraálláspont.** A jelenlegi implementáció egyetlen felülnézeti kamerát használ; erős takarás (pl. nagy bábu mögötti kis bábu) esetén az osztályozó tévedhet.

**Kalibrációs mérés pontossága.** A robot pozíciópontossága az A1 és H8 mérési pontosságától függ; szisztematikus mérési hiba az egész táblán eltolódást okoz.

### 7.2 Jövőbeli fejlesztési irányok

- **12 osztályos osztályozó:** Elegendő tanítóadat esetén lehetséges a bábu típusának közvetlen optikai felismerése; ez megnyitná az utat tetszőleges állásból való indulás, ill. állásbeállítás felé.
- **Többkamerás rendszer:** Két egymásra merőleges kamera csökkentené a takarási problémákat és javítaná a tábladetektálás robusztusságát.
- **Homográfia-finomhangolás optikai áramlással:** Folyamatos pontosítás az optikai áramlás alapú nyeregpont-követéssel minden képkockán, nem csak periodikusan.
- **Adaptív kalibrációs korrekció:** A robot tényleges érintkezési pozícióinak visszacsatolásával a kalibrációs rácsot menet közben lehetne finomítani.

---

## 8. Összefoglalás

A bemutatott rendszer igazolja, hogy a fizikai sakktábla valós idejű digitalizálása és robotkaros végrehajtása megvalósítható egy háromrétegű megközelítéssel: Hessian-alapú tábladetektálás és perspektíva-korrekció, háromoszályos ResNet18 foglaltsági osztályozó részleges újraosztályozási optimalizációval, valamint szavazásos temporal stabilizálás. A lépésfelismerés az occupancy-különbség és a sakkszabályok kombinációján alapul, elkerülve a bábu-típus optikai azonosításának nehézségeit. A robotvezérlés MoveIt2-alapú Cartesian path tervezéssel és erő alapú fogással pontos és biztonságos bábumozgatást tesz lehetővé. A moduláris architektúra lehetővé teszi az egyes komponensek független fejlesztését és tesztelését.
