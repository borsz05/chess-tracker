# Indítási útmutató

Minden terminálban az első lépés mindig ugyanaz — belép a mappába és aktiválja a venv-et:

```bash
cd /home/berci/Asztal/chess-tracker
source .venv/bin/activate
```

Ha aktiválva van, a prompt elején megjelenik a `(venv)` jelzés.

---

## A. Mód: Csak vision (robot nélkül)

Akkor használd, ha nincs fizikai robot csatlakoztatva, vagy csak a kamerás felismerést
és a Stockfish elemzést akarod tesztelni.

**Szükséges:** kamera, sakktábla, bábok, modell súly fájl

---

### 1. terminál — Backend

```bash
cd /home/berci/Asztal/chess-tracker
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

Várd meg amíg megjelenik:
```
Application startup complete.
```

---

### 2. terminál — Vision pipeline

```bash
cd /home/berci/Asztal/chess-tracker
source .venv/bin/activate
python -m vision.app.run_live --no-robot
```

Megnyílik egy előnézeti ablak a kamera képével.
A státusz overlay mutatja hogy inicializált-e már a tábla.

**Billentyűk az előnézeti ablakban:**
- `r` — tracker reset (ha elcsúszott a detektálás)
- `q` — kilépés

---

### 3. terminál — Frontend (böngésző UI)

```bash
cd /home/berci/Asztal/chess-tracker
python -m tools.serve_frontend
```

Utána nyisd meg böngészőben: **http://localhost:8000**

---

### Leállítás (A. mód)

Minden terminálban: `Ctrl+C`

---
---

## B. Mód: Teljes rendszer fizikai robottal

**Szükséges:** Franka Research 3 csatlakoztatva Etherneten, kamera, sakktábla, bábok

---

### ELSŐ ALKALOMMAL (egyszer kell elvégezni)

```bash
cd /home/berci/Asztal/chess-tracker

# Docker image megépítése (~15-30 perc, internet kapcsolat szükséges)
mkdir -p ~/franka_ros2_ws/src
cd franka
docker-compose build
cd ..
```

---

### 1. terminál — Backend

```bash
cd /home/berci/Asztal/chess-tracker
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

Várd meg:
```
Application startup complete.
```

---

### 2. terminál — Robot container (MoveIt2 + executor)

```bash
cd /home/berci/Asztal/chess-tracker
./run.sh real 192.168.1.1
```

*(Az IP helyére a Franka robot tényleges Ethernet IP-je kerül.)*

Ez egy **tmux** ablakot nyit két panellel:
- **Felső panel:** MoveIt2 indul (sok szöveg, normális)
- **Alsó panel:** 8 másodperc várakozás után elindul a `chess_executor.py`

Várd meg amíg az alsó panelen megjelenik:
```
[chess_executor] Robot kész. HTTP szerver: port 8002
```

---

### 3. terminál — Kalibrálás

**Ezt minden alkalommal el kell végezni** ha a robot vagy a tábla pozíciója megváltozott.
Ha semmi nem mozdult el az előző mentett kalibrálás óta, kihagyható.

```bash
cd /home/berci/Asztal/chess-tracker
source .venv/bin/activate
python -m robot.calibrate
```

A script lépésről lépésre vezet végig:

1. Mozgasd a robotkar végét az **A1** mező fölé (fehér bástyasor, bal sarok) → Enter
2. Mozgasd a robotkar végét a **H8** mező fölé (fekete bástyasor, jobb sarok) → Enter
3. Opcionálisan: temető zóna kalibrálása (ahol a leütött bábuk kerülnek) → `y` vagy Enter a kihagyáshoz

Sikeres kalibrálás után létrejön/frissül a `robot/calibration.json` fájl.

---

### 4. terminál — Vision pipeline

**Csak akkor indítsd el, ha a kalibrálás már kész és a robotkar + kezeid nincsenek a tábla felett.**

```bash
cd /home/berci/Asztal/chess-tracker
source .venv/bin/activate
python -m vision.app.run_live
```

A program megkérdezi:
```
Calibration complete and board clear of hands/robot? [Y/n]:
```
Írd be `y` és nyomj Entert.

Megnyílik az előnézeti ablak. A pipeline:
1. Megkeresi a tábla sarkait (egyszer, induláskor)
2. 3 frame-t átlagol az alappozíció meghatározásához
3. Ha a tábla a kezdőpozícióban van: `init-ok` — kész, követi a lépéseket

**Billentyűk:**
- `r` — tracker reset
- `q` — kilépés

---

### 5. terminál — Frontend (böngésző UI)

```bash
cd /home/berci/Asztal/chess-tracker
python -m tools.serve_frontend
```

Böngészőben: **http://localhost:8000**

---

### Leállítás (B. mód)

```bash
# Robot container és tmux session leállítása:
tmux kill-session -t chess

# A többi terminálban:
Ctrl+C
```

---
---

## C. Mód: Szimuláció (robot stack robot nélkül)

Akkor használd, ha tesztelni akarod a robot stack működését fizikai robot nélkül.
**Fontos:** a vision ebben a módban nem igazán értelmes, mert a bábok fizikailag
nem mozognak — a fizikai tábla és a szoftver állapota szétcsúszik az első fekete
lépés után. Inkább az A. módot használd vision teszteléshez.

```bash
cd /home/berci/Asztal/chess-tracker
./run.sh sim
```

A többi lépés ugyanaz mint a B. módban (1, 3, 4, 5. terminál),
kivéve hogy a kalibráláskor a robot stub módban fut
(kézzel kell beírni a koordinátákat, nem olvassa a robot pozícióját).

---

## Gyors összefoglaló táblázat

| Mit indítasz | Parancs | Melyik mód |
|---|---|---|
| Backend | `uvicorn backend.main:app --host 0.0.0.0 --port 8001` | mindegyik |
| Robot (valódi) | `./run.sh real <IP>` | B. mód |
| Robot (szimuláció) | `./run.sh sim` | C. mód |
| Kalibrálás | `python -m robot.calibrate` | B. és C. mód |
| Vision (robot nélkül) | `python -m vision.app.run_live --no-robot` | A. mód |
| Vision előnézet nélkül | `python -m vision.app.run_live --no-robot --no-preview` | ha kell a CPU |
| Vision (robottal) | `python -m vision.app.run_live` | B. mód |
| Frontend | `python -m tools.serve_frontend` | mindegyik |

---

## Hibakeresés

| Hibaüzenet | Ok | Megoldás |
|---|---|---|
| `uvicorn: command not found` | nincs aktiválva a venv | `source .venv/bin/activate` |
| `Calibration file not found` | nincs még kalibrálás | `python -m robot.calibrate` |
| `Robot executor nem elérhető` | nincs futó executor | indítsd el a `./run.sh`-t |
| `detect-failed` az előnézeti ablakban | nem találja a tábla sarkait | helyezd a táblát a kamera látóterébe, A1 legyen bal felül |
| `init-too-far` az előnézeti ablakban | a bábuk nincsenek kezdőpozícióban | állítsd fel a bábokat, vagy nyomj `r`-t |
| a gép belassul, akadozik a weboldal | a vision előnézeti ablaka is CPU-t eszik | indítsd `--no-preview`-vel (nincs ablak és nincs `q`/`r`/`d` billentyű; leállítás Ctrl+C-vel, az időzítési adatok úgy is kimentődnek) |
| a frontend a régi kódot mutatja | a böngésző gyorsítótárazta az ES-modulokat | `python -m tools.serve_frontend` (no-store fejlécekkel indul) — a `python -m http.server` erre nem alkalmas |
