# Első valódi robotpróba — lépésről lépésre

Ez a lap arra való, hogy a laborban ne kelljen gondolkodni. Fentről lefelé,
és minden lépésnél ott van, mit kell látni.

A `robot-integracio` branchen vagy? Ellenőrizd: `git branch --show-current`.

---

## 0. Preflight — még mielőtt bármit indítanál

```bash
bash tools/robot_preflight.sh 192.168.1.1 > robot_preflight.txt 2>&1
```

Semmit nem mozgat. Utána nézd meg a fájl elejét:

| Amit keresel | Miért |
|---|---|
| `PREEMPT_RT` sor | ha `is not set`, a kernel nem valós idejű — ez később okozhat `communication_constraints_violation`-t |
| default útvonalak száma | ha 1-nél több, a robot felé menő forgalom rossz interfészen mehet ki |
| ping statisztika | FCI-hez 0% loss és < 1 ms kell; az USB-C adapter itt bukna meg |

---

## 1. Robot stack indítása

```bash
./run.sh real 192.168.1.1
```

A felső panelen a MoveIt indul. Az alsó panel **már nem 8 másodpercet vár vakon**,
hanem addig, amíg a `move_group` tényleg megjelenik — ezt ki is írja:
`move_group él (12s)`.

Utána az executor indul, és **kiírja, mit talált meg**:

```
[chess_executor] Felderítés: {"recovery_action": "...", "franka_state_topic": "...", "ready": {...}}
```

Ez a sor a legfontosabb az egész próbán. Ha valamelyik `ready` mező `false`,
az ott a baj, és nem kell tovább keresni.

---

## 2. Állapot lekérdezése — mozgás nélkül

```bash
curl -s localhost:8002/status | python3 -m json.tool
```

Amit nézni kell:

- `position` — ha `null`, a TF nem él, és semmi más nem fog működni
- `franka_errors` — ha nem üres, a robot **hibaállapotban van**, előbb ezt kell feloldani
- `capabilities.ready` — minden `true`?
- `capabilities.recovery_action` — ha `null`, ezen a stacken nincs hibafeloldás, és egy reflex után újra kell indítani

Ha van aktív hiba:

```bash
curl -s -X POST localhost:8002/recover | python3 -m json.tool
```

---

## 3. Az első mozgás

Ez az a pont, ahol eddig elbukott. Egyetlen pose, biztonságos magasságban:

```bash
curl -s -X POST localhost:8002/move -H 'Content-Type: application/json' -d '{"x":400,"y":0,"z":300}' | python3 -m json.tool
```

- **Ha `{"ok": true}`** — a kar megmozdult, a fő hiba javítva van. Mehetsz tovább.
- **Ha hiba jön**, az most már részletes: `where`, `error`, `franka_errors`, `recovery`.
  Ezt a JSON-t másold ki egészben — ebből lehet dolgozni, a korábbi néma hibából nem lehetett.

Ha nem mozdul, de hibát sem ad: nézd meg a felső tmux panelt (MoveIt naplója),
és a `/status`-t újra.

---

## 4. Gripper

```bash
curl -s -X POST localhost:8002/gripper/open
curl -s -X POST localhost:8002/gripper/close
```

Mostantól a gripper hibát is jelez: ha nincs semmi a fogóban, a `close`
`success: false`-t ad vissza, és ez **nem tűnik el csendben**, mint eddig.

---

## 5. Asztalmagasság ellenőrzése (2 perc, de fontos)

A `Z_PICK`/`Z_PLACE` konstansok azt feltételezik, hogy az asztal lapja a robot
bázisában `z = 0`. Ezt még soha senki nem ellenőrizte, pedig minden fogás
magassága ezen áll.

Ereszd le a megfogót, amíg épp hozzáér az asztalhoz (`/move` hívásokkal,
20 mm-es lépésekben, a végén 2 mm-esekkel), majd:

```bash
curl -s localhost:8002/position | python3 -m json.tool
```

A `z` értéke ~0 legyen. Ha nem az, jegyezd fel — ez egy eddig ismeretlen
eltolás, ami minden fogásba beleszámít.

---

## 6. Kalibrálás

```bash
python -m robot.calibrate
```

**Közben figyeld egy másik terminálban**, hogy a stack él-e:

```bash
watch -n1 'docker exec franka_ros2_humble bash -lc "source /opt/ros/humble/setup.bash; ros2 node list | grep -c ." '
```

A gyanú az, hogy a kézi vezetéshez használt guiding gombok leütik a
`franka_hardware` node-ot, és emiatt halt el a kalibrálás a **második** sarok
előtt (lásd `docs/robot_debug_brief.md` 4b. szakasz). Ha a node-szám lecsökken,
megvan az ok — és akkor a kalibrálást a `/move` endpointtal kell csinálni,
kézi vezetés helyett.

---

## 7. Egy teljes sakklépés

```bash
python test_robot.py
```

`6` → `e2e4` → `i`. A kar a mostani javítás után **először biztonságosan rááll**
a forrásmezőre (felemelkedik, vízszintesen halad `Z_TRAVEL`-en, leereszkedik),
és csak utána indul a pick-and-place. Ez a rész robot nélkül le van tesztelve
(`tests/test_executor_waypoints.py`), de a valódi geometriát csak itt látod.

---

## Ha valami elromlik

Minden hiba a `/status` `last_error` mezőjében marad meg, a Franka hibajelzőkkel
együtt. Kérdés esetén ezt küldd:

```bash
curl -s localhost:8002/status | python3 -m json.tool > robot_status.txt
```
