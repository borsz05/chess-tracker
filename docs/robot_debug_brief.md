# Robotkar — hibakeresési brief

Készült: 2026-09-07. Ez a dokumentum a **kiindulási tudás** a robotos szakaszhoz.
A benne leírt elemzés már megtörtént — **ne derítsd ki újra**, csak ellenőrizd és javíts.

## Állapot (2026-09-08, `robot-integracio` branch)

Robot nélkül elvégezve — **ezeket ne csináld meg újra**:

- **H1 javítva**: a Cartesian pályák már nem a kar aktuális pózából indulnak;
  `_approach` biztonságos ráállással, az invariáns a `_run_cartesian`-ban van
  kikényszerítve. 13 teszt: `tests/test_executor_waypoints.py`.
- **H2 javítva**: `/recover` endpoint, futásidőben megkeresett hibafeloldó
  action; végrehajtási hiba után magától lefut (a lépést nem ismétli meg).
- **H3 javítva**: `/status` endpoint, strukturált hibák, a gripper action és a
  MoveIt kimenetelének tényleges ellenőrzése. 11 további teszt.
- **H4 javítva**: megszűnt a dupla spin; callback group, 4 executor szál,
  `_wait_ready` a fix `time.sleep(1.5)` helyett.
- Apróságok: `_CAL_PATH` több helyen keres, `run.sh` bemásolja a
  `calibration.json`-t és a `move_group` tényleges indulására vár.

**H5 (kernel, USB-C adapter, útvonalválasztás) nyitva** — host-szintű, csak a
felhasználó tudja megcsinálni. A `tools/robot_preflight.sh` méri.

A valódi robotos próba menete: `docs/robot_elso_proba.md`.

---

## 1. A tünet (a felhasználó szavaival)

> „Már próbálgattam a robotkart, de még egyszer se sikerült irányítanom kódból, se a futó
> programmal. Valami szinkronizálási hiba lehetett, mert mindig lecsapott a robot és nem
> mozdult meg, mert úgy érzékelte, mintha túl gyorsan indult volna el — azért, mert a
> kiinduló állapot a gépen és a tényleges kiinduló állapot között lehet hogy eltérés volt."

Fontos: **a robot soha nem mozdult meg**, tehát a hiba a legelső mozgásparancsnál üt be.
Ez nem kalibrációs és nem sakklogikai probléma.

## 2. A fizikai összeállítás

- Franka Research 3, Ethernet.
- A laptopon **két Ethernet kapcsolat** egyszerre: az egyik a robothoz, a másik a LAN-hoz.
- A robot felé **USB-C → Ethernet adapter (1000 Mbps)** van közbeiktatva, nem beépített NIC.
- Host kernel: `7.0.0-31-generic`, **PREEMPT_DYNAMIC** — a `/boot/config-*`-ban
  `# CONFIG_PREEMPT_RT is not set`. **Ez nem valós idejű kernel.**
- A ROS2 stack Dockerben fut (`franka/docker-compose.yml`, `network_mode: host`,
  `privileged: true`, rtprio 99).

## 3. Hipotézisek fontossági sorrendben

### H1 — A Cartesian pálya első pontja nem a robot aktuális pózja (legvalószínűbb)

`franka/chess_executor.py`:

- `_pick_and_place` (237–247) első fázisa `_vertical_path(fx, fy, Z_LIFT, z_pick)`.
- `_vertical_path` (208–214) waypointjai: `[[fx, fy, Z_LIFT], [fx, fy, z_pick]]`.
- `_arc_path` (217–234) szintén a **kiindulási pontot is beleteszi** a waypoint-listába.

A `MoveIt2.compute_cartesian_path` a waypointokat az **aktuális pózhoz képest** értelmezi:
a `waypoints[0]` egy *célpont*, nem az „itt vagyok" pont. A robot viszont a szekvencia
elején bárhol lehet — indulás után ott, ahol hagyták; egy lépés után a *cél* mező fölött.
Így a generált trajektória első szegmense egy ugrás az aktuális pózból a `waypoints[0]`-ba.

Ennek két következménye van, és **mindkettő pontosan a leírt tünetet adja**:
- MoveIt `trajectory_execution.allowed_start_tolerance` (alapértelmezés 0.01 rad) →
  `Invalid Trajectory: start point deviates from current robot state`, a mozgás el sem indul;
- ha mégis elindul, a Franka a nem folytonos indulási sebességre reflexet dob
  (`joint_motion_generator_velocity_discontinuity` / `cartesian_motion_generator_*`),
  ami után a kar bemerevedik.

`VELOCITY = ACCELERATION = 0.05` (69–70), azaz 5% skálázás — a robot **nem attól gyors**,
hogy a sebességhatár magas. Ez megerősíti, hogy indulási ugrásról van szó.

### H2 — Nincs hibafeloldás (error recovery)

Reflex után a Franka hibaállapotba kerül, és **minden további parancsot elutasít**, amíg
nem fut le az error-recovery action. A kódban sehol nincs ilyen hívás. Ez magyarázza, hogy
az *első* hiba után semmi nem működött többé — a felhasználó valószínűleg végig egy be nem
ismert hibaállapotú robotot próbált vezérelni.

### H3 — A hibák némán elvesznek

- `_move_to` (165–171) nem ellenőrzi a MoveIt eredményét; a pymoveit2 tervezési hiba esetén
  általában nem dob kivételt, csak nem csinál semmit.
- `_assert_reached` (174–187) csak a `_cartesian_move`-ban fut, a `_move_to`-ban nem.
- A HTTP handler (328–329) minden kivételt egyetlen `str(e)`-vé lapít.

Emiatt a felhasználó soha nem látta a valódi Franka hibakódot. **A legfontosabb egyetlen
javítás az, hogy a valódi hibaüzenet megjelenjen** — enélkül minden további lépés találgatás.

### H4 — rclpy dupla spin

`_init_robot` (106–108) egy `MultiThreadedExecutor`-t pörget háttérszálon a `_node`-on.
Ezután `_gripper_open` / `_gripper_grasp` (140–160) ugyanazon a node-on hív
`rclpy.spin_until_future_complete`-et. Ugyanazt a node-ot két helyről pörgetni rclpy-ban
nem megengedett: futásidejű hibát vagy holtpontot okoz. A gripper-hívásokat a már futó
executorra kell bízni (`future.result()` várakozás vagy `spin_until_future_complete` a
saját executorral), nem újra pörgetni.

Ugyanitt: `num_threads=2` szűk a MoveIt2 + TF + két action kliens mellé, és a pymoveit2
callback group beállítása hiányzik. A `time.sleep(1.5)` (128) nem garantálja, hogy a
`/joint_states` már megérkezett — **ha a pymoveit2 belső joint-state-je üres vagy elavult,
az önmagában is „a gépen lévő kiinduló állapot ≠ tényleges" hibát ad.**

### H5 — Hálózat és kernel (host-szintű, nem kódból javítható)

- Az FCI 1 kHz-es vezérlést vár. USB Ethernet adapter jitteret visz be, és a Franka
  dokumentáció beépített NIC-et ajánl. Tünete `communication_constraints_violation` lenne.
- **A kernel nem PREEMPT_RT.** Ez a Franka FCI hivatalos követelménye.
- Két aktív interfész mellett ellenőrizni kell, hogy a robot IP-je tényleg a robotos
  interfészen megy-e ki (két default route esetén nem biztos).

Ezeket a `tools/robot_preflight.sh` kimenete méri. **Ne próbáld kódból megkerülni** —
ha ez a szűk keresztmetszet, az a felhasználó feladata (RT kernel telepítése, adapter
cseréje), és meg kell mondani neki.

## 4. Egyéb, biztosan hibás apróságok

| Hely | Probléma |
|---|---|
| `chess_executor.py:75` | `_CAL_PATH` = a script mappája = `/ros2_ws/`, de a `run.sh:59` csak a `chess_executor.py`-t másolja be. A `calibration.json` sosincs ott → a `piece_grasp_heights` override néma halott kód. |
| `run.sh:71` | Fix `sleep 8` a MoveIt indulására. Ehelyett a tényleges készenlétre kell várni. |
| `chess_executor.py:44` | `_DOWN_QUAT = [1,0,0,0]` `quat_xyzw` sorrendben `x=1` — ellenőrizendő, hogy tényleg lefelé néző TCP-t jelent-e. |
| `robot/calibration.json` | Jelenleg **nem létezik**. A `RobotExecutionService` emiatt indulásnál elszáll, a backend vision-only módba esik. |

## 4b. Második tünet: a kalibrálás az első sarok után elhalt

> „Elkezdtem a kalibrálást, az első sarok megvolt, majd a rendszer bedöglött,
> vagy a robottal volt megint valami baj."

A `robot/calibrate.py` maga nem mozgatja a kart: a kezelő kézzel odavezeti a sarok fölé,
Entert nyom, és a script kiolvassa a `/position`-t (TF `fr3_link0 → fr3_hand_tcp`).
Tehát a hiba nem a kalibráló logikában van.

Legvalószínűbb ok (**ellenőrizendő, nem bizonyított**): a Franka kézi vezetéséhez a karon
lévő guiding gombokat kell nyomni, ami átveszi a vezérlést. Ha közben az FCI vezérlőhurok
aktív, a robot hibát dob, és a `franka_hardware` node gyakran leáll — ezzel viszont a
MoveIt és a TF is elmegy, így a **második** sarok `/position` hívása már nem tud mit
kiolvasni. Ez pontosan „az első sarok megvolt, aztán bedöglött" mintázat.

Ellenőrzés: kalibrálás közben nézd meg, él-e még a `franka_hardware` node és jön-e a
`/joint_states`. Ha a guiding mód valóban leüti a stacket, a kalibrálást vagy guiding
módban futó, de FCI nélküli állapotban kell csinálni, vagy a kart a `/move` endpointtal
kell pozicionálni kézi vezetés helyett.

(A `_get_position` időközben kapott egy rövid újrapróbálkozást, így most legalább értelmes
hibaüzenetet ad nyers TF-kivétel helyett — de ez nem javítja meg az okot.)

## 5. Amihez NEM szabad hozzányúlni

- `vision/` — a képfeldolgozó pipeline le van mérve és validálva, ez a feladat nem érinti.
- `raw_to_standard` és a tábla-orientáció logikája.
- `frontend/` és `tools/timing_dashboard.py`.
- A sakklogika (`chess_logic/`, `robot/move_descriptor.py`) — ezek unit teszttel fedettek.

## 6. Kész állapot definíciója

1. Egyetlen mozgásparancs (`POST /move`) hibátlanul lefut valódi roboton, tetszőleges
   kiindulási pózból.
2. Ha nem fut le, a HTTP válasz **a valódi Franka/MoveIt hibakódot** tartalmazza, nem
   általános szöveget.
3. Reflex után a rendszer magától kilábal (error recovery), vagy világosan megmondja,
   hogy manuális beavatkozás kell.
4. Egy teljes `simple` sakklépés (`e2e4`) végigfut.
