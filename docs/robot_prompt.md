# Prompt a Fable 5.1-nek — robotos szakasz

Ezt másold be a Fable chatbe. A `robot_preflight.txt`-t is csatold/illeszd be, ha már megvan.

---

A `chess-tracker` projekt robotos részét kell működésre bírni. A vision pipeline kész és
validált — **ahhoz nem nyúlsz**.

**Előbb olvasd el: `docs/robot_debug_brief.md`.** Az elemzés már megtörtént: benne van a
tünet, a hardveres összeállítás, öt rangsorolt hipotézis konkrét fájl:sor hivatkozásokkal,
és hogy mihez nem szabad hozzányúlni. Ne kezdd elölről a felderítést, és ne térképezd fel
a kódbázist — a brief a kiindulópont.

## Költségkeret — ez kemény korlát

Kb. **15 euró usage kredit** maradt, ennyiből kell végigérni. Ezért:

- **Felülről lefelé haladj.** A briefben H1–H5 fontossági sorrendben van. Ne menj mélységbe,
  amíg a fenti szint nincs kész.
- Csak azokat a fájlokat olvasd be, amiket a brief megnevez. A `vision/`, `frontend/`,
  `tools/`, `tests/` mappákat **ne olvasd**.
- Ne írj új teszt-infrastruktúrát, ne refaktorálj, ne optimalizálj, ne javíts stílust.
  Kizárólag azt, ami a robot megmozdulásához kell.
- Ne mérj és ne profilozz semmit. Ez nem teljesítmény-feladat.
- Ha valami a host gépen múlik (kernel, hálózat, adapter), **ne próbáld kódból megkerülni** —
  írd le egy mondatban, és lépj tovább.

## Munkamenet — három szakasz, mindegyik végén ÁLLJ MEG

Minden szakasz után adj egy rövid jelentést, és várd meg a válaszomat, mielőtt folytatod.
Ne fusd végig egyben mindhármat.

### 1. szakasz — Láthatóvá tenni a hibát  (ez a legfontosabb, ~1/3 a keretből)

A felhasználó soha nem látta a valódi Franka hibakódot, mert a kód elnyeli. Amíg ez nincs
meg, minden javítás találgatás.

- A `franka/chess_executor.py`-ben a MoveIt tervezés és végrehajtás eredményét (error code)
  ki kell olvasni és a HTTP válaszba tenni, a `_move_to`-ban is, nem csak a
  `_cartesian_move`-ban.
- Fel kell iratkozni a Franka hibaállapotára, és a `/health` (vagy egy új `/status`)
  endpointnak meg kell mutatnia: van-e aktív reflex, és melyik.
- Kell egy `POST /recover` endpoint, ami lefuttatja a Franka error-recovery actionjét;
  és végrehajtási hiba után a rendszer próbálja meg magától.
- Írd meg egy `POST /move`-ot használó minimális próbaparancsot, amit a felhasználó
  egyetlen sorral lefuttathat.

**Jelentés végén:** mit fog most látni a felhasználó, ha újra próbálja, és pontosan mit
futtasson le, hogy megkapd a valódi hibaüzenetet.

### 2. szakasz — A start-állapot javítása  (csak az 1. szakasz visszajelzése után)

A H1 és H4 hipotézis javítása:

- A Cartesian szegmensek ne tartalmazzák a kiindulási pontot célként. Az első waypointra
  külön, tervezett (joint-space) mozgással kell eljutni, és onnan indul a Cartesian pálya.
  Ellenőrizd a pymoveit2 tényleges szemantikáját a containerben:
  `/ros2_ws/src/pymoveit2/pymoveit2/moveit2.py`.
- Minden szekvencia elején legyen egy explicit „menj a kiindulási pozícióba" lépés, hogy a
  `_pick_and_place` ne feltételezze, hol áll a kar.
- Az `_init_robot` várja meg ténylegesen az első `/joint_states`-t és a MoveIt action
  szerverek elérhetőségét, ne fix `time.sleep(1.5)`-tel.
- A gripper-hívásokban szűnjön meg a dupla spin ugyanazon a node-on.

### 3. szakasz — Végigvitel  (csak ha a 2. szakasz működik valódi roboton)

- Egy teljes `simple` lépés (`e2e4`) fusson végig.
- A `run.sh` fix `sleep 8`-a helyett tényleges készenlét-várakozás.
- A `_CAL_PATH` / `calibration.json` bemásolási hibája.
- Frissítsd a `STARTUP.md` robotos szakaszát azzal, ami ténylegesen működik.

## Amit tudnod kell a tesztelésről

Valódi robot van, de **én futtatom a parancsokat, nem te** — nincs hozzáférésed a robothoz.
Ezért minden szakasz végén add meg pontosan, mit írjak be a terminálba, és mit küldjek
vissza neked. Feltételezd, hogy a `./run.sh real <IP>` már fut.

Ha a `robot_preflight.txt` a beszélgetésben van, olvasd el az elején — az tartalmazza a
kernel, hálózat, ROS-kontroller és Franka hibaállapot pillanatképét.

Magyarul válaszolj.
