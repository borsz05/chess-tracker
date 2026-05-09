# Hogyan működik a sakkrobot? — közérthető leírás

Ez a rendszer egy olyan "automatikus sakkpartner", amelyik egy kamera segítségével figyeli a fizikai sakktáblát, érti a játékot, és egy robotkar segítségével ténylegesen lép a bábuival.

---

## Hogyan "látja" a rendszer a táblát?

A tábla fölé egy kamera néz. A program nem egyszerűen lefényképezi az egészet és megkérdezi magától: "mi van ott?" — ennél okosabb módszert alkalmaz. Először megkeresi a tábla sarkait: a sakktábla jellegzetes fekete-fehér mintázata jól felismerhető nyeregpontokat hoz létre a négyzethatárokon. Ezeket a pontokat megtalálva a program "kiegyenesíti" a kamera által ferdén látott táblát, mintha egyenesen felülről nézné. Ezután a 64 mezőt egyenként megvizsgálja: mindegyikről kivág egy kis képet, és egy mesterséges intelligencia-modell eldönti, hogy az adott mező üres-e, fehér bábu van-e rajta, vagy fekete.

Fontos: a program nem tudja, hogy huszár vagy futó áll-e egy mezőn — csak annyit tud, hogy foglalt-e, és ha igen, milyen színű bábu foglalja el. Ez az egyszerűsítés szándékos: megbízhatóbb és gyorsabb, mintha 12-féle bábut próbálna felismerni. Ehelyett a sakkszabályokat használja arra, hogy kitalálja, mi mozdult.

---

## Hogyan ismeri fel, hogy lépés történt?

A kamera folyamatosan rögzít képeket — minden hatodik képkockát dolgoz fel a rendszer. Minden feldolgozásnál összehasonlítja a mostani állást az előzővel: mely mezők változtak? Ha például az e2-es mező kiürült és az e4-es mező megtelt egy fehér bábuval, a program végigmegy a lehetséges legális lépéseken (mindazok, amelyeket a sakkszabályok megengednek), és megnézi, melyik illik erre a változásra.

Ez olyan, mint egy detektív munkája: nem látta a gyilkosságot, de a nyomokból — ki hiányzik, ki jelent meg — kitalálja, mi történt.

Egy kis biztonsági mechanizmus is működik: mielőtt a rendszer "hivatalosnak" nyilvánítja a lépést, öt egymást követő képkockán át figyeli, hogy a változás valóban stabil-e. Így egy kéz véletlen átnyúlása a tábla felett nem okoz téves lépésfelismerést.

---

## Mi az a Stockfish, és hogyan "gondolkodik" a rendszer?

A Stockfish egy szoftver, amelyik sakkban játszik — és nagyon jól. Egy megakora döntési fa bejárásával milliónyi lehetséges folytatást értékel ki, minden álláshoz pontszámot rendel, és kiválasztja a legjobb lépést. A rendszer a Stockfish-t 16 lépés mélységig futtatja, és a három legjobb lehetséges folytatást is kiszámolja.

Miután a kamera regisztrálta az ember lépését, a program elküldi az állást a Stockfish-nek. Az elemzés kész, a legjobb válaszlépés kiválasztásra kerül — és ezt a lépést a robotkar hajtja végre.

---

## Mit csinál a robotkar?

A robotkar egy Franka Research 3 nevű ipari robot, amely apró, precíz mozdulatokkal képes bábukat fogni és elengedni. Minden bábunak van egy koordinátája a tábla felett: ezt egy egyszeri kalibráláskor rögzítik (az A1-es és H8-as sarkok mérésével), és a többi 62 mező helyzetét a rendszer ebből számolja ki.

Amikor a robotnak lépnie kell, a következő sorrendben cselekszik:

1. Ha ütés történik: először a leütendő bábut félrerakja egy "temető" nevű területre a tábla mellett.
2. Lefelé ereszkedik a mozgatandó bábuhoz, megfogja (20 newton erővel szorítja meg a fogó ujjait), felemeli.
3. Ívelt pályán átviszi a célmezőre — nem egyenesen, hanem egy kis domborulattal, hogy ne söpörje le az útba eső bábukat.
4. Leteszi a bábut, elengedi, visszahúzódik.

Speciális eseteket — sáncolás, en passant, gyalogcsere — a rendszer mind külön kezeli.

---

## Mik a korlátai?

A rendszer csak a bábuk színét érzékeli, típusukat nem. Ezért a játéktörténetet ("emlékezetet") a program tárolja: tudja, hogy az e1-es mezőn a fehér király áll, mert onnan kezdte és oda se lépett más. Ha viszont a játékot félbeszakítják, és nem a kezdő állásból indítják el, a program zavarba jön.

Erős oldalfény, takarás, vagy egy vendég keze is megzavarhatja a kamerát — erre a rendszer egy rövid "várakozó" üzemmóddal válaszol, amíg a helyzet nem tisztázódik.

---

## Az egész rendszer, egy mondatban

Ha analógiát keresünk: olyan ez, mintha egy vak (de emlékezőképes) bírónak két játékos játszana, aki csak annyit érzékel, hogy egy szék üres vagy foglalt — és a lépéseket ebből, a szabálykönyv alapján rekonstruálja —, miközben egy gépi eszed tanácsaira hallgatva maga is lép a saját bábuival egy robotkéz segítségével.
