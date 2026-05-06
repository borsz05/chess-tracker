#!/bin/bash
#
# Indítás:
#   ./run.sh              → szimuláció
#   ./run.sh real <IP>    → valódi Franka Research 3  (pl. ./run.sh real 192.168.1.1)
#
# Első telepítéshez (új gépen) a Docker image buildelése:
#   cd franka && docker-compose up -d
#
# Leállítás:
#   tmux kill-session -t chess

MODE="${1:-sim}"
ROBOT_IP="${2:-}"
SESSION="chess"
CHESS_DIR="$(cd "$(dirname "$0")" && pwd)"
FRANKA_DIR="$CHESS_DIR/franka"

# ── Argumentumok ellenőrzése ─────────────────────────────────────────────────
case "$MODE" in
    sim)
        USE_FAKE="true"
        ROBOT_IP_ARG="dont-care"
        echo "▶ Mód: SZIMULÁCIÓ"
        ;;
    real)
        if [ -z "$ROBOT_IP" ]; then
            echo "Hiba: valódi robot módhoz IP cím szükséges."
            echo "Használat: $0 real <ROBOT_IP>"
            exit 1
        fi
        USE_FAKE="false"
        ROBOT_IP_ARG="$ROBOT_IP"
        echo "▶ Mód: VALÓDI robot  IP=$ROBOT_IP"
        ;;
    *)
        echo "Ismeretlen mód: '$MODE'"
        echo "Használat: $0 [sim | real <ROBOT_IP>]"
        exit 1
        ;;
esac

# ── Docker container indítása ────────────────────────────────────────────────
echo "▶ Docker container indítása..."
xhost +local:docker 2>/dev/null || true

# Ha már létezik a container: egyszerűen indítjuk (nem buildelünk újra)
# Ha nem létezik: a franka/docker-compose.yml-ből hozzuk létre
if docker ps -a --format '{{.Names}}' | grep -q '^franka_ros2_humble$'; then
    docker stop franka_ros2_humble 2>/dev/null || true
    docker start franka_ros2_humble
else
    echo "  Container nem létezik — buildelés a franka/docker-compose.yml alapján..."
    cd "$FRANKA_DIR" && docker-compose up -d
fi

# Mindig a legfrissebb executor.py-t másoljuk be a containerbe
echo "▶ executor.py frissítése a containerben..."
docker cp "$CHESS_DIR/franka/chess_executor.py" franka_ros2_humble:/ros2_ws/chess_executor.py
echo "▶ Container kész."

# ── tmux session felépítése ──────────────────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Felső pane — MoveIt2 launch a Docker containerben
tmux new-session -d -s "$SESSION" -n "robot" \
    "docker exec franka_ros2_humble bash -c 'source /ros2_ws/install/setup.bash && ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=$ROBOT_IP_ARG use_fake_hardware:=$USE_FAKE fake_sensor_commands:=false'; echo '--- MoveIt2 leállt, nyomj Entert ---'; read"

# Alsó pane — chess_executor HTTP szerver (8mp várakozás a MoveIt2 után)
tmux split-window -v -p 40 -t "$SESSION:0" \
    "echo '>>> Várakozás MoveIt2 indulásra (8mp)...' && sleep 8 && docker exec franka_ros2_humble bash -c 'source /ros2_ws/install/setup.bash && python3 /ros2_ws/chess_executor.py'; echo '--- Executor leállt, nyomj Entert ---'; read"

tmux select-pane -t "$SESSION:0.0"
tmux attach-session -t "$SESSION"
