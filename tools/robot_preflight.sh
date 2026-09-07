#!/usr/bin/env bash
#
# Robot preflight — CSAK OLVAS, semmit nem mozgat és nem indít.
#
# Cél: egyetlen futtatással összegyűjteni minden tényt, amit különben a
# modell drága körökben derítene ki. A kimenetet mentsd fájlba és add oda
# a Fable-nek:
#
#   bash tools/robot_preflight.sh 192.168.1.1 > robot_preflight.txt 2>&1
#
# Az IP a Franka vezérlő IP-je (ugyanaz, amit a ./run.sh real <IP> kap).

ROBOT_IP="${1:-}"
CONTAINER="franka_ros2_humble"

hr()  { printf '\n===== %s =====\n' "$1"; }
try() { timeout 15 "$@" 2>&1 || echo "  [nem futott le: $*]"; }
inc() { timeout 20 docker exec "$CONTAINER" bash -lc "source /opt/ros/humble/setup.bash 2>/dev/null; source /ros2_ws/install/setup.bash 2>/dev/null; $1" 2>&1 || echo "  [nem futott le a containerben]"; }

hr "1. HOST KERNEL"
uname -a
printf 'PREEMPT_RT: '
grep -m1 'PREEMPT_RT' "/boot/config-$(uname -r)" 2>/dev/null || echo "(nem olvasható a kernel config)"
printf 'rtprio limit: '; ulimit -r 2>/dev/null || echo "?"
printf 'memlock limit: '; ulimit -l 2>/dev/null || echo "?"

hr "2. HALOZATI INTERFESZEK"
try ip -br addr
echo "--- utvonalak ---"
try ip route
echo "--- alapertelmezett utvonalak szama (1-nel tobb gond lehet) ---"
ip route 2>/dev/null | grep -c '^default'

hr "3. ROBOT ELERHETOSEG"
if [ -z "$ROBOT_IP" ]; then
    echo "Nem adtal meg IP-t. Hasznalat: bash tools/robot_preflight.sh <ROBOT_IP>"
else
    echo "--- melyik interfeszen megy ki a forgalom ---"
    try ip route get "$ROBOT_IP"
    echo "--- 100 ping, keslekedes-statisztika (FCI-hez < 1 ms es 0% loss kell) ---"
    try ping -c 100 -i 0.01 -q "$ROBOT_IP"
    echo "--- Desk elerheto? ---"
    try curl -sk -o /dev/null -w 'HTTP %{http_code}  csatlakozas %{time_connect}s\n' "https://$ROBOT_IP"
fi

hr "4. USB ETHERNET ADAPTER"
try lsusb
echo "--- interfesz driverek ---"
for i in /sys/class/net/*; do
    n=$(basename "$i")
    [ "$n" = "lo" ] && continue
    d=$(readlink -f "$i/device/driver" 2>/dev/null | xargs -r basename)
    echo "  $n  driver=${d:-?}  $(cat "$i/operstate" 2>/dev/null)"
done

hr "5. DOCKER CONTAINER"
try docker ps -a --filter "name=$CONTAINER" --format '{{.Names}}  {{.Status}}'

hr "6. ROS2 ALLAPOT A CONTAINERBEN"
echo "--- node-ok ---";        inc "ros2 node list"
echo "--- kontrollerek ---";   inc "ros2 control list_controllers"
echo "--- hardware ---";       inc "ros2 control list_hardware_interfaces | head -40"
echo "--- franka topicok ---"; inc "ros2 topic list | grep -i franka"
echo "--- action szerverek ---"; inc "ros2 action list"

hr "7. FRANKA HIBAALLAPOT"
for t in /franka_robot_state_broadcaster/current_errors \
         /franka_robot_state_broadcaster/robot_state \
         /franka_state_controller/franka_states; do
    echo "--- $t ---"
    inc "timeout 5 ros2 topic echo --once $t"
done
echo "--- error recovery action letezik-e ---"
inc "ros2 action list | grep -i recover"

hr "8. MOVEIT PARAMETEREK (start-tolerancia)"
inc "ros2 param list /move_group 2>/dev/null | grep -i -E 'trajectory_execution|tolerance'"
inc "ros2 param get /move_group trajectory_execution.allowed_start_tolerance"

hr "9. AKTUALIS IZULETALLAS"
inc "timeout 5 ros2 topic echo --once /joint_states"

hr "VEGE"
echo "Ezt a kimenetet add oda a modellnek."
