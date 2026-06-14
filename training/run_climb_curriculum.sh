#!/bin/zsh
# Height curriculum for the G1 step-up, validated 2026-06-14: floor->platform is
# UNLEARNABLE jumping straight to 0.08 m but learnable at 0.02 m (floor-start 10%).
# So climb the height ladder, each stage --resume-ing the previous height's policy.
# CPU (measured 12.7k fps vs MPS 6.8k on this env) + 24 envs + RSI 0.5. ~2 h.
set -e
cd /Users/hoshinafumito/development/Colapis_project/MuJoCo-skills
export G1CLIMB_RSI_P=0.5
PY=.venv-rl/bin/python
TR=training/g1_climb_curriculum_train.py
R=runs_climb
LOG=$R/ladder.log
: > $LOG

echo "===STAGE strengthen h=0.02 ===" | tee -a $LOG
$PY $TR --step_h 0.02 --resume $R/climb_curric_h0.02_latest --device cpu \
    --steps 8000000 --envs 24 >> $LOG 2>&1

heights=(0.04 0.06 0.08 0.11 0.14 0.17 0.20 0.22)
steps=(8000000 8000000 10000000 10000000 12000000 14000000 16000000 18000000)
prev=0.02
for i in {1..${#heights[@]}}; do
    h=${heights[$i]}; s=${steps[$i]}
    echo "===STAGE h=$h steps=$s resume=h$prev ===" | tee -a $LOG
    $PY $TR --step_h $h --resume $R/climb_curric_h${prev}_latest --device cpu \
        --steps $s --envs 24 >> $LOG 2>&1
    prev=$h
done
echo "===LADDER DONE final=h0.22 ===" | tee -a $LOG
