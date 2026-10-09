#!/usr/bin/env bash
# Toy-scale ablation sweep (A1 / A3 / A4) on the copying task.
#
# Toy scale. Two layers, 128 dimensions, 256 tokens of context, a synthetic
# copying task. These runs can show that an arm trains and can separate arms
# that differ a lot; they cannot support a claim about language modelling at
# scale, and the paper says so wherever it reports them.
#
# Two seeds per arm. Two seeds is enough to notice a large effect and not enough
# to trust a small one; the paper reports both seeds rather than a mean.

set -u
cd "$(dirname "$0")"
mkdir -p results

STEPS=${STEPS:-1000}
ARMS="dense phasor a1_frozen_rotation a3_route_on_rope a4_pure_skip"
SEEDS="0 1"
JOBS=2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_one() {
  local arm=$1 seed=$2
  local out="results/${arm}_seed${seed}.json"
  [ -f "$out" ] && { echo "skip $out"; return; }
  python3 train_toy.py \
    --variant "$arm" --seed "$seed" --steps "$STEPS" \
    --context 256 --batch 8 --vocab 64 --layers 2 --d-model 128 --heads 4 \
    --window 8 --rope-fraction 0.5 --lr 1e-3 --log-every 100 \
    --out "$out" > "results/${arm}_seed${seed}.log" 2>&1
  echo "done $arm seed$seed"
}

i=0
for arm in $ARMS; do
  for seed in $SEEDS; do
    run_one "$arm" "$seed" &
    i=$((i + 1))
    [ $((i % JOBS)) -eq 0 ] && wait
  done
done
wait
echo "sweep complete"
