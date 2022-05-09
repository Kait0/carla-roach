#!/bin/bash

export CARLA_ROOT=/home/kchitta/Documents/CARLA_0.9.10.1
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

export SCENARIO_RUNNER_ROOT=/home/kchitta/Documents/carla_planning/scenario_runner
export LEADERBOARD_ROOT=/home/kchitta/Documents/carla_planning/leaderboard
export TEAM_CODE_ROOT=/home/kchitta/Documents/carla_planning/team_code_planner
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export TEAM_AGENT=${TEAM_CODE_ROOT}/privileged_agent.py
export TEAM_CONFIG=${TEAM_CODE_ROOT}/model_ckpt
export CHALLENGE_TRACK_CODENAME=MAP
export REPETITIONS=1
export CHECKPOINT_ENDPOINT=../misc/carla_results/longest2_mpc.json # output results file
export SAVE_PATH=../misc/carla_results/privileged_agent/longest2 # path for saving episodes (comment to disable)

export DEBUG_CHALLENGE=0
export DISABLE_RENDERING=0
export INCREMENT_DP_SEED=0
export ROUTES=leaderboard/data/longest2.xml
export SCENARIOS=leaderboard/data/scenarios/eval_scenarios.json

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME}

