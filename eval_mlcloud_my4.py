import subprocess
import time
from pathlib import Path
import os
import fnmatch
import json

def get_free_port(initial_port=15000):
    """
    Returns a free port.
    """

    port = initial_port
    port_free = False

    while not port_free:
        try:
            pid = int(
                subprocess.check_output(
                    f"lsof -t -i :{port} -s TCP:LISTEN",
                    shell=True,
                )
                    .decode("utf-8")
            )
            # print(f'Port {port} is in use by PID {pid}')
            port += 5

        except subprocess.CalledProcessError:
            port_free = True
            return port
            # print(f'Port {port} is free')

def get_carla_command(gpu, num_try, start_port):
    command = f"SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE={int(gpu)} /mnt/qb/work/geiger/bjaeger25/CARLA/CarlaUE4.sh --world-port={int(gpu)*1000+start_port+num_try*10} -opengl"
    return command


def create_run_eval_bash(bash_save_dir, results_save_dir, route_path, iter, route, model, start_port, tm_port, checkpoint):
    print("Port: ", start_port+iter*10)
    print("TM_Port: ", str(int(start_port+iter*10 + tm_port)))
    Path(f'{results_save_dir}').mkdir(parents=True, exist_ok=True)
    with open (f'{bash_save_dir}/eval_{route}_{Path(model).name}.sh', 'w') as rsh:
        rsh.write('''\
export CARLA_ROOT=/mnt/qb/work/geiger/bjaeger25/CARLA
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=scenario_runner
export LEADERBOARD_ROOT=leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
''')
        rsh.write(f'''
export PORT={start_port+iter*10}
export TM_PORT={str(int(start_port+iter*10 + tm_port))}
export ROUTES={route_path}{route}.xml
export SCENARIOS=leaderboard/data/scenarios/eval_scenarios.json
export TEAM_AGENT=agents1/rl_birdview/rl_birdview_agent_local.py
export TEAM_CONFIG=agents1/rl_birdview/config_agent.yaml
export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export RESUME=1
export CHECKPOINT_ENDPOINT={results_save_dir}/{route}.json
export SAVE_PATH_DISABLED={results_save_dir}/{route}
export DEBUG_CHALLENGE=0
export DISABLE_RENDERING=0
export INCREMENT_DP_SEED=0
''')
        rsh.write('''
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
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}
''')


def make_jobsub_file(commands, job_number, exp_name):
    os.makedirs(f"evaluation/{exp_name}/run_files/logs", exist_ok=True)
    os.makedirs(f"evaluation/{exp_name}/run_files/job_files", exist_ok=True)
    job_file = f"evaluation/{exp_name}/run_files/job_files/{job_number}.sh"
    qsub_template = f"""#!/bin/bash
#SBATCH --job-name={exp_name}{job_number}
#SBATCH --partition=gpu-2080ti-dev
#SBATCH -o evaluation/{exp_name}/run_files/logs/qsub_out{job_number}.log
#SBATCH -e evaluation/{exp_name}/run_files/logs/qsub_err{job_number}.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=00-12:00
#SBATCH --gres=gpu:1
"""
    for cmd in commands:
        qsub_template = qsub_template + f"""
{cmd}

"""

    with open(job_file, "w") as f:
        f.write(qsub_template)
    return job_file


def get_num_jobs(job_name, username="bjaeger25"):
    # print(job_name)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:7,name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        )
        .decode("utf-8")
        .replace("\n", "")
    )
    max_num_parallel_jobs = int(open("max_num_jobs.txt", "r").read())
    return num_running_jobs, max_num_parallel_jobs


if __name__ == "__main__":
    exp_names_tmp = ['Roach_longest6e1', 'Roach_longest6e2', 'Roach_longest6e3']
    model = 'latest'
    route_path = 'leaderboard/data/longest6_split/'
    route_pattern = '*.xml'

    epochs = ['model_37']
    epoch_id = 0
    for epoch in epochs:
        exp_names = []
        for name in exp_names_tmp:
            exp_names.append(name)#+ '_' + epoch)

        carla_port = 15000 + 1000*epoch_id
        carla_port = get_free_port(carla_port)
        tm_port = 10001 + 1000*epoch_id #Gets added to the carla_port
        epoch_id += 1
        checkpoint = 'TransFuserAllNLayer4NoVelocityTPReg32Reg32_Ensemble'
        checkpoint_new_name = checkpoint #+ '_' + epoch

        copy_model = False

        if copy_model:
            # copy checkpoint to my folder
            cmd = f"mkdir team_code_{model}/checkpoints/{checkpoint_new_name}"
            print(cmd)
            os.system(cmd)
            cmd = f"cp /mnt/qb/geiger/bjaeger25/training_logdir/{checkpoint}/args.txt team_code_{model}/checkpoints/{checkpoint_new_name}/"
            print(cmd)
            os.system(cmd)
            cmd = f"ln -sf /mnt/qb/geiger/bjaeger25/training_logdir/{checkpoint}/{epoch}.pth team_code_{model}/checkpoints/{checkpoint_new_name}/model.pth"
            print(cmd)
            os.system(cmd)

        #exit(-1)

        route_files = []
        for root, dirs, files in os.walk(route_path):
            for name in files:
                if fnmatch.fnmatch(name, route_pattern):
                    route_files.append(os.path.join(root, name))

        for exp_name in exp_names:
            bash_save_dir = Path(f'evaluation/{exp_name}/run_bashs')
            results_save_dir =  Path(f'evaluation/{exp_name}/results')
            exps_save_dir =  Path(f'evaluation/{exp_name}/exps')
            bash_save_dir.mkdir(parents=True, exist_ok=True)
            results_save_dir.mkdir(parents=True, exist_ok=True)

        iter = 0
        job_nr = 1

        meta_jobs = {}

        for ix, route in enumerate(route_files):
            route = Path(route).stem
            for exp_name in exp_names:
                bash_save_dir = Path(f'evaluation/{exp_name}/run_bashs')
                results_save_dir =  Path(f'evaluation/{exp_name}/results')
                exps_save_dir =  Path(f'evaluation/{exp_name}/exps')

                commands = []

                carla_cmd = get_carla_command(0, iter, carla_port)
                commands.append(f'{carla_cmd} &')
                commands.append('sleep 180')
                create_run_eval_bash(bash_save_dir, results_save_dir, route_path, iter, route, model, carla_port, tm_port, checkpoint_new_name)
                commands.append(f"chmod u+x {bash_save_dir}/eval_{route}_{model}.sh")
                commands.append(f"./{bash_save_dir}/eval_{route}_{model}.sh")
                commands.append('sleep 2')
                iter += 1

                job_file = make_jobsub_file(commands=commands, job_number=job_nr, exp_name=exp_name)
                result_file = f"{results_save_dir}/{route}.json"

                # HACK: Wait until submitting new jobs that the #jobs are at below max
                num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=exp_name)
                print(f"{num_running_jobs}/{max_num_parallel_jobs} jobs are running...")
                while num_running_jobs >= max_num_parallel_jobs:
                    num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=exp_name)
                    time.sleep(5)
                time.sleep(2)
                print(f"Submitting job {job_nr}/{len(route_files)}: {job_file}")
                #os.system(f"sbatch {job_file}")
                jobid = subprocess.check_output(f"sbatch {job_file}", shell=True).decode("utf-8").strip().split(' ')[-1]
                meta_jobs[jobid] = (False, job_file, result_file)

                job_nr += 1
                #exit(-1)

        training_finished = False
        while not training_finished:
            num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=exp_name)
            print(f"{num_running_jobs} jobs are running...")
            time.sleep(10)

            # resubmit unfinished jobs
            for k in list(meta_jobs.keys()):
                job_finished, job_file, result_file = meta_jobs[k]
                need_to_resubmit = False
                if not job_finished:
                    # check whether job is runing
                    if int(subprocess.check_output(f'squeue | grep {k} | wc -l', shell=True).decode("utf-8").strip()) == 0:
                         # check whether result file is finished?
                         if os.path.exists(result_file):
                             progress = json.load(open(result_file))['_checkpoint']['progress']
                             if len(progress) < 2 or progress[0] < progress[1]:
                                 # resubmit
                                 need_to_resubmit = True
                             else:
                                 # delete old job
                                 meta_jobs[k] = (True, None, None)
                         else:
                             need_to_resubmit = True

                if need_to_resubmit:
                    print(f"resubmit sbatch {job_file}")
                    jobid = subprocess.check_output(f"sbatch {job_file}", shell=True).decode("utf-8").strip().split(' ')[-1]
                    meta_jobs[jobid] = (False, job_file, result_file)
                    meta_jobs[k] = (True, None, None)

            time.sleep(10)

            if num_running_jobs == 0:
                training_finished = True

