import wandb
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dl_run_path', default='zhejun/ppo_feb/1y2uq1ed')
parser.add_argument('--ckpt_step', default=11833344)
parser.add_argument('--ul_name', default='Roach')
parser.add_argument('--ul_group', default='RL_experts')
parser.add_argument('--save_dir', default='/home/trace01/ckpts')
parser.add_argument('--ul_project', default='trained-models')
args = parser.parse_args()


save_path = Path(args.save_dir) / (args.dl_run_path.split('/')[2])
save_path.mkdir(parents=True, exist_ok=True)


api = wandb.Api()
run = api.run(args.dl_run_path)

run.file("config_agent.yaml").download(replace=True, root=save_path.as_posix())
run.file(f"ckpt/ckpt_{args.ckpt_step}.pth").download(replace=True, root=save_path.as_posix())

# ckpt_path = save_path/'ckpt'/ f'ckpt_{args.ckpt_step}.pth'
# ckpt_path.rename(save_path/'ckpt/ckpt_0.pth')


wandb.init(project=args.ul_project, name=args.ul_name,group=args.ul_group,save_code=True,entity='iccv21-roach')

wandb.save(save_path.as_posix() + '/config_agent.yaml', base_path=save_path.as_posix())
wandb.save(save_path.as_posix() + f"/ckpt/ckpt_{args.ckpt_step}.pth", base_path=save_path.as_posix())