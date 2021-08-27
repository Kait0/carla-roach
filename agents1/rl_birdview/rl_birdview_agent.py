import logging
import numpy as np
from omegaconf import OmegaConf
import wandb
import copy

from carla_gym.utils.config_utils import load_entry_point

from carla_gym.core.obs_manager.obs_manager_handler import ObsManagerHandler
from carla_gym.core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from carla_gym.core.task_actor.common.task_vehicle import TaskVehicle
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.scenarios.route_scenario import convert_transform_to_location
from planner import RoutePlanner



def get_entry_point():
    return 'RlBirdviewAgent'

class RlBirdviewAgent(AutonomousAgent):
    def __init__(self, path_to_conf_file='config_agent.yaml'):
        self._logger = logging.getLogger(__name__)
        self._render_dict = None
        self.supervision_dict = None
        #self.setup(path_to_conf_file)
        super().__init__(path_to_conf_file)

    def sensors(self):
        return []

    def setup(self, path_to_conf_file):
        self.cfg = OmegaConf.load(path_to_conf_file)

        #TODO wb_run is none so we need to create the policy.
        # load checkpoint from wandb
        if self.cfg.wb_run_path is not None:
            api = wandb.Api()
            run = api.run(self.cfg.wb_run_path)
            all_ckpts = [f for f in run.files() if 'ckpt' in f.name]

            if self.cfg.wb_ckpt_step is None:
                f = max(all_ckpts, key=lambda x: int(x.name.split('_')[1].split('.')[0]))
                self._logger.info(f'Resume checkpoint latest {f.name}')
            else:
                wb_ckpt_step = int(self.cfg.wb_ckpt_step)
                f = min(all_ckpts, key=lambda x: abs(int(x.name.split('_')[1].split('.')[0]) - wb_ckpt_step))
                self._logger.info(f'Resume checkpoint closest to step {wb_ckpt_step}: {f.name}')

            f.download(replace=True)
            run.file('config_agent.yaml').download(replace=True)
            self.cfg = OmegaConf.load('config_agent.yaml')
            self._ckpt = f.name
        else:
            self._ckpt = None

        self.cfg = OmegaConf.to_container(self.cfg)

        self._obs_configs = self.cfg['obs_configs']
        self._train_cfg = self.cfg['training']

        # prepare policy
        self._policy_class = load_entry_point(self.cfg['policy']['entry_point'])
        self._policy_kwargs = self.cfg['policy']['kwargs']
        if self._ckpt is None:
            self._policy = None
        else:
            self._logger.info(f'Loading wandb checkpoint: {self._ckpt}')
            self._policy, self._train_cfg['kwargs'] = self._policy_class.load(self._ckpt)
            self._policy = self._policy.eval()

        self._wrapper_class = load_entry_point(self.cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = self.cfg['env_wrapper']['kwargs']

        self._om_handler = ObsManagerHandler({0:self._obs_configs})
        self.initialized = False

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        # set route to be followed
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        # route in the form of locations, as required by the outside route lane tester
        self.route = convert_transform_to_location(global_plan_world_coord)

        # route in the form of transforms, as required by the route planners
        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def local_init(self):
        self._route_planner = RoutePlanner(3.0, 50.0)
        self._route_planner.set_route(self._plan_gps_HACK, True)

        self._client = CarlaDataProvider.get_client()

        self._ev_handler = EgoVehicleHandler(self._client, None, None)
        self._vehicle = CarlaDataProvider.get_hero_actor()

        self._spawn_transforms = self._ev_handler._get_spawn_points(self._ev_handler._map)

        #TODO add waypoints as second parameter
        self._ev_handler.ego_vehicles[0] = TaskVehicle(self._vehicle, [], self._spawn_transforms, False)

        #ev_spawn_locations = self._ev_handler.reset({0:test})

        self._om_handler.reset(self._ev_handler.ego_vehicles)
        self.initialized = True


    def run_step(self, input_data, timestamp):
        if(self.initialized == False):
            self.local_init()

        obs_dict = self._om_handler.get_observation(timestamp)
        input_data = obs_dict[0]

        input_data = copy.deepcopy(input_data)

        policy_input = self._wrapper_class.process_obs(input_data, self._wrapper_kwargs['input_states'], train=False)

        actions, values, log_probs, mu, sigma, features = self._policy.forward(
            policy_input, deterministic=True, clip_action=True)
        control = self._wrapper_class.process_act(actions, self._wrapper_kwargs['acc_as_action'], train=False)
        self.supervision_dict = {
            'action': np.array([control.throttle, control.steer, control.brake], dtype=np.float32),
            'value': values[0],
            'action_mu': mu[0],
            'action_sigma': sigma[0],
            'features': features[0],
            'speed': input_data['speed']['forward_speed']
        }
        self.supervision_dict = copy.deepcopy(self.supervision_dict)

        self._render_dict = {
            'timestamp': timestamp,
            'obs': policy_input,
            'im_render': input_data['birdview']['rendered'],
            'action': actions,
            'action_value': values[0],
            'action_log_probs': log_probs[0],
            'action_mu': mu[0],
            'action_sigma': sigma[0]
        }
        self._render_dict = copy.deepcopy(self._render_dict)

        return control

    def reset(self, log_file_path):
        # logger
        self._logger.handlers = []
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)

    def learn(self, env, total_timesteps, callback, seed):
        if self._policy is None:
            self._policy = self._policy_class(env.observation_space, env.action_space, **self._policy_kwargs)

        # init ppo model
        model_class = load_entry_point(self._train_cfg['entry_point'])
        model = model_class(self._policy, env, **self._train_cfg['kwargs'])
        model.learn(total_timesteps, callback=callback, seed=seed)

    def render(self, reward_debug, terminal_debug):
        '''
        test render, used in benchmark.py
        '''
        self._render_dict['reward_debug'] = reward_debug
        self._render_dict['terminal_debug'] = terminal_debug

        return self._wrapper_class.im_render(self._render_dict)

    @property
    def obs_configs(self):
        return self._obs_configs

    def destroy(self):
        pass
