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
from leaderboard.autoagents import autonomous_agent
from leaderboard.autoagents import autonomous_agent_local
from carla_gym.utils.traffic_light import TrafficLightHandler
import gym
import os




def get_entry_point():
    return 'RlBirdviewAgent'

class RlBirdviewAgent(autonomous_agent_local.AutonomousAgent):
    def __init__(self, path_to_conf_file='config_agent.yaml', route_index=None):
        self._logger = logging.getLogger(__name__)
        self._render_dict = None
        self.supervision_dict = None
        #self.setup(path_to_conf_file)
        super().__init__(path_to_conf_file)

    def sensors(self):
        return []

    def setup(self, path_to_conf_file, route_index=None):
        self.cfg = OmegaConf.load(path_to_conf_file)
        self.track = autonomous_agent.Track.SENSORS

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
        self._acc_as_action = self._wrapper_kwargs['acc_as_action']
        self.initialized = False

    def local_init(self):
        self._client = CarlaDataProvider.get_client()

        self._ev_handler = EgoVehicleHandler(self._client, None, None)
        self._vehicle = CarlaDataProvider.get_hero_actor()

        self._spawn_transforms = self._ev_handler._get_spawn_points(self._ev_handler._map)

        self._world = CarlaDataProvider.get_world()
        # register traffic lights
        TrafficLightHandler.reset(self._world)

        self.route = CarlaDataProvider.get_original_trajectory()
        self._ev_handler.ego_vehicles[0] = TaskVehicle(self._vehicle, self.route, self._spawn_transforms, False)

        #ev_spawn_locations = self._ev_handler.reset({0:test})

        self._om_handler.reset(self._ev_handler.ego_vehicles)

        state_spaces = []
        if 'speed' in self.cfg['obs_configs']:
            state_spaces.append(self._om_handler.observation_space[0]['speed']['speed_xy'])
        if 'speed_limit' in self.cfg['obs_configs']:
            state_spaces.append(self._om_handler.observation_space[0]['control']['speed_limit'])
        if 'control' in self.cfg['obs_configs']:
            state_spaces.append(self._om_handler.observation_space[0]['control']['throttle'])
            state_spaces.append(self._om_handler.observation_space[0]['control']['steer'])
            state_spaces.append(self._om_handler.observation_space[0]['control']['brake'])
            state_spaces.append(self._om_handler.observation_space[0]['control']['gear'])
        if 'acc_xy' in self.cfg['obs_configs']:
            state_spaces.append(self._om_handler.observation_space[0]['velocity']['acc_xy'])
        if 'vel_xy' in self.cfg['obs_configs']:
            state_spaces.append(self._om_handler.observation_space[0]['velocity']['vel_xy'])
        if 'vel_ang_z' in self.cfg['obs_configs']:
            state_spaces.append(self._om_handler.observation_space[0]['velocity']['vel_ang_z'])

        state_low = np.concatenate([s.low for s in state_spaces])
        state_high = np.concatenate([s.high for s in state_spaces])

        self.observation_space = gym.spaces.Dict(
            {'state': gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
             'birdview': self._om_handler.observation_space[0]['birdview']['masks']})

        if self._acc_as_action:
            # act: acc(throttle/brake), steer
            self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            # act: throttle, steer, brake
            self.action_space = gym.spaces.Box(low=np.array([0, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        
        if self._policy is None:
            self._policy = self._policy_class(self.observation_space, self.action_space, **self._policy_kwargs).to('cuda')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._policy,self._train_cfg['kwargs'] = self._policy.load(dir_path + r'/ckpt_11833344.pth')
        self._policy = self._policy.eval()
        
        self.initialized = True


    def run_step(self, input_data, timestamp):
        if(self.initialized == False):
            self.local_init()

        _ = self._ev_handler.ego_vehicles[0]._truncate_global_route_till_local_target()

        obs_dict = self._om_handler.get_observation(timestamp)
        input_data = obs_dict[0]

        input_data = copy.deepcopy(input_data)

        #Debug
        #from matplotlib import pyplot as plt
        #plt.ion()
        #plt.imshow(input_data['birdview']['rendered'])
        #plt.show()

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
