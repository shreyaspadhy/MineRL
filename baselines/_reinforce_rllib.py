import gym
import ray
from ray.rllib.agents import ppo
from gym.spaces import Discrete, Box
from env_wrappers import *
import tensorflow as tf
import os
import minerl
from utils import DotDict
# Template of custom environment


# class MyEnv(gym.Env):
#     def __init__(self, env_config):
#         self.action_space = <gym.Space >
#         self.observation_space = <gym.Space >

#     def reset(self):
#         return < obs >

#     def step(self, action):
#         return < obs >, < reward: float > , < done: bool > , < info: dict >


# # Wrapping the MineRL Environment
# class MineRLEnv(gym.Env):
#     def __init__(self, env_config):
#         """MineRL Environment for use

#         env_config is a dict containing all parameters to pass thru trainer
#         """
#         # Is there a way to access the obs and action space without making the
#         # env, needing MC to start up and take hella time
#         self.env_name = env_config['name']
#         self.environ = gym.make(self.env_name)

#         if self.env_name == 'MineRLNavigateDense-v0' or self.env_name == 'MineRLNavigate-v0':
#             self.action_space = Discrete(4)

#         self.observation_space = self.environ.observation_space

#     def reset(self):
#         return self.environ.reset()

#     def step(self, action):
#         obs, rew, done, info = self.environ.step(action)

#         return obs, rew, done, info

#     def mine_to_gym_act(self, mine_a):
#         """
#             Takes in mine_a, ordered dict of minerl format actions
#         """

#     def gym_to_mine_act(self, gym_a):
#         """
#             Takes in gym_a, a Discrete space
#         """

def main():

    args = {}
    args['logging_level'] = 20
    args['monitor'] = True
    args['seed'] = 42
    args['outdir'] = './output'
    args['env'] = 'MineRLNavigateDense-v0'

    os.makedirs(args['outdir'], exist_ok=True)

    # log_format = '%(levelname)-8s - %(asctime)s - [%(name)s %(funcName)s %(lineno)d] %(message)s'
    # logging.basicConfig(filename=os.path.join(args.outdir, 'log.txt'),
    #                     format=log_format, level=args.logging_level)

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(args.logging_level)
    # console_handler.setFormatter(logging.Formatter(log_format))
    # logging.getLogger('').addHandler(console_handler)  # add hander to the root logger

    # logger.info('Output files are saved in {}'.format(args.outdir))

    # utils.log_versions()

    # try:
    _main(args)
    # except:  # noqa
    #     logger.exception('execution failed.')
    #     raise


def _main(args):
    # logger.info('The first `gym.make(MineRL*)` may take several minutes. Be patient!')

    # os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = args.outdir

    # Set a random seed used in ChainerRL.

    # Set different random seeds for train and test envs.
    train_seed = args['seed']  # noqa: never used in this script
    test_seed = 2 ** 31 - 1 - args['seed']

    # wrap_env converts MineRL dict space to discrete space
    def wrap_env(env, test):
        raise Exception('Reimplemented inside WrappedEnv')
        # I believe this is because MineRL's environments do something weird
        # with time_limit
        # if isinstance(env, gym.wrappers.TimeLimit):
        #     logger.info('Detected `gym.wrappers.TimeLimit`! Unwrap it and re-wrap our own time limit.')
        #     env = env.env
        #     max_episode_steps = env.spec.max_episode_steps
        #     env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)

        # Changes MineRL reset() return of (obs, info) to (obs) only
        env = ResetTrimInfoWrapper(env)

        # ContinuingTimeLimit is some chainer stuff, can ignore
        if test and args.monitor:
            env = ContinuingTimeLimitMonitor(
                env, os.path.join(args.outdir, 'monitor'),
                mode='evaluation' if test else 'training', video_callable=lambda episode_id: True)

        # if args.frame_skip is not None:
        #     env = FrameSkip(env, skip=args.frame_skip)
        # if args.gray_scale:
        #     env = GrayScaleWrapper(env, dict_space_key='pov')

        # PoVWith...() returns compass direction as an extra channel to image POV
        if args.env.startswith('MineRLNavigate'):
            env = PoVWithCompassAngleWrapper(env)
        else:
            env = ObtainPoVWrapper(env)

        # env = MoveAxisWrapper(env, source=-1, destination=0)  # convert hwc -> chw as Chainer requires.

        # env = ScaledFloatFrame(env) # I believe it rescales image pixels to [0,1]

        # if args.frame_stack is not None and args.frame_stack > 0:
        #     env = FrameStack(env, args.frame_stack, channel_order='chw')

        # if not args.disable_action_prior:
        #     env = SerialDiscreteActionWrapper(
        #         env,
        #         always_keys=args.always_keys,
        #         reverse_keys=args.reverse_keys,
        #         exclude_keys=args.exclude_keys,
        #         exclude_noop=args.exclude_noop)
        # else:

        env = CombineActionWrapper(env)
        env = SerialDiscreteCombineActionWrapper(env)

        env_seed = test_seed if test else train_seed
        # env.seed(int(env_seed))  # TODO: not supported yet
        return env

    # Actually Create the environments
    class WrappedEnv(gym.Env):
        def __init__(self, env_config):
            args = env_config['args']
            test = env_config['test']
            env = gym.make(args['env'])
            # env = wrap_env(core_env, test=False)
            env = ResetTrimInfoWrapper(env)
            if env_config['test'] and args['monitor']:
                env = ContinuingTimeLimitMonitor(
                    env,
                    os.path.join(args['outdir'], 'monitor'),
                    mode='evaluation' if test else 'training',
                    video_callable=lambda episode_id: True
                )
            if args['env'].startswith('MineRLNavigate'):
                env = PoVWithCompassAngleWrapper(env)
            else:
                env = ObtainPoVWrapper(env)
            env = CombineActionWrapper(env)
            env = SerialDiscreteCombineActionWrapper(env)

            env_seed = test_seed if test else train_seed
            # env.seed(int(env_seed))  # TODO: not supported yet
            self._env = env
            self.__getattr__ = self._env.__getattr__
            self.__setattr__ = self._env.__setattr__
            self.__delattr__ = self._env.__delattr__

    ray.init()
    trainer = ppo.PPOTrainer(
        env=WrappedEnv,
        config={
            "env_config": {
                'args': args,
                'test': False,
            },  # config to pass to env class
        })

    while True:
        print(trainer.train())


if __name__ == '__main__':
    main()
