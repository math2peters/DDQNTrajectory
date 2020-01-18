import utils
import numpy as np

class TrajectoryEnv:

    def __init__(self, agent_xyz_init, agent_acc, tgt_xyz_init, tgt_traj, max_time, time_reward_factor=0.01):
        """
        Initialize the trajectory environment
        :param agent_xyz_init: the initial agent position
        :param agent_acc: the acceleration available to the agent
        :param tgt_xyz_init: the target initial position
        :param tgt_traj: function defining the target trajectory parametrically vs. time
        :param max_time: the maximum trajectory time
        """

        # define the initial separation between target/agent and how close they need to be to terminate the env
        self.init_sep = utils.calc_distance(agent_xyz_init, tgt_xyz_init)
        self.done_sep_threshold = 0.001

        # set the time equal to 0, identify max time, and define the time step, which works out to 50 Hz if the time
        # is in seconds
        self.time = 0
        self.time_step = 0.02
        self.max_time = max_time

        # the time factor is a weight that determines how much the agent cares about time progressing
        # Higher scores <-> lower times
        self.time_reward_factor = time_reward_factor

        # the agent characteristics
        self.agent_xyz = agent_xyz_init
        self.agent_vxyz = np.zeros(3)
        self.agent_axyz = np.zeros(3)
        self.agent_acc = agent_acc

        # the target characteristics
        self.tgt_xyz = tgt_xyz_init
        self.tgt_xyz_next = self.tgt_position_calc(self.time_step)
        self.tgt_traj = tgt_traj

    def tgt_position_calc(self):
        return self.tgt_traj(self.time)

    def agent_position_update(self, thrust_vector):
        """
        update the agent position given a thrust vector
        :param thrust_vector: a vector which determines the direction of the agent acceleration
        """
        # calculate acceleration
        self.agent_axyz = thrust_vector * self.agent_acc
        # update the velocity with the AVERAGE (hence the 0.5) value of acceleration of the timestep
        self.agent_vxyz += 0.5 * self.agent_axyz * self.time_step
        # move the agent forward. This is analagous to x(t) = x0 + 0.5 * a(t)^2
        self.agent_xyz += self.agent_vxyz * self.time_step
        return

    def calc_reward(self):
        """
        Calculate the agent score, which is a linear combination of the distance between the agent and the target
        and the time elapsed since the beginning of the simulation
        :return: the reward
        """
        r1 = - utils.calc_distance(self.agent_xyz, self.tgt_xyz)
        r2 = - self.time * self.time_reward_factor
        return r1 + r2

    def step(self, thrust_vector):
        """
        Move the agent forward one step forward in time
        :param thrust_vector: the direction the agent moves
        :return: the current state (the acceleration, agent position, target position, and next target position),
        the reward the agent receives for its action, and whether or not the simulation is finished
        """
        # get the unit vector representation of the thrust vector
        thrust_vector = utils.norm_vector(thrust_vector)

        # increment time by one step
        self.time += self.time_step

        # update the target position
        self.tgt_xyz = self.tgt_xyz_next
        self.tgt_xyz_next = self.tgt_position_calc()

        # update the agent position
        self.agent_position_update(thrust_vector)

        # calculate the reward
        reward = self.calc_reward()

        # determine if the simulation is finished or not. It finishes if the distance between the two is less
        # than the threshold or if the time is greater than the maximum time
        if utils.calc_distance(self.agent_xyz, self.tgt_xyz) < self.done_sep_threshold * self.init_sep \
                or self.time > self.max_time:
            done = True
            reward = 0
        else:
            done = False

        # concatenate the important information to feed into the DQN for the next prediction
        cur_state = np.concatenate(self.agent_acc, self.agent_xyz, self.tgt_xyz, self.tgt_xyz_next)

        return cur_state, reward, done





