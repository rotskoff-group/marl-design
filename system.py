from openmm import *
from openmm.app import *
from openmm.unit import *

from mdtraj.reporters import HDF5Reporter
import freud
import numpy as np
from scipy.stats import gamma, binom, norm
import matplotlib.pyplot as plt

class MolecularSystem:
    def __init__(self, filename, region_num, surrounding_grids_type,
                 num_particles, dt, dim_length, target_dist):
        self.num_particles = num_particles
        self.filename = filename
        self.dimensions = 2
        self.dt = dt
        self.invdt = int(1 / self.dt)
        self.dim_length = dim_length
        self.region_num = region_num
        self.region_int = np.linspace(0, self.dim_length, self.region_num + 1)
        self.region_action = np.zeros((self.region_num, self.region_num))
        self.surrounding = surrounding_grids_type
        self.target_dist = target_dist
        self.q = self.bin = None

    def _surrounding_grid_indices(self):
        """Returns the indices of surrounding regions
        Raises:
            ValueError: If surrounding region is larger than system or if
            incorrect description provided
        """
        if (self.surrounding == "plaquette"):
            if (self.region_num < 3):
                raise ValueError("Need more regions")
            return [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        elif (isinstance(self.surrounding, int) and 1 <= self.surrounding < (self.region_num // 2)):
            a = [[(i, j) for i in range(-self.surrounding, self.surrounding + 1, 1)]
                 for j in range(-self.surrounding, self.surrounding + 1, 1)]
            a = builtins.sum(a, [])
            return a
        elif (self.surrounding >= (self.region_num // 2)):
            raise ValueError("Inputted too many surrounding grids")
        else:
            raise ValueError("Incorrect Description provided")

    def _init_target_distribution(self, dist):
        raise NotImplementedError()

    def plot_target_distribution(self, dist):
        raise NotImplementedError()

    def _init_system(self):
        raise NotImplementedError()

    def _init_integrator(self):
        raise NotImplementedError()

    def _init_simulation(self):
        raise NotImplementedError()

    def _init_position(self):
        """Initializes positions on a lattice

        Returns:
            Array of particle positions.
        """
        num_per_dim = round(((self.num_particles)**(1 / self.dimensions))
                            + 0.5)
        lattice_spacing = self.dim_length / num_per_dim
        particle_position = self.num_particles * [0]
        for i in range(self.num_particles):
            x = i % num_per_dim
            y = i // num_per_dim
            x_pos = lattice_spacing * (x + 0.5 * (y % 2))
            y_pos = lattice_spacing * y
            particle_position[i] = Vec3(x_pos, y_pos, 0)

        return particle_position

    def _get_region_action(self, particle_pos):
        """For a given particle position returns activity of the region that
           particle is in

        Returns:
            Action of region particle is in
        """
        # Get region indices of x and y
        x_in = np.sum([self.region_int < particle_pos[0]]) - 1
        y_in = np.sum([self.region_int > particle_pos[1]]) - 1
        return self.region_action[y_in, x_in]

    def update_action(self, new_action):
        """Updates region_action to be new_action
        Args:
            new_action: 1D (flattened) array of actions of regions
        """
        if (not len(new_action) == (self.region_num ** 2)):
            raise ValueError("Incorrect Action Length")
        self.region_action = np.array(new_action).reshape((self.region_num,
                                                           self.region_num))

    def _update_regions(self):
        raise NotImplementedError()

    def _get_KL(self, p):
        """Calculates KL Div from target_distribution to p
        Args:
            p: A normalized distribution of cluster sizes
        Returns:
            KL divergence from target_distribution to p or None if p is None
        Raises:
            ValueError: If q does not have full support over sample space
        """

        if p is None:
            return None
        sum = 0
        ss_len = len(self.q[0])
        for i in range(ss_len):
            p_i = p[0][i] * (p[1][i + 1] - p[1][i])
            q_i = self.q[0][i] * (self.q[1][i + 1] - self.q[1][i])
            try:
                if (p_i == 0):
                    continue
                sum += p_i * np.log(p_i / q_i)
            except:
                raise ValueError("Define q with support over sample space")
        return sum

    def _duplicate_element_by_val(self, count):
        """Duplicates elements by current value. Use to get number of particles per cluster
        E.g. Given an input of [1, 2, 3] it will return [1, 2, 2, 3, 3, 3]
        Args:
            count: A List of all cluster sizes
        Returns:
            A List of the cluster size that each particle belongs to
            or empty list if the input list is empty (i.e. no clusters present)
        """
        dup_count = []
        for val in count:
            dup_count += [val] * val
        # Return empty list for empty region
        return dup_count

    def _get_all_cluster_sizes(self):
        """Gets all cluster size within each region
        Returns:
            cs_region: A 2D List of all cluster sizes in each region
        """

        cl = freud.cluster.Cluster()
        box = freud.box.Box.square(L=self.dim_length)
        positions = self.simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True).getPositions()
        positions = [list(x) for x in positions._value]  # Convert to 2D list
        cl.compute((box, positions), neighbors={'r_max': 1.25})  # In nm
        index, counts = np.unique(cl.cluster_idx, return_counts=True)

        cs_region = [[[] for i in range(self.region_num)]
                     for j in range(self.region_num)]
        for p_i in range(self.num_particles):
            particle_pos = positions[p_i]
            x_in = np.sum([self.region_int < particle_pos[0]]) - 1
            y_in = np.sum([self.region_int > particle_pos[1]]) - 1
            current_cluster_index = cl.cluster_idx[p_i]
            # Get all the cluster indices in each region
            if current_cluster_index not in cs_region[y_in][x_in]:
                cs_region[y_in][x_in].append(current_cluster_index)

        # Get all the unique cluster sizes in each region
        cs_region = [[counts[cs_region[i][j]]
                     for j in range(self.region_num)]
                     for i in range(self.region_num)]

        # Get all the particles in a cluster sizes in each region
        cs_region = [[self._duplicate_element_by_val(cs_region[i][j])
                      for j in range(self.region_num)]
                     for i in range(self.region_num)]
        return cs_region

    def _get_cluster_distribution_all(self, tag=None):
        """Gets the cluster distribution of the entire system (not individual grids)
        Args:
            tag: A string describing the end of the filename
        Returns:
            p: 2D list of normalized distribution of cluster sizes in the entire system
            counts: A List of all cluster sizes in the entire system
        Saves:
            Image of distribution and cluster distribution if tag is not None
        """

        cl = freud.cluster.Cluster()
        box = freud.box.Box.square(L=self.dim_length)
        positions = self.simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True).getPositions()
        positions = [list(x) for x in positions._value]  # Convert to 2D list
        cl.compute((box, positions), neighbors={'r_max': 1.25})  # In nm
        index, counts = np.unique(cl.cluster_idx, return_counts=True)
        counts = self._duplicate_element_by_val(counts)
        p = plt.hist(counts, bins=self.bin +
                     [max(np.max(counts), self.bin[-1] + 1)], density=True)
        if (tag is not None):
            self.plot_target_distribution(dist=self.target_dist)
            filename = self.filename[:-3] + tag + ".png"
            plt.savefig(filename)
        plt.close()
        return p, counts

    def _get_cluster_distribution(self):
        """Gets the cluster distribution of the current grid and surrounding grids
        Returns:
            p: A 2D list of histograms of cluster sizes in each grid
            surrounding: A 2D list of histograms of cluster sizes in surrounding system
            cs_region: A 3D List of all cluster sizes in each region
            surrounding_clusters: A 3D List of all cluster sizes in each surrounding region
        """
        cs_region = self._get_all_cluster_sizes()
        # GET CURRENT REGION
        dist0 = np.histogram([], bins=self.bin + [self.bin[-1] + 1])

        p = [[dist0 if (len(cs_region[i][j]) == 0)
              else np.histogram(cs_region[i][j], bins=self.bin + [max(max(cs_region[i][j]), self.bin[-1] + 1)], density=True)
              for j in range(self.region_num)]
             for i in range(self.region_num)]

        surrounding_clusters = []
        # GET SURROUNDING REGIONS
        if (self.surrounding == None):
            dist = np.histogram([], bins=self.bin + [self.bin[-1] + 1])
            surrounding = [[dist
                            for j in range(self.region_num)]
                           for i in range(self.region_num)]
        elif (self.surrounding == "all"):
            p_all, _ = self._get_cluster_distribution_all()
            surrounding = [[p_all for j in range(self.region_num)]
                           for i in range(self.region_num)]
        else:
            all_cluster_sizes = np.array(cs_region, dtype="object")
            indices = np.array(self._surrounding_grid_indices())
            dist0 = np.histogram([], bins=self.bin + [self.bin[-1] + 1])

            # This will double count clusters that span multiple regions
            surrounding_clusters = [[all_cluster_sizes[tuple(((indices + [i, j]) % self.region_num).T.tolist())].sum()
                                     for j in range(self.region_num)]
                                    for i in range(self.region_num)]
            surrounding = [[dist0 if (len(surrounding_clusters[i][j]) == 0)
                            else np.histogram(surrounding_clusters[i][j], bins=self.bin + [max(max(surrounding_clusters[i][j]), self.bin[-1] + 1)], density=True)
                            for j in range(self.region_num)]
                           for i in range(self.region_num)]

        return p, surrounding, cs_region, surrounding_clusters

    def get_state_reward(self):
        """Returns the current state, reward, and list of cluster sizes of each region and surrounding regions
        Returns:
            dist: 2D list of normalized distribution of cluster sizes in each grid
            surrounding_dist: A 2D list of normalized distribution of cluster sizes in each surrounding region
            reward: A 2D list of the KL divergence in each region
            surrounding_reward: A 2D list of the KL divergence in each surrounding region
            cs_region: A 3D List of all cluster sizes in each region
            surrounding_clusters: A 3D List of all cluster sizes in each surrounding region
        """
        if self.q is None or self.bin is None:
            raise ValueError("Define q and/or bin")
        p, surrounding, cs_region, surrounding_clusters = self._get_cluster_distribution()

        reward = []
        surrounding_reward = []
        dist = []
        surrounding_dist = []
        for i in range(self.region_num):
            for j in range(self.region_num):
                reward.append(self._get_KL(p[i][j]))
                surrounding_reward.append(self._get_KL(surrounding[i][j]))

                curr_dist = p[i][j][0] * np.diff(p[i][j][1])
                dist.append(curr_dist.tolist())

                curr_s_dist = surrounding[i][j][0] * \
                    np.diff(surrounding[i][j][1])
                surrounding_dist.append(curr_s_dist.tolist())
        return [dist, surrounding_dist, reward, surrounding_reward, cs_region, surrounding_clusters]

    def get_state_reward_all(self, tag):
        """Returns the current state, reward, and list of the entire system
        Args:
            tag: A string describing the end of the filename
        Returns:
            dist: list of normalized distribution of cluster sizes in the entire system
            reward: KL divergence of entire system
            cs_region: A List of all cluster sizes in entire system
        """
        if self.q is None or self.bin is None:
            raise ValueError("Define q and/or bin")
        p, counts = self._get_cluster_distribution_all(tag)
        reward = self._get_KL(p)
        dist = p[0] * np.diff(p[1])
        state = dist.tolist()
        return [state, reward, counts]

    def _run_sim(self, time):
        """Runs a simulation for time seconds
        Args:
            time: number of seconds to run simulation
        """
        total_sim_time = int(time * self.invdt)
        self.simulation.step(total_sim_time)

    def run_step(self, is_detailed, tag):
        raise NotImplementedError()

    def run_decorrelation(self, time):
        raise NotImplementedError()

    def reset_context(self, filename):
        raise NotImplementedError()

    def get_dissipation(self):
        raise NotImplementedError()


class ActiveSystem(MolecularSystem):
    def __init__(self, filename, region_num=48, target_dist="default_gaussian",
                 surrounding_grids_type=None):
        super().__init__(filename=filename, region_num=region_num,
                         num_particles=5000, dt=0.00005,
                         surrounding_grids_type=surrounding_grids_type,
                         dim_length=(134 * 1.41), target_dist=target_dist)
        self.active_mag_def = 2
        self.D_t = self.active_mag_def * 0.0625 / 3
        self.A = 0.87
        self.bin, self.q = self._init_target_distribution(
            dist=self.target_dist)
        self.num_bins = len(self.bin)
        self.system = self._init_system()
        self.integrator = self._init_integrator()
        self.simulation = self._init_simulation()

    def _init_target_distribution(self, dist="default_gaussian"):
        """Initializes the target distribution

        Args:
            dist: The name of the target distribution
        Returns:
            bin: The positions of the endpoints of each bin. Width of each bin
                 is used to calculate probability
            q: The height of each bin
        Raises:
            ValueError: If inputted distribution is not found
        """
        if (dist == "default_gaussian"):
            bin = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                   22, 24, 26, 28, 30, 32, 34, 36, 38, 39, 100]
            q_0 = np.diff(norm.cdf(bin, 20, 3))
            # Because gaussian distribution is symmetric and we want to bin (-inf, 2) into 1
            q_0[0] += q_0[-1]
            # Just create a vector whose differences will be 1
            # This is important for calculating KL, where q_1 is bin_width
            # and q_0 is bin height. Here the bin_height is already a probability
            # so create an artificial bin_width of 1
            # Remove the last element of bin so its consistent with the
            # approaches done below with plt.hist()
            bin = bin[:-1]
            q_1 = np.arange(len(q_0) + 1)
            q = (q_0, q_1)

        elif (dist == "default_binom"):
            n, p = 20, 15 / 20
            bin = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20, 100]
            q_0 = np.diff(binom.cdf(bin, n, p))
            q_0[-1] = 1 - np.sum(q_0)
            # Just create a vector whose differences will be 1
            # This is important for calculating KL, where q_1 is bin_width
            # and q_0 is bin height. Here the bin_height is already a probability
            # so create an artificial bin_width of 1
            # Remove the last element of bin so its consistent with the
            # approaches done below with plt.hist()
            bin = bin[:-1]
            q_1 = np.arange(len(q_0) + 1)
            q = (q_0, q_1)

        elif (dist == "default_gamma"):
            bin = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18,
                   20, 22, 24, 26, 28, 30, 50]  # Regular
            q = plt.hist(np.random.gamma(25, 3 / 5, 100000000),
                         bins=(bin + [100]), density=True)
            plt.close()
            q_hist = q[0]
            q_hist[0] = 3.11E-14
            q_hist[-1] = 3.644E-16
            if (q_hist[2] == 0):
                q_hist[2] = 1.5E-8
            q = (q_hist, q[1])
        else:
            raise ValueError("Dist supplied not defined")
        return bin, q

    def plot_target_distribution(self, dist="default_gaussian"):
        """
        Plots target distribution
        Args:
            dist: The name of the target distribution
        Raises:
            ValueError: If inputted distribution is not found
        """
        if (dist == "default_gaussian"):
            plt.plot(np.linspace(0, 100, 500), norm.pdf(
                np.linspace(0, 100, 500), loc=20, scale=3))
        elif (dist == "default_binom" or dist == "old_binom"):
            plt.plot(self.bin, self.q[0], 'bo')

        elif (dist == "default_gamma"):
            plt.plot(np.linspace(0, 100, 500), gamma.pdf(
                np.linspace(0, 100, 500), a=25, scale=3 / 5))

        else:
            raise ValueError("Dist supplied not defined")

    def _init_system(self):
        """Initializes an OpenMM system

        Returns:
            Initialized OpenMM System
        """

        a = Quantity((self.dim_length * nanometer,
                     0 * nanometer, 0 * nanometer))
        b = Quantity((0 * nanometer, self.dim_length *
                     nanometer, 0 * nanometer))
        c = Quantity((0 * nanometer, 0 * nanometer,
                     self.dim_length * nanometer))
        system = System()
        system.setDefaultPeriodicBoxVectors(a, b, c)

        sigma = 1 * nanometer
        # Adding 1 to integrator to represent shift for WCA
        epsilon = 1 * kilojoule_per_mole
        cutoff_type = NonbondedForce.CutoffPeriodic

        wca = CustomNonbondedForce(
            "step((2^(1/6) * sigma)-r)* (4*epsilon*((sigma/r)^12-(sigma/r)^6) + epsilon)")

        wca.addGlobalParameter("sigma", sigma)
        wca.addGlobalParameter("epsilon", epsilon)
        wca.setCutoffDistance(3 * sigma)
        wca.setNonbondedMethod(cutoff_type)

        A_val = (self.active_mag_def ** 2) * self.A * elementary_charge

        attractive_force = CustomNonbondedForce("A*((1/r)^2); A=-sqrt(A1*A2)")
        attractive_force.addPerParticleParameter("A")

        attractive_force.setCutoffDistance(3 * sigma)
        attractive_force.setNonbondedMethod(cutoff_type)

        for particle_index in range(self.num_particles):
            system.addParticle(2 * amu)
            wca.addParticle()
            attractive_force.addParticle()
            attractive_force.setParticleParameters(particle_index, [A_val])

        system.addForce(wca)
        system.addForce(attractive_force)

        return system

    def _init_integrator(self):
        """Initializes an OpenMM integrator

        Returns:
            Initialized OpenMM integrator
        """

        self.D_t = self.active_mag_def * 0.0625 / 3  # Assumes mu=1
        D_r = self.active_mag_def * 0.0625  # Assumes mu=1
        # Intialize Variables
        abp_integrator = CustomIntegrator(self.dt)
        abp_integrator.addGlobalVariable("D_t", self.D_t)
        abp_integrator.addGlobalVariable("box_length", self.dim_length)

        abp_integrator.addGlobalVariable("D_r", D_r)

        abp_integrator.addPerDofVariable("theta", 0)
        abp_integrator.addPerDofVariable("active_mag", self.active_mag_def)
        abp_integrator.addPerDofVariable("active", 0)
        abp_integrator.addPerDofVariable("x_dot", 0)
        abp_integrator.addPerDofVariable("dissipation", 0)
        abp_integrator.addPerDofVariable("total_force", 0)

        vec = []
        for i in range(self.num_particles):
            rand_theta = 2 * np.pi * np.random.random()
            vec.append(Vec3(rand_theta, 0, 0))
        abp_integrator.setPerDofVariableByName("theta", vec)

        # Use x1 to store previous x to calculate dx
        abp_integrator.addComputePerDof("active",
                                        "vector(_x(active_mag) * cos(_x(theta)), \
            _x(active_mag) * sin(_x(theta)), 0)")

        abp_integrator.addComputePerDof("x_dot", "x")
        abp_integrator.addComputePerDof("total_force", "f + active")
        abp_integrator.addComputePerDof("x", "x + dt*(f + active) + \
            gaussian * sqrt(2 * D_t * dt)")
        abp_integrator.addComputePerDof("x", "vector(_x(x), _y(x), 0)")
        abp_integrator.addComputePerDof("x_dot", "x - x_dot")

        abp_integrator.addComputePerDof(
            "x_dot", "x_dot + step(x_dot - 0.5*box_length)*(-0.5*box_length)")
        abp_integrator.addComputePerDof(
            "x_dot", "x_dot + step(-(x_dot + 0.5*box_length))*(0.5*box_length)")

        abp_integrator.addComputePerDof(
            "dissipation", "dissipation + dot(x_dot, total_force)")
        abp_integrator.addComputePerDof("theta", "theta + \
            gaussian * sqrt(2 * D_r * dt)")
        abp_integrator.addUpdateContextState()
        return abp_integrator

    def _init_simulation(self):
        """Initializes an OpenMM Simulation

        Returns:
            Initialized OpenMM Simulation
        """
        topology = Topology()
        element = Element.getBySymbol('H')
        chain = topology.addChain()
        for particle in range(self.num_particles):
            residue = topology.addResidue('abp', chain)
            topology.addAtom('abp', element, residue)
        topology.setUnitCellDimensions(
            Quantity(3 * [self.dim_length], nanometer))
        simulation = Simulation(topology, self.system, self.integrator)
        simulation.context.getPlatform().setPropertyDefaultValue("CudaDeviceIndex", "0")
        simulation.context.setPositions(self._init_position())
        simulation.reporters.append(
            HDF5Reporter(self.filename, self.invdt // 4))
        return simulation

    def _update_regions(self):
        """Updates activity of all particles based on the region it is in
        """
        attractive_force = self.system.getForce(1)
        positions = self.simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True).getPositions()
        all_particle_active = [self._get_region_action(x._value)
                               for x in positions]
        active_mag_vec = [Vec3(particle_i_active, 0, 0)
                          for particle_i_active in all_particle_active]
        charge_vec = [(particle_i_active ** 2) * self.A * elementary_charge
                      for particle_i_active in all_particle_active]
        self.simulation.integrator.setPerDofVariableByName("active_mag",
                                                           active_mag_vec)
        [attractive_force.setParticleParameters(index, [element])
            for(index, element) in enumerate(charge_vec)]
        attractive_force.updateParametersInContext(self.simulation.context)

    def run_decorrelation(self, time):
        """Runs a decorrelation step of zero activity to "decorrelate" from some current state
        Args:
            time: time in seconds to run decorrelation
            tag: A string describing the end of the filename
        """

        new_active_mag = [0.0] * self.region_num**2
        self.update_action(new_action=new_active_mag)
        self._update_regions()
        self._run_sim(time)

    def run_step(self, is_detailed=False, tag=""):
        """Runs simulation for one time "step" (i.e. decision) of RL algorithm
        Updates particle activity every 0.5 seconds based on what region particle
        is in. Runs for a total of 3 seconds (i.e. 1 decision)
        Args:
            is_detailed: Include information about states/rewards of entire system
            tag: A string describing the end of the filename
        Returns:
            A list of states, rewards and cluster sizes of the system if more_detailed
            The states, rewards and cluster sizes of the system if is _detailed
            None, None, None if not (is_detailed)
        """
        assert self.simulation.context.getPlatform().getName() == "CUDA"
        for i in range(6):
            self._update_regions()
            self._run_sim(0.5)
        if (is_detailed):
            curr_tag = tag + "_" + str(i)
            system_state, system_reward, system_cluster_counts = self.get_state_reward_all(
                tag)
            all_system_states = system_state
            all_system_rewards = system_reward
            all_system_states_cluster = system_cluster_counts
            return all_system_states, all_system_rewards, all_system_states_cluster
        else:
            return None, None, None

    def reset_context(self, filename):
        """Resets position to lattice and closes h5 file
        Args:
            filename: file to save new trajectory in
        """

        self.filename = filename
        self.simulation.reporters[0].close()
        self.simulation.reporters[0] = HDF5Reporter(
            self.filename, self.invdt // 4)
        self.simulation.context.setPositions(self._init_position())

    def get_dissipation(self):
        """Gets dissipation of simulation
        Returns:
            Mean total dissipation across all particles
        """
        dissipation = self.simulation.integrator.getPerDofVariableByName(
            "dissipation")
        dissipation = np.array([d_n[0] for d_n in dissipation])
        dissipation /= (self.D_t)
        return np.mean(dissipation)


class LJSystem(MolecularSystem):
    def __init__(self, filename, region_num=15, target_dist="default_gamma",
                 surrounding_grids_type=None):
        super().__init__(filename=filename, region_num=region_num,
                         num_particles=100, dt=0.0002,
                         surrounding_grids_type=surrounding_grids_type,
                         dim_length=30, target_dist=target_dist)

        self.bin, self.q = self._init_target_distribution(
            dist=self.target_dist)
        self.num_bins = len(self.bin)
        self.system = self._init_system()
        self.integrator = self._init_integrator()
        self.simulation = self._init_simulation()

    def _init_target_distribution(self, dist="default_gamma"):
        """Initializes the target distribution

        Args:
            dist: The name of the target distribution
        Returns:
            bin: The positions of the endpoints of each bin. Width of each bin
                 is used to calculate probability
            q: The height of each bin
        Raises:
            ValueError: If inputted distribution is not found
        """
        if (dist == "default_gamma"):
            bin = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Regular
            target_data = np.random.gamma(16, 0.25, 10000000)
            q = plt.hist(target_data, bins=(bin + [100]), density=True)
            plt.close()
        else:
            raise ValueError("Dist supplied not defined")
        return bin, q

    def plot_target_distribution(self, dist="default_gamma"):
        """
        Plots target distribution
        Args:
            dist: The name of the target distribution
        Raises:
            ValueError: If inputted distribution is not found
        """
        if (dist == "default_gamma"):
            plt.plot(np.linspace(0, 10, 500), gamma.pdf(
                np.linspace(0, 10, 500), a=16, scale=0.25))
        else:
            raise ValueError("Dist supplied not defined")

    def _init_system(self):
        """Initializes an OpenMM system

        Returns:
            Initialized OpenMM System
        """

        a = Quantity((self.dim_length * nanometer,
                     0 * nanometer, 0 * nanometer))
        b = Quantity((0 * nanometer, self.dim_length *
                     nanometer, 0 * nanometer))
        c = Quantity((0 * nanometer, 0 * nanometer,
                     self.dim_length * nanometer))
        system = System()
        system.setDefaultPeriodicBoxVectors(a, b, c)

        sigma = 1 * nanometer
        epsilon = 0.5 * kilojoule_per_mole
        cutoff_type = NonbondedForce.CutoffPeriodic

        lj = CustomNonbondedForce("4*epsilon*(((sigma/r)^12-(sigma/r)^6))")
        lj.addGlobalParameter("sigma", sigma)
        lj.addGlobalParameter("epsilon", epsilon)
        lj.setCutoffDistance(15 * sigma)
        lj.setNonbondedMethod(cutoff_type)

        for particle_index in range(self.num_particles):
            system.addParticle(2 * amu)
            lj.addParticle()

        system.addForce(lj)

        return system

    def _init_integrator(self):
        """Initializes an OpenMM Integrator

        Returns:
            Initialized OpenMM Integrator
        """

        lj_integrator = CustomIntegrator(self.dt)
        lj_integrator.addGlobalVariable("box_length", self.dim_length)
        lj_integrator.addPerDofVariable("D_t", 1.2)

        lj_integrator.addPerDofVariable("dissipation", 0)
        lj_integrator.addPerDofVariable("x_dot", 0)
        lj_integrator.addPerDofVariable("total_force", 0)

        lj_integrator.addComputePerDof("x_dot", "x")
        lj_integrator.addComputePerDof("total_force", "f")
        lj_integrator.addComputePerDof("x", "x + dt*(f) + \
            gaussian * sqrt(2 * D_t * dt)")

        lj_integrator.addComputePerDof("x", "vector(_x(x), _y(x), 0)")
        lj_integrator.addComputePerDof("x_dot", "x - x_dot")
        lj_integrator.addComputePerDof(
            "x_dot", "x_dot + step(x_dot - 0.5*box_length)*(-0.5*box_length)")
        lj_integrator.addComputePerDof(
            "x_dot", "x_dot + step(-(x_dot + 0.5*box_length))*(0.5*box_length)")
        lj_integrator.addComputePerDof(
            "dissipation", "dissipation + (dot(x_dot, total_force)/D_t)")

        lj_integrator.addUpdateContextState()
        return lj_integrator

    def _init_simulation(self):
        """Initializes an OpenMM Simulation

        Returns:
            Initialized OpenMM Simulation
        """
        topology = Topology()
        element = Element.getBySymbol('H')
        chain = topology.addChain()
        for particle in range(self.num_particles):
            residue = topology.addResidue('lj', chain)
            topology.addAtom('lj', element, residue)
        topology.setUnitCellDimensions(
            Quantity(3 * [self.dim_length], nanometer))
        simulation = Simulation(topology, self.system, self.integrator)
        # simulation.context.getPlatform().\
        #     setPropertyDefaultValue("CudaDeviceIndex", "0")
        simulation.context.setPositions(self._init_position())
        simulation.reporters.append(
            HDF5Reporter(self.filename, self.invdt // 100))
        return simulation

    def update_action(self, new_action):
        super().update_action(new_action)
        if np.any((self.region_action <= 0) | (self.region_action > 2.0)):
            raise ValueError("Unallowed Temperatures Inputted")

    def _update_regions(self):
        """Updates temperature of all particles based on the region it is in
        """
        positions = self.simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True).getPositions()
        all_particle_temps = [self._get_region_action(x._value)
                              for x in positions]
        temp_vec = [Vec3(particle_i_temp, particle_i_temp, 0)
                    for particle_i_temp in all_particle_temps]

        self.simulation.integrator.setPerDofVariableByName("D_t",
                                                           temp_vec)

    def run_decorrelation(self, time):
        """Runs a decorrelation step of high temperature to "decorrelate" from some current state
        Args:
            time: time in seconds to run decorrelation
            tag: A string describing the end of the filename
        """
        new_temp = [1.2] * self.region_num**2
        self.update_action(new_temp)
        self._update_regions()
        self._run_sim(time)

    def run_step(self, is_detailed=False, tag=""):
        """Runs simulation for one time "step" (i.e. decision) of RL algorithm
        Updates particle activity every 0.25 seconds based on what region particle
        is in. Runs for a total of 0.25 seconds (i.e. 1 decision)
        Args:
            is_detailed: Include information about states/rewards of entire system
            tag: A string describing the end of the filename
        Returns:
            The states, rewards and cluster sizes of the system if is _detailed
            None, None, None if not (is_detailed)
        """
        for i in range(1):
            # Updating once every second
            self._update_regions()
            self._run_sim(0.25)
        if (is_detailed):
            curr_tag = tag + "_" + str(i)
            system_state, system_reward, system_cluster_counts = self.get_state_reward_all(
                tag)
            all_system_states = system_state
            all_system_rewards = system_reward
            all_system_states_cluster = system_cluster_counts
        if (is_detailed):
            return all_system_states, all_system_rewards, all_system_states_cluster
        else:
            return None, None, None

    def reset_context(self, filename):
        """Resets position to lattice and closes h5 file
        Args:
            filename: file to save new trajectory in
        """

        self.filename = filename
        self.simulation.reporters[0].close()
        self.simulation.reporters[0] = HDF5Reporter(
            self.filename, self.invdt // 100)
        self.simulation.context.setPositions(self._init_position())

    def get_dissipation(self):
        """Gets dissipation of simulation
        Returns:
            Mean total dissipation across all particles
        """
        dissipation = self.simulation.integrator.getPerDofVariableByName(
            "dissipation")
        dissipation = np.array([d_n[0] for d_n in dissipation])
        return np.mean(dissipation)


if __name__ == "__main__":
    active = ActiveSystem(filename="temp.h5", region_num=2)
    active.run_decorrelation(10)
    active.update_action([0, 0.25, 0.5, 0.75])
    for i in range(10):
        active.run_step()
    print(active.get_state_reward())
    print(active.get_state_reward_all(tag=""))
    active.update_action([1.5, 1.5, 1.5, 1.5])
    for i in range(10):
        active.run_step()

    #
    # lj = LJSystem(filename="temp.h5", region_num=2)
    # lj.run_decorrelation(10)
    #
    # lj.update_action([0.001, 0.1, 0.5, 0.75])
    # for i in range(50):
    #     lj.run_step()
    # print(lj.get_state_reward())
    # print(lj.get_state_reward_all(tag=""))
    # lj.update_action([1.5, 1.5, 1.5, 1.5])
    # for i in range(50):
    #     lj.run_step()
