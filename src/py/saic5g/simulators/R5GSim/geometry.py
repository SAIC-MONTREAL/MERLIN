
from hexalattice.hexalattice import create_hex_grid
import numpy as np
import copy

class Geometry:
    def __init__(self, map_radius, site_radius):
        """ 
        Args:
            map_radius (int): map radius if circle, or half the map side length if square in meters.
            site_radius (int): site coverage radius in meters.

        """
        self.map_radius = map_radius
        self.map_dims = (2*self.map_radius, 2*self.map_radius)  # map dimensions for square case
        self.site_radius = site_radius # (int): site coverage radius
        self.site_positions = None  # (np.array) (n_sites, 2) sites centers' (x,y) positions 
        self.ue_coordinates = None  # (np.array) (n_UEs, 2) UEs (x,y) coordinates
        self.ue_waypoints = None    # (np.array) (n_UEs, 2) UEs (x,y) waypoints
        self.ue_assignments = None  # (np.array) (n_UEs,) UEs' site ID
        self.d2_matrix = None       # (np.array) (n_UEs, n_sites) UEs' d2 distances from each site
        self.d3_matrix = None       # (np.array) (n_UEs, n_sites) UEs' d3 distances from each site
        self.relative_thetas = None # (np.array) (n_sites, n_sites) the angle between each two sites
        self.n_sites = None         # (int) number of sites
        self.n_UEs = None           # (int) number of UEs
        self.adj_m = None           # (np.array) (n_sites, n_sites) adjacency matrix between sites
        self.site_list = None       # (list): A list of the site objects
        self.ues_sector_ID = None   # (np.array) (n_UEs,) UEs' sector ID
 

    def generate_hexs(self, do_plot, shape):
        """ 
        Generate the hex map
        Args:
            do_plot (bool): plot the map
            shape (string): the map shape (square or circle).

        Returns:
            self.n_sites (int):  number of sites
            self.site_positions (np.array):  (n_sites, 2) sites centers (x,y) positions .
            self.adj_m (np.array):  (n_sites, n_sites) the adjacency matrix between sites.
            self.relative_thetas (np.array):  (n_sites, n_sites) the angle matrix between each two sites.

        """
        if shape == 'circle':
            self.site_positions, _ = create_hex_grid(n=100000, min_diam=2*self.site_radius, crop_circ=1.7*self.map_radius, do_plot=do_plot)

        elif shape == 'square':
            map_h, map_w = self.map_dims
            site_radius = self.site_radius
            site_area = 3*np.sqrt(3)/2 * site_radius**2
            n_sites = int((map_h*map_w)/site_area) - 1
            self.site_positions, _ = create_hex_grid(n=n_sites, min_diam=2*site_radius, do_plot=do_plot)
        else:
            raise NotImplementedError
        self.adj_m = self.calc_adj_matrix(self.site_positions, self.site_radius)
        self.n_sites = len(self.site_positions)
        self.calc_relative_thetas()
        return self.n_sites, self.site_positions, self.adj_m, self.relative_thetas
    

    def calc_adj_matrix(self, site_positions, site_radius):
        """ 
        Calculate the adjacency matrix between sites.
        Args:
            site_positions (np.array): (n_sites, 2) sites centers' (x,y) positions
            site_radius (int): site coverage radius

        Returns:
            adj_matrix (np.array):  (n_sites, n_sites) adjacency matrix between sites

        """
        site_x_positions, site_y_positions =  site_positions[:,0], site_positions[:,1]
        site_dist_x = np.abs(np.subtract.outer(site_x_positions, site_x_positions))
        site_dist_y = np.abs(np.subtract.outer(site_y_positions, site_y_positions))
        sites_dist = np.sqrt(site_dist_x**2 + site_dist_y**2)
        adj_matrix = np.logical_and(sites_dist>0, sites_dist<= 2.01*site_radius)
        return adj_matrix*1 # Multiply by 1 to convert a boolean array to a binary array.
    
    def calc_relative_thetas(self):
        """ 
        Calculate the angles between each two sites.

        Returns:
            self.relative_thetas (np.array):  (n_sites, n_sites) the angles between each two sites
        """
        site_x_positions, site_y_positions =  self.site_positions[:,0], self.site_positions[:,1]
        relative_x = np.subtract.outer(site_x_positions, site_x_positions)
        relative_y = np.subtract.outer(site_y_positions, site_y_positions)
        thetas = 360 + np.arctan2(relative_y, relative_x) * 180 / np.pi -180
        thetas[thetas>=360] = thetas[thetas>=360] - 360
        self.relative_thetas = thetas - 180*np.eye(self.n_sites)
        return self.relative_thetas
    

    def distribute_UEs(self, ue_density, ue_max_velocity):
        """ 
        Generate UEs in each site.
        Args:
            ue_density (int): average number of UEs withing a site

        Returns:
            self.ue_assignments (np.array):  (n_UEs,) UEs' site ID.
        """
        site_radius = self.site_radius
        site_positions = self.site_positions
        n_sites = self.n_sites
        

        circle_rad = .9*site_radius # just to ensure the user are inside the hex
        site_x_positions, site_y_positions = site_positions[:,0] , site_positions[:,1]
        ue_assignments = []
        ue_coordinates = []
        for i in range(n_sites):
            n_UEs = np.random.poisson(ue_density)
            ue_assignments.append(i*np.ones(n_UEs))
            theta = np.random.rand(n_UEs)*2*np.pi
            rho = np.clip( np.abs(np.random.normal(0, circle_rad/3, n_UEs)), 10, circle_rad)
            x_cord = rho * np.cos(theta)
            y_cord = rho * np.sin(theta)
            z_cord = np.clip( np.random.normal(2, 4, n_UEs) , 1, 3) # user height
            x_pos_UE, y_pos_UE, z_pos_UE = site_x_positions[i]+x_cord, site_y_positions[i]+y_cord, z_cord
            for x_pos, y_pos, z_pos in zip(x_pos_UE, y_pos_UE, z_pos_UE): 
                ue_coordinates.append( [x_pos, y_pos, z_pos] )
        self.ue_assignments = np.concatenate(ue_assignments).astype(int)
        self.ue_coordinates = np.array(ue_coordinates)
        self.n_UEs = len(self.ue_coordinates)
        self.ues_velcotiy = np.random.rand(self.n_UEs)*ue_max_velocity
        self.ues_sector_ID = np.zeros(self.n_UEs).astype(int)
        self.calc_d2_distances()
        self.calc_d3_distances()
        self.generate_waypoints()

        return self.ue_assignments
    
    def generate_waypoints(self):
        """ 
        Generate UEs' waypoints for simulating the movements of UEs.

        Returns:
            self.waypoints (np.array): (n_UEs, 2) UEs (x,y) waypoints
        """

        target_sites = np.random.randint(0, self.n_sites, self.n_UEs)
        max_distance = .5*self.site_radius
        rho = np.random.rand(self.n_UEs)*max_distance
        theta = np.random.rand(self.n_UEs)*2*np.pi
        x_cord = rho * np.cos(theta)
        y_cord = rho * np.sin(theta)

        waypoints = []
        for i, t_s in enumerate(target_sites):
            x, y = self.site_positions[t_s]
            x, y = x+x_cord[i], y+y_cord[i]
            waypoints.append([x, y])
        
        self.waypoints = np.array(waypoints)
        return self.waypoints
        


    def move_UEs(self, t):
        """
        Move UEs in the simulation towards waypoints.

        Returns:
            self.ue_coordinates (np.array):  (n_UEs, 2) UEs (x,y,z) coordinates
        """
        # speed is m/s
        if (t+1)%(3600*12) == 0:
            self.generate_waypoints()

        ue_coordinates = copy.deepcopy(self.ue_coordinates)
        ue_is_moving = (np.random.rand(self.n_UEs)<0.5)
        ues_velocity = np.array([(v if moving else 0) for moving, v in zip(ue_is_moving, self.ues_velcotiy)])
        ue_heights = ue_coordinates[:, 2]
        x_y_pos = ue_coordinates[:, 0:2]
        new_coords = []
        v_s = self.waypoints - x_y_pos

        for i, (h, v) in enumerate(zip(ue_heights, v_s)):
            x, y = x_y_pos[i] + ues_velocity[i] * v / np.linalg.norm(v)
            z = 1 if (np.random.random()<0.5) else h
            new_coords.append([x, y, z])
        
        self.ue_coordinates = np.array(new_coords)
        self.calc_d2_distances()
        self.calc_d3_distances()

        return self.ue_coordinates

    
    def calc_d2_distances(self):
        """
        Calculate the d2 distances between UEs and Basestations
        
        Returns:
            self.d2_matrix (np.array):  (n_UEs, n_sites) UEs' d2 distances from each site
        """
        ue_x_coord, ue_y_coord = self.ue_coordinates[:,0], self.ue_coordinates[:,1]
        site_x_pos, site_y_pos = self.site_positions[:,0], self.site_positions[:,1]
        dist_x = np.abs(np.subtract.outer(ue_x_coord, site_x_pos))
        dist_y = np.abs(np.subtract.outer(ue_y_coord, site_y_pos))
        self.d2_matrix = np.sqrt(dist_x**2 + dist_y**2)
        return self.d2_matrix
    
    def calc_d3_distances(self):
        """
        Calculate the d3 distances between UEs and Basestations
        
        Returns:
            self.d3_matrix (np.array):  (n_UEs, n_sites) UEs' d3 distances from each site
        """
        site_z_coord = np.array([site.height for site in self.site_list])
        ue_z_coord = self.ue_coordinates[:,2]
        dist_z = np.abs(np.subtract.outer(ue_z_coord, site_z_coord))
        self.d3_matrix = np.sqrt(self.d2_matrix**2 + dist_z**2)
        return self.d3_matrix

    def update_UEs(self):
        """
        Update UEs indicies and info in terms of coordinates, distances from each BS, and the assignments.

        Returns:
            out_of_coverage_ind (list):  the indices of the UEs that went out of coverage
        """
        out_of_coverage = self.d2_matrix>1.5*self.site_radius
        out_of_coverage = np.squeeze( np.all(out_of_coverage, axis=1) )
        out_of_coverage_ind = np.where(out_of_coverage)[0]
        self.d2_matrix = np.delete(self.d2_matrix,  out_of_coverage_ind, axis=0)
        self.d3_matrix = np.delete(self.d3_matrix,  out_of_coverage_ind, axis=0)
        self.ue_coordinates = np.delete(self.ue_coordinates,  out_of_coverage_ind, axis=0)
        self.ue_assignments = np.argmin(self.d2_matrix, axis=1)
        self.n_UEs = len(self.ue_coordinates)
        return out_of_coverage_ind
    

    # TODO: Do some math to do it :) now it's hard coded. 
    def get_sector_adj_and_theta(self, n_sectors):
        """
        "Calculate" the sector adjacency info in each site for the SINR calculation
        Args:
            n_sectors (int): the number of sectors in each site

        Returns:
            sector_adj_m (np.array):  (n_sectors, 6) sector adjacency matrix with sector ID
            sector_thetas (dict): sectors starting thetas associated with their IDs
        """
        if n_sectors == 1:
            sector_adj_m = np.zeros(self.n_sites)
            sector_thetas = {300: 0, 0: 0, 60: 0, 120: 0, 180: 0, 240: 0}
        elif n_sectors == 2:
            # angle = [300 0 60 120 180 240]
            sector_adj_m = np.array( [[0, 0, 0, np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan, 1, 1, 1]])
            sector_thetas = {300: 1, 0: 1, 60: 1, 120: 0, 180: 0, 240: 0}
        elif n_sectors == 3:
            # angle = [300 0 60 120 180 240]
            sector_adj_m = np.array( [[1, 2, np.nan, np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, 2, 0, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan, np.nan, 0, 1]])
            sector_thetas = {300: 0, 0: 0, 60: 1, 120: 1, 180: 2, 240: 2}
        elif n_sectors == 6:
            # angle = [300 0 60 120 180 240]
            sector_adj_m = np.array( [[3, np.nan, np.nan, np.nan, np.nan, np.nan],
                                     [np.nan, 4, np.nan, np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, 5, np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan, 0, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan, np.nan, 1, np.nan],
                                     [np.nan, np.nan, np.nan, np.nan, np.nan, 2]])
            sector_thetas = {300: 0, 0: 1, 60: 2, 120: 3, 180: 4, 240: 5}
        else:
            raise NotImplementedError
        return sector_adj_m, sector_thetas

    
    def set_site_list(self, site_list):
        """
        Setter method for the list of sites.
        Args:
            site_list (int): list of sites in the map

        """
        self.site_list = site_list

    # TODO (easy): implement
    def add_UEs(self):
        """
        Add UEs to the simulation. 
        1- If UEs goes active from inactive status.
        2- New UEs enter the coverage area
        """
        raise NotImplementedError

