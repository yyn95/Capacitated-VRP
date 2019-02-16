"""
A wrapper to translate the data in our defined format
"""

from tsplibparser import TSPLIBParser
import numpy as np
from scipy.spatial import distance_matrix

class CVRPFileParser(TSPLIBParser):


    def parse(self):
        """
        Call the parent parser and calcuate the following instance attributes

        1. (int) capacity
        2. (list of int) demands
        3. (list of list of int) distances
        """

        # call the parnet method to parse the vrp file
        super(CVRPFileParser, self).parse()

        # set self.capacity
        self.capacity = self.data['capacity']

        #
        self.dimension = self.data['dimension']

        # set self.demand
        self._set_demands()

        self._set_coordinates()

        # set self.distances
        self._set_distances()


        return self.data

    def _set_demands(self):
        self.demands = list()
        for d in self.data['demand_section']:
            self.demands.append(d['demand'])

    def _set_coordinates(self):
        point_matrix = np.zeros((self.dimension, 2))
        # get x-arr and y-arr
        for i, d in enumerate(self.data['node_coord_section']):
            point_matrix[i, 0] = d['x']
            point_matrix[i, 1] = d['y']

        self.point_matrix = point_matrix
        self.coordinates = point_matrix.tolist()

    def _set_distances(self):


        # use library to increase the performance
        self.distances_matrix = distance_matrix(self.point_matrix, self.point_matrix, p=2)

        # round to nearest int
        self.distances_matrix = np.rint(self.distances_matrix).astype(int)

        # set the distances attributes
        self.distances = self.distances_matrix.tolist()
