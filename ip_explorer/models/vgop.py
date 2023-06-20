from .base import PLModelWrapper

import ast
import math
import torch
import numpy as np
import networkx as nx
from ase import neighborlist
from matscipy.neighbours import neighbour_list
from itertools import combinations_with_replacement


class VGOPModelWrapper(PLModelWrapper):
    def __init__(self, model_dir, **kwargs):
        """
        Arguments:

            min_cut (float):
                The smallest cutoff distance to use

            max_cut (float):
                The largest cutoff distance to use

            num_cutoffs (int):
                The number of cutoffs to use

            elements (list):
                A list of element types

            interactions (list or 'all'):
                A list of interactions. Graphs will be constructed for each
                interaction type. Example: `[(0, 0), (0, 1), (1, 1)]`. If 'all',
                will use all possible interaction types based on `elements`.
        """
        if 'min_cut' not in kwargs:
            raise RuntimeError("Must specify min_cut for VGOP. Use --additional-kwargs argument.")
        if 'max_cut' not in kwargs:
            raise RuntimeError("Must specify max_cut for VGOP. Use --additional-kwargs argument.")
        if 'num_cutoffs' not in kwargs:
            raise RuntimeError("Must specify num_cutoffs distances for VGOP. Use --additional-kwargs argument.")

        self.cutoffs = np.linspace(
            ast.literal_eval(kwargs['min_cut']),
            ast.literal_eval(kwargs['max_cut']),
            ast.literal_eval(kwargs['num_cutoffs'])
        )

        if 'elements' not in kwargs:
            raise RuntimeError("Must specify elements for VGOP. Use --additional-kwargs argument.")

        self.elements = ast.literal_eval(kwargs['elements'])

        if 'interactions' not in kwargs:
            raise RuntimeError("Must specify interactions for VGOP. Should be in format [(0,0),(0,1),(1,1),...]. Use --additional-kwargs argument.")

        self.interactions = ast.literal_eval(kwargs['interactions'])

        if self.interactions == 'all':
            self.interactions = list(combinations_with_replacement(
                list(range(len(self.elements))),
                len(self.elements)
            ))  # for all bond types

        if 'pad_atoms' not in kwargs:
            self.pad_atoms = False
        else:
            self.pad_atoms = ast.literal_eval(kwargs['pad_atoms'])

        if 'take_chemical' not in kwargs:
            self.take_chemical = False
        else:
            self.take_chemical = ast.literal_eval(kwargs['take_chemical'])

        if 'take_radial' not in kwargs:
            self.take_radial = True
        else:
            self.take_radial = ast.literal_eval(kwargs['take_radial'])


        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, model_path):
        self.model = VGOPModel(
            cutoffs=self.cutoffs,
            elements=self.elements,
            interactions=self.interactions,
            take_chemical=self.take_chemical,
            take_radial=self.take_radial,
        )


    def compute_structure_representations(self, batch):
        """
        Assumes that 'batch' is just a list of ASE Atoms objects with supercell
        energies stored under the `atoms.info['energy']` field.
        """

        representations = []
        splits          = []
        energies        = []

        for original in batch:
            atoms = original.copy()

            natoms = len(atoms)

            if self.pad_atoms:
                atoms.center(vacuum=2*np.max(self.cutoffs))
                atoms.pbc = True

            v = torch.from_numpy(self.model(atoms))

            representations.append(v.tile((natoms, 1)))

            splits.append(natoms)
            energies.append(atoms.info['energy']/natoms)

        representations = torch.cat(representations, dim=0)
        energies        = torch.Tensor(energies)

        return {
            'representations': representations,
            'representations_energy': energies,
        }


    def copy(self, model_path):
        pass


class VGOPModel:
    """
    A class for computing the VGOP of a set of atoms.

    The code for this model was taken from a personal correspondence with Dr.
    James Chapman (jc112358@bu.edu), and is not publicly available outside of
    LLNL yet.
    """

    def __init__(self, cutoffs, elements, interactions, take_chemical, take_radial):
        self.cutoffs        = cutoffs
        self.elements       = elements
        self.interactions   = interactions
        self.take_chemical  = take_chemical
        self.take_radial  = take_radial

    @staticmethod
    def get_chemical_graph_node_degrees(a,sg, graph):
        degrees = []
        for node in sg:
            degree = 0.0
            edges = list(graph.edges(node))
            for edge in edges:
                chem_index = self.elements.index(a[edge[1]].symbol) + 1
                degree += float(chem_index)
            degree /= float(len(edges))
            degrees.append(degree)

        return degrees

    @staticmethod
    def get_radial_graph_node_degrees(a,sg, graph):
        degrees = []
        for node in sg:
            degree = 0.0
            edges = list(graph.edges(node))
            for edge in edges:
                dist = a.get_distance(edge[0],edge[1])
                degree += 1.0 / dist**2

            degrees.append(degree)

        return degrees

    @staticmethod
    def visualize_graph(g, a, elements):
        color_map = []
        for node in g:
            if a[node].symbol == elements[0]:
                color_map.append('pink')
            else:
                color_map.append('grey')
        nx.draw(g, node_color=color_map, edge_color='lightsteelblue', width=1, edgecolors='k', alpha=0.75)
        plt.show()

    # @staticmethod
    # def create_graph(a,nls):
    #     gr = nx.Graph()
    #     for i in range(len(a)):
    #         gr.add_node(i)
    #     for i in range(len(a)):
    #         neighs = nls.get_neighbors(i)[0]
    #         for neigh in neighs:
    #             gr.add_edge(i,neigh)

    #     return gr

    @staticmethod
    def create_graph(a, idx_i, idx_j):
        gr = nx.Graph()
        for i in range(len(a)):
            gr.add_node(i)
        for i,j in zip(idx_i, idx_j):
            gr.add_edge(i, j)

        return gr

    @staticmethod
    def subgraph_op(subgraph,degrees):
        op = 0.0
        vals, cnts = np.unique(degrees, return_counts=True)

        probs = []
        total_nodes = len(subgraph)
        for count in cnts:
            probs.append(float(count) / float(total_nodes))

        for j,prob in enumerate(probs):
            # Use masked array for safe log
            op += prob*np.ma.log(np.atleast_1d(prob)).filled(0) + prob*vals[j]
        op = op**3

        return op

    def __call__(self, atoms):

        theta = []
        for ti in self.interactions:
                keep = []
                copy_atoms = atoms[:]
                for atom in copy_atoms:
                    check = 0
                    for element in ti:
                        if self.elements[element] == atom.symbol:
                            check = 1
                            break
                    if check == 0:
                        del atom

                for i, cut in enumerate(self.cutoffs):
                    n = []
                    for atom in copy_atoms:
                        n.append(cut)
                    # nl = neighborlist.NeighborList(n, skin=0, self_interaction=False, bothways=True)
                    # nl.update(copy_atoms)
                    # graph = self.create_graph(copy_atoms, nl)
                    idx_i, idx_j = neighbour_list('ij', copy_atoms, cut)
                    graph = self.create_graph(copy_atoms, idx_i, idx_j)
                    subgraphs = list(nx.connected_components(graph))
                    # if visualize_graphs:
                    #     if i == 0:
                    #         self.visualize_graph(graph, copy_atoms, elements)

                    chemical_graph_op = 0.0
                    radial_graph_op = 0.0
                    for sg in subgraphs:
                        sg = graph.subgraph(sg)

                        if self.take_chemical:
                            chemical_graph_degrees = self.get_chemical_graph_node_degrees(copy_atoms, sg, graph, elements)
                            chemical_graph_op += self.subgraph_op(sg, chemical_graph_degrees)
                        if self.take_radial:
                            radial_graph_degrees = self.get_radial_graph_node_degrees(copy_atoms, sg, graph)
                            radial_graph_op += self.subgraph_op(sg, radial_graph_degrees)
                    if self.take_chemical:
                        theta.append(chemical_graph_op)
                    if self.take_radial:
                        theta.append(radial_graph_op)

        theta = np.array(theta)
        return theta[:, 0]
