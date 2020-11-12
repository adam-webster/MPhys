import networkx as nx
import matplotlib.pyplot as plt


class GraphGenerator(object):

    def __init__(self, players):
        self.agents = players
        self.names = self.player_naming()
        self.len = len(self.names)

    def __str__(self):
        '''
        The GraphGenerator object will simply be a list of
        the names of strategies present in the graph
        '''
        return str(self.names)

    def player_naming(self):
        '''
        To generate a list of player names
        '''
        player_names = []
        for agent in self.agents:
            player_names.append(str(agent))
        return player_names

    def ring_graph(self, print_list: bool = False):
        '''
        Generates a ring (cycle) graph based on
        the order of players given
        '''
        edges = [(i, i+1) for i in range(self.len-1)]
        edges.append((self.len-1, 0))
        if print_list:
            print(f"Edges list is: {edges}")
        return edges

    def complete_graph(self, print_list: bool = False):
        '''
        Generates a complete graph. This is where all nodes
        are connected to all other nodes
        '''
        edges = [(i, j) for i in range(self.len)
                        for j in range(i+1, self.len)]
        if print_list:
            print(f"Edges list is: {edges}")
        return edges



    def visualise_graph(self, edges):
        '''
        This function will produce a visualisation of the
        graph structure. It should work regardless of the
        structure of the graph
        '''
        nodes = {}
        for name in self.names:
            nodes[self.names.index(name)] = name
        G = nx.Graph(edges)
        G = nx.relabel_nodes(G,nodes)
        nx.draw(G, with_labels=True)
        plt.show() #to ensure they show, matplotlib is not very friendly lately
