import numpy as np
import fst
import utils
from utils import peek
from utils import phonology_edit
from variables import Variable_Observed
from variables import Variable_EP
from variables import Variable_EP2
from variables import Variable
from factors import ChangeFactor
from factors import Factor
from edges import Edge

class Node(object):
    """
    Node in the phylogenetic tree
    """
    def __init__(self, name, value=None):
        self.name = name
        self.num = -1
        self.children = []
        self.value = value
        self.i, self.j = -1, -1

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.children)

    def __str__(self):
        return self.name
        
    def __repr__(self):
        return str(self)


class Reconstruction(object):
    """
    Class for Reconstruction
    """
    def __init__(self, sigma, phylogeny, num_nodes):
        self.sigma = sigma
        self.root = phylogeny
        self.num_nodes = num_nodes
        self.variables = np.ndarray((self.num_nodes), dtype="object")
        # TODO: compute exact number of factors
        self.factors = np.ndarray((self.num_nodes-1), dtype="object")
        self._build()

    def _build(self):
        """
        walk the tree
        """

        def walk1(node):
            for child in node:
                walk1(child)

            if len(node) == 0:
                self.variables[self.counter] = Variable_Observed(str(node), self.sigma, node.value)
            else:
                self.variables[self.counter] = Variable_EP2(str(node), self.sigma, 2)
            node.num = self.counter
            self.counter += 1

        def walk2(node):
            for child in node:
                self.factors[self.counter] = ChangeFactor(str(node)+"-"+str(child), self.sigma, phonology_edit(self.sigma, .99))
                i = len(self.variables[node.num].edges)
                j = len(self.factors[self.counter].edges)
                edge = Edge(self.variables[node.num], self.factors[self.counter], self.sigma, True, i, j)
                self.variables[node.num].edges.append(edge)
                self.factors[self.counter].edges.append(edge)
                
                i = len(self.variables[child.num].edges)
                j = len(self.factors[self.counter].edges)
                edge = Edge(self.variables[child.num], self.factors[self.counter], self.sigma, False, i, j)
                self.variables[child.num].edges.append(edge)
                self.factors[self.counter].edges.append(edge)
                
                self.counter += 1
                walk2(child)

        def walk3(node):
            for child in node:
                walk3(child)
            print self.variables[node.i], self.factors[node.j]

        self.counter = 0
        walk1(self.root)
        self.counter = 0
        walk2(self.root)

        # create passing order
        self.forward, self.backward = [], []
        nodes = [self.variables[-1]]
        while len(nodes) > 0:
            node = nodes.pop()
            for edge in node.edges:
                if isinstance(node, Factor) and not edge.child:
                    nodes.append(edge.v)
                    self.forward.append((edge.i, edge.v))
                    self.backward.append((edge.j, edge.f))

                elif isinstance(node, Variable) and edge.child:
                    nodes.append(edge.f)
                    self.forward.append((edge.j, edge.f))
                    self.backward.append((edge.i, edge.v))
                    
        self.backward.reverse()
        

    def inference(self, iterations=5):
        " Perform BP in this network"
        

        # pass up
        print "...passing up"
        for i, node in self.backward:
            #print i, node
            if isinstance(node, Factor):
                edge = node.edges[i]
                print "PASSING", edge.v, "to", node, i
                edge.v.pass_message(edge.i)
                #peek(edge.m_f, 10)

            else:
                edge = node.edges[i]
                print "PASSING", edge.f, "to", node
                edge.f.pass_message(1)
                #peek(edge.m_v, 10)    
                
        # pass down
        print "...passing down"
        for i, node in self.forward:
            #print i, node
            if isinstance(node, Factor):
                edge = node.edges[i]
                print "PASSING", edge.v, "to", node, i
                #edge.v.pass_message(edge.i)
                #peek(edge.m_f, 10)

            else:
                edge = node.edges[i]
                print "PASSING", edge.f, "to", node
                #edge.f.pass_message(1)
                #peek(edge.m_v, 10)    
                



        belief = self.variables[-1].compute_belief()
        peek(belief, 10)
                

                
        
def main():
    # nodes
    spanish = Node("Spanish", "diente")
    portuguese = Node("Portuguese", "dente")
    french = Node("French", "dent")
    italian = Node("Italian", "dente")

    iberian = Node("Iberian")
    western = Node("Western")
    latin = Node("Romance")

    # tree
    latin.children.append(western)
    latin.children.append(italian)
    western.children.append(french)
    western.children.append(iberian)
    iberian.children.append(spanish)
    iberian.children.append(portuguese)
    
    # model
    sigma = fst.SymbolTable()
    counter = 1
    for letter in list("abcdefghijklmnopqrstuvwxyz"):
        sigma[letter] = counter
        counter += 1

    reconstruction = Reconstruction(sigma, latin, 7)
    reconstruction.inference()

if __name__ == "__main__":
    main()
