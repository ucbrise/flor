#!/usr/bin/env python3
import inspect
import os

from jarvis.experiment_graph import ExperimentGraph
import jarvis.global_state as global_state

from ground.client import GroundClient

class State:

    def __init__(self):
        self.edges = []
        self.eg : ExperimentGraph = None
        self.EXPERIMENT_NAME = None
        self.gc : GroundClient = None
        self.ghostFiles = set([])

        if global_state.interactive:
            self.jarvisFile = global_state.nb_name
        else:
            frame = inspect.stack()[2]
            module = inspect.getmodule(frame[0])
            self.jarvisFile = module.__file__.split('/')[-1]

        self.literals = []
        self.literalFilenames = 0
        self.literalNames = set([])
        self.literalNameToObj = {}
        self.nodes = {}
        self.tmpexperiment = '/tmp/de9f2c7fd25e1b3afad3e85a0bd17d9b100db4b3'
        self.versioningDirectory = os.path.expanduser('~') + '/' + 'jarvis.d' #please check; maybe redundant with overriding in pulls
        self.visited = []
        self.ray = {}

    def literalFilenamesAndIncr(self):
        x = self.literalFilenames
        self.literalFilenames += 1
        return x
