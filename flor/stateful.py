#!/usr/bin/env python3
import inspect
import os

from typing import Union

from flor.experiment_graph import ExperimentGraph
import flor.global_state as global_state

from grit.client import GroundClient as GritClient
from ground.client import GroundClient


class State:

    def __init__(self):
        self.actionNameToObj = {}
        self.artifactNameToObj = {}
        self.edges = []
        self.eg : ExperimentGraph = None
        self.EXPERIMENT_NAME = None
        self.gc : Union[GroundClient, GritClient] = None
        self.ghostFiles = set([])

        if global_state.interactive:
            self.florFile = global_state.nb_name
        else:
            frame = inspect.stack()[2]
            module = inspect.getmodule(frame[0])
            self.florFile = module.__file__.split('/')[-1]

        self.literals = []
        self.literalFilenames = 0
        self.literalNames = set([])
        self.literalNameToObj = {}
        self.nodes = {}
        self.tmpexperiment = '/tmp/de9f2c7fd25e1b3afad3e85a0bd17d9b100db4b3'
        self.versioningDirectory = os.path.expanduser('~') + '/' + 'flor.d' #please check; maybe redundant with overriding in pulls
        self.outputDirectory = 'out.d'
        self.visited = []
        self.ray = {}

        # Fall 2018

        self.pre_pull = False
        self.pull_write_version = None

    def literalFilenamesAndIncr(self):
        x = self.literalFilenames
        self.literalFilenames += 1
        return x
