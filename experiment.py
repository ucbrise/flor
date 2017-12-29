#!/usr/bin/env python3

import os
import inspect

from . import global_state
from . import util
from .ground import GroundClient
from .object_model import *

class Experiment:

    def __init__(self, name):
        self.xp_state = State()
        self.experimentName = name

    def groundClient(self, backend):
        self.xp_state.gc = GroundClient(backend)

    def literal(self, v, name=None):
        return Literal(v, name, self.xp_state)

    def artifact(self, loc, parent=None, manifest=False):
        return Artifact(loc, parent, manifest, self.xp_state)

    def action(self, func, in_artifacts=None):
        if in_artifacts:
            temp_artifacts = []
            for in_art in in_artifacts:
                if not util.isJarvisClass(in_art):
                    if util.isIterable(in_art):
                        in_art = self.literal(in_art)
                        in_art.forEach()
                    else:
                        in_art = Literal(in_art)
                temp_artifacts.append(in_art)
            in_artifacts = temp_artifacts
        return Action(func, in_artifacts, self.xp_state)

class State:

    def __init__(self):
        self.edges = []
        self.gc = None
        self.ghostFiles = set([])

        if global_state.interactive:
            self.jarvisFile = global_state.nb_name
        else:
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            self.jarvisFile = module.__file__.split('/')[-1]

        self.literals = []
        self.literalFilenames = 0
        self.literalNames = set([])
        self.literalNameToObj = {}
        self.nodes = {}
        self.versioningDirectory = os.path.expanduser('~') + '/' + 'jarvis.d'
        self.visited = []

    def literalFilenamesAndIncr(self):
        x = self.literalFilenames
        self.literalFilenames += 1
        return x

