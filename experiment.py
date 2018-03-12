#!/usr/bin/env python3

import os
import inspect
import json
import pandas as pd
import time

from . import global_state
from . import util
from .jground import GroundClient
from .object_model import *
from .decorators import func

class Experiment:

    def __init__(self, name):
        self.xp_state = State()
        self.xp_state.EXPERIMENT_NAME = name

        # https://stackoverflow.com/a/13699329/9420936
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        filename = module.__file__
        if not global_state.interactive:
            target_dir = '/'.join(filename.split('/')[0:-1])
            if target_dir:
                os.chdir(target_dir)

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
                        in_art = self.literal(in_art)
                temp_artifacts.append(in_art)
            in_artifacts = temp_artifacts
        return Action(func, in_artifacts, self.xp_state)

    def plate(self, name):
        return Plate(self, name)

class State:

    def __init__(self):
        self.edges = []
        self.EXPERIMENT_NAME = None
        self.gc = None
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

class Plate:

    def __init__(self, experiment, name):
        self.experiment = experiment
        self.name = name + '.pkl'
        self.artifacts = []

    def artifact(self, loc, parent=None, manifest=False):
        artifact = self.experiment.artifact(loc, parent, manifest)
        self.artifacts.append(artifact)
        return artifact

    def action(self, func, in_artifacts=None):
        return self.experiment.action(func, in_artifacts)

    def exit(self):

        @func
        def plate_aggregate(*args):
            original_dir = os.getcwd()
            os.chdir(self.experiment.xp_state.tmpexperiment)
            dirs = [x for x in os.listdir() if util.isNumber(x)]
            table = []

            # NOTE: Use the jarvis.Experiment experimentName in the future
            experimentName = self.experiment.xp_state.jarvisFile.split('.')[0]

            literalNames = self.experiment.xp_state.ray['literalNames']

            for trial in dirs:
                os.chdir(trial)
                with open('.' + experimentName + '.jarvis', 'r') as fp:
                    config = json.load(fp)
                record = {}
                for literalName in literalNames:
                    record[literalName] = config[literalName]
                for artifact in self.artifacts:
                    while True:
                        try:
                            # I need to wait for every to finish. not just this one.
                            record[artifact.loc.split('.')[0]] = util.loadArtifact(artifact.loc)
                            break
                        except:
                            time.sleep(5)
                    if util.isNumber(record[artifact.loc.split('.')[0]]):
                        record[artifact.loc.split('.')[0]] = eval(record[artifact.loc.split('.')[0]])
                table.append(record)
                os.chdir('../')

            os.chdir(original_dir)

            return pd.DataFrame(table)

        plate_aggregate = (self.experiment.xp_state.jarvisFile, plate_aggregate[1], plate_aggregate[2])

        do_action = self.experiment.action(plate_aggregate, self.artifacts)
        artifact_df = self.experiment.artifact(self.name, do_action)

        return artifact_df

