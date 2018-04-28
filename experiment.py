#!/usr/bin/env python3

import os
import inspect
import json
import pandas as pd
import time
import git

import flor.global_state as global_state
import flor.util as util
import flor.above_ground as ag
from flor.jground import GroundClient
from flor.object_model import *
from flor.decorators import func
from flor.experiment_graph import ExperimentGraph
from flor.stateful import State

from ground.client import GroundClient
from shutil import copytree
from shutil import rmtree
from shutil import move

class Experiment(object):

    def __init__(self, name):
        self.xp_state = State()
        self.xp_state.EXPERIMENT_NAME = name
        self.xp_state.eg = ExperimentGraph()

        if not global_state.interactive:
            # https://stackoverflow.com/a/13699329/9420936
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            filename = module.__file__
            if not global_state.interactive:
                target_dir = '/'.join(filename.split('/')[0:-1])
                if target_dir:
                    os.chdir(target_dir)

    def __enter__(self):
        return self

    def __exit__(self, typ=None, value=None, traceback=None):
        self.xp_state.eg.serialize()
        original = os.getcwd()
        if os.path.exists(self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME):
            move(self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME + '/.git', '/tmp/')
            rmtree(self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME)
            move('/tmp/.git', self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME + '/.git')
            copytree(os.getcwd(), self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME + '/0')
            os.chdir(self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME)
            repo = git.Repo(os.getcwd())
            repo.git.add(A=True)
            repo.index.commit('incremental commit')
        else:
            copytree(os.getcwd(), self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME + '/0')
            os.chdir(self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME)
            repo = git.Repo.init(os.getcwd())
            repo.git.add(A=True)
            repo.index.commit('initial commit')
        os.chdir(original)
        ag.commit(self.xp_state, 'Pre')


    def groundClient(self, backend):
        # self.xp_state.gc = GroundClient(backend)
        self.xp_state.gc = GroundClient()

    def literal(self, v, name=None):
        lit =  Literal(v, name, self.xp_state)
        self.xp_state.eg.node(lit)

        return lit

    def artifact(self, loc, parent=None, manifest=False):
        art = Artifact(loc, parent, manifest, self.xp_state)
        self.xp_state.eg.node(art)
        if parent:
           self.xp_state.eg.edge(parent, art)

        return art

    def action(self, func, in_artifacts=None):
        if in_artifacts:
            temp_artifacts = []
            for in_art in in_artifacts:
                if not util.isFlorClass(in_art):
                    if util.isIterable(in_art):
                        in_art = self.literal(in_art)
                        in_art.forEach()
                    else:
                        in_art = self.literal(in_art)
                temp_artifacts.append(in_art)
            in_artifacts = temp_artifacts
        act =  Action(func, in_artifacts, self.xp_state)
        self.xp_state.eg.node(act)
        if in_artifacts:
            for in_art in in_artifacts:
                self.xp_state.eg.edge(in_art, act)

        return act

    def plate(self, name):
        return Plate(self, name)

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

            # NOTE: Use the flor.Experiment experimentName in the future
            experimentName = self.experiment.xp_state.florFile.split('.')[0]

            literalNames = self.experiment.xp_state.ray['literalNames']

            for trial in dirs:
                os.chdir(trial)
                with open('.' + experimentName + '.flor', 'r') as fp:
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

        plate_aggregate = (self.experiment.xp_state.florFile, plate_aggregate[1], plate_aggregate[2])

        do_action = self.experiment.action(plate_aggregate, self.artifacts)
        artifact_df = self.experiment.artifact(self.name, do_action)

        return artifact_df

