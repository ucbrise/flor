#!/usr/bin/env python3

import os
import inspect

import pandas as pd

import flor.global_state as global_state
import flor.util as util
import flor.above_ground as ag
from flor.jground import GroundClient
from flor.object_model import *
from flor.experiment_graph import ExperimentGraph
from flor.stateful import State
from flor.data_controller.versioner import Versioner

from datetime import datetime
from ground.client import GroundClient
from grit.client import GroundClient as GritClient
import requests


class Experiment(object):

    def __init__(self, name, backend="git"):
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

        backend = backend.lower().strip()
        if backend == "ground":
            ######################### GROUND GROUND GROUND ###################################################
            # Is Ground Server initialized?
            # Localhost hardcoded into the url
            try:
                requests.get('http://localhost:9000')
            except:
                # No, Ground not initialized
                raise requests.exceptions.ConnectionError('Please start Ground first')
            ######################### </> GROUND GROUND GROUND ###################################################
            self.xp_state.gc = GroundClient()
        elif backend == "git":
            self.xp_state.gc = GritClient()
        else:
            raise ModuleNotFoundError("Only 'git' or 'ground' backends supported, but '{}' entered".format(backend))

    def __enter__(self):
        return self

    def __exit__(self, typ=None, value=None, traceback=None):
        self.xp_state.eg.serialize()
        version = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        Versioner(version, self.xp_state.eg, self.xp_state).save_commit_event()
        ag.CommitTracker(self.xp_state).commit()
        self.xp_state.eg.clean()

    def groundClient(self, backend):
        """
        Keeping for backward compatibility. Will clean up.
        :param backend:
        :return:
        """
        pass

    def literal(self, v=None, name=None, parent=None, utag=None):
        """
        TODO: support version
        :param v:
        :param name:
        :param parent:
        :return:
        """
        if v is None and parent is None:
            raise ValueError("The value or the parent of the literal must be set")
        if v is not None and parent is not None:
            raise ValueError("A literal with a value may not have a parent")
        lit = Literal(v, name, parent, self.xp_state, default=None, version=utag)
        self.xp_state.eg.node(lit)

        if parent:
            self.xp_state.eg.edge(parent, lit)

        return lit

    def literalForEach(self, v=None, name=None, parent=None, default=None, utag=None):
        """
        TODO: Support version
        :param v:
        :param name:
        :param parent:
        :param default:
        :return:
        """
        if v is None and parent is None:
            raise ValueError("The value or the parent of the literal must be set")
        if v is not None and parent is not None:
            raise ValueError("A literal with a value may not have a parent")
        lit = Literal(v, name, parent, self.xp_state, default, version=utag)
        self.xp_state.eg.node(lit)
        lit.__forEach__()

        if parent:
            self.xp_state.eg.edge(parent, lit)

        return lit

    def artifact(self, loc, name, parent=None, utag=None, identifier=None):
        """

        :param loc:
        :param name:
        :param parent:
        :param version:
        :return:
        """
        if parent is not None and utag is not None:
            raise ValueError("Can't set a utag for a derived artifact")
        art = Artifact(loc, parent, name, utag, identifier, self.xp_state)
        self.xp_state.eg.node(art)

        if parent:
            self.xp_state.eg.edge(parent, art)

        return art

    def action(self, func, in_artifacts=None):
        """
        # TODO: What's the equivalent of re-using code from a previous experiment version?

        :param func:
        :param in_artifacts:
        :return:
        """
        filenameWithFunc, funcName, _ = func

        if filenameWithFunc in self.xp_state.eg.loc_map:
            code_artifact = self.xp_state.eg.loc_map[filenameWithFunc]
        else:
            code_artifact = self.artifact(filenameWithFunc, funcName)

        if in_artifacts:
            temp_artifacts = []
            for in_art in in_artifacts:
                if not util.isFlorClass(in_art):
                    if util.isIterable(in_art):
                        in_art = self.literal(in_art)
                        in_art.__forEach__()
                    else:
                        in_art = self.literal(in_art)
                temp_artifacts.append(in_art)
            in_artifacts = temp_artifacts

        in_artifacts = in_artifacts or []

        act = Action(func, [code_artifact, ] + in_artifacts, self.xp_state)
        self.xp_state.eg.node(act)
        self.xp_state.eg.edge(code_artifact, act)
        if in_artifacts:
            for in_art in in_artifacts:
                self.xp_state.eg.edge(in_art, act)

        return act

    def summarize(self):
        # For now summarizes everything
        # TODO: get_next_block access method (history may be very long)

        repo_path = os.path.join(self.xp_state.versioningDirectory,self.xp_state.EXPERIMENT_NAME)

        with util.chinto(repo_path):
            ld = util.git_log()

        ld4 =[]
        d4 = {}
        for i, d in enumerate(ld):
            if ((i % 4) == 0):
               if i > 0:
                   ld4.append(d4)
                   d4 = {}
            d4.update(d)
        ld4.append(d4)

        pulls = filter(lambda x: 'pull' == x['message'].split(':')[0], ld4)

        semistructured_rep = []
        columns = ['utag',]

        with util.chinto(repo_path):
            for i, pull_d in enumerate(pulls):
                util.__runProc__(['git', 'checkout', pull_d['commit']])
                eg = ExperimentGraph.deserialize()

                semistructured_rep.append({})
                for sink in eg.sinks:
                    response = eg.summary_back_traverse(sink)
                    if sink.name not in columns:
                        columns.append(sink.name)
                    for v in response['Literal']:
                        if v.name not in columns:
                            columns.append(v.name)
                    for v in response['Artifact']:
                        if v.name not in columns:
                            columns.append(v.name)
                    semistructured_rep[i][pull_d['message'] + ':' + str(id(sink))] = set(map(lambda x: (x.name, x.v), response['Literal']))
                    semistructured_rep[i][pull_d['message'] + ':' + str(id(sink))] |= set(map(lambda x: (x.name, x.isolated_loc), response['Artifact']))
            util.__runProc__(['git', 'checkout', 'master'])

        ltuples = []

        for pull_container in semistructured_rep:
            for kee in pull_container:
                tuple = {}
                tuple['utag'] = kee.split(':')[1]
                for name, value in pull_container[kee]:
                    tuple[name] = value
                ltuples.append(tuple)

        return pd.DataFrame(ltuples, columns=columns)




