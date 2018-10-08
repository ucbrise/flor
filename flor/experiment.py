#!/usr/bin/env python3

import os
import inspect
from typing import Dict

import pandas as pd

import flor.global_state as global_state
import flor.util as util
import flor.above_ground as ag
from flor.jground import GroundClient
from flor.object_model import *
from flor.experiment_graph import ExperimentGraph
from flor.stateful import State
from flor.data_controller.versioner import Versioner
from flor import viz
from flor.global_state import interactive

from datetime import datetime
from ground.client import GroundClient
from grit.client import GroundClient as GritClient
import requests
import warnings
import itertools
from functools import reduce
from graphviz import Source


class Experiment(object):

    def __init__(self, name, backend="git"):
        self.xp_state = State()
        self.xp_state.pre_pull = True
        self.xp_state.EXPERIMENT_NAME = name
        self.xp_state.eg = ExperimentGraph()

        self.repo_path = os.path.join(self.xp_state.versioningDirectory, self.xp_state.EXPERIMENT_NAME)

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
        # ag.CommitTracker(self.xp_state).commit()
        self.xp_state.eg.clean()

    def groundClient(self, backend):
        """
        Keeping for backward compatibility. Will clean up.
        :param backend:
        :return:
        """
        pass

    def literal(self, v=None, name=None, parent=None, label=None):
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
        lit = Literal(v, name, parent, self.xp_state, default=None, version=label)
        self.xp_state.eg.node(lit)

        if parent:
            self.xp_state.eg.edge(parent, lit)

        return lit

    def literalForEach(self, v=None, name=None, parent=None, default=None, label=None):
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
        lit = Literal(v, name, parent, self.xp_state, default, version=label)
        self.xp_state.eg.node(lit)
        lit.__forEach__()

        if parent:
            self.xp_state.eg.edge(parent, lit)

        return lit

    def artifact(self, loc, name, parent=None, label=None, identifier=None):
        """

        :param loc:
        :param name:
        :param parent:
        :param version:
        :return:
        """
        if parent is not None and label is not None:
            raise ValueError("Can't set a utag for a derived artifact")
        art = Artifact(loc, parent, name, label, identifier, self.xp_state)
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
            code_artifact = self.artifact(filenameWithFunc, '.'.join(filenameWithFunc.split('.')[0:-1]))

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

    def __get_pulls__(self):
        with util.chinto(self.repo_path):
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
        return pulls

    def summarize(self):
        # For now summarizes everything
        # TODO: get_next_block access method (history may be very long)

        pulls = self.__get_pulls__()

        semistructured_rep = []
        columns = ['label',]
        artifacts = []



        with util.chinto(self.repo_path):
            for i, pull_d in enumerate(pulls):
                util.__runProc__(['git', 'checkout', pull_d['commit']])
                eg = ExperimentGraph.deserialize()

                sinks = eg.sinks

                semistructured_rep.append({})
                for sink_action in eg.actions_at_depth[max(eg.actions_at_depth.keys())]:
                    if eg.d[sink_action].issubset(sinks):
                        response = eg.summary_back_traverse(sink_action)

                        for sink in eg.d[sink_action]:
                            if type(sink).__name__[0:len('Literal')] == "Literal":
                                if sink.name not in columns:
                                    columns.insert(1, sink.name)
                                response['Literal'] |= {sink,}
                            else:
                                response['Artifact'] |= {sink,}

                        for v in response['Literal']:
                            if v.name not in columns:
                                columns.append(v.name)
                        for v in response['Artifact']:
                            if v.name not in columns:
                                columns.append(v.name)
                                artifacts.append(v.name)
                        try:
                            semistructured_rep[i][pull_d['message'] + ':' + str(id(sink))] = set(map(lambda x: (x.name, x.v), response['Literal']))
                            semistructured_rep[i][pull_d['message'] + ':' + str(id(sink))] |= set(map(lambda x: (x.name, os.path.basename(x.isolated_loc)), response['Artifact']))
                        except:
                            continue
            util.__runProc__(['git', 'checkout', 'master'])

        ltuples = []

        for pull_container in semistructured_rep:
            for kee in pull_container:
                tuple = {}
                tuple['label'] = kee.split(':')[1]
                for name, value in pull_container[kee]:
                    tuple[name] = value
                ltuples.append(tuple)

        # return pd.DataFrame(ltuples, columns=columns)
        return FlorFrame(data=ltuples, columns=columns, artifacts=artifacts)

    def plot(self, label):
        # Deprecated: keeping for backward compatibility
        pulls = self.__get_pulls__()

        with util.chinto(self.repo_path):
            for i, pull_d in enumerate(pulls):
                if label == pull_d['message'].split(':')[1]:
                    util.__runProc__(['git', 'checkout', pull_d['commit']])
                    if not interactive:
                        Source.from_file('output.gv').view()
                    else:
                        output_image = Source.from_file('output.gv')
                    break
            util.__runProc__(['git', 'checkout', 'master'])

        return output_image

    def flor_plan(self, label):
        # Deprecated: keeping for backward compatibility
        pulls = self.__get_pulls__()

        with util.chinto(self.repo_path):
            for i, pull_d in enumerate(pulls):
                if label == pull_d['message'].split(':')[1]:
                    util.__runProc__(['git', 'checkout', pull_d['commit']])
                    if not interactive:
                        Source.from_file('output.gv').view()
                    else:
                        output_image = Source.from_file('output.gv')
                    break
            util.__runProc__(['git', 'checkout', 'master'])

        return output_image


    def diff(self, ulabel, vlabel, flags=['--name-only']):
        pulls = list(self.__get_pulls__())

        ud, = filter(lambda x: ulabel == x['message'].split(':')[1], pulls)
        vd, = filter(lambda x: vlabel == x['message'].split(':')[1], pulls)

        with util.chinto(self.repo_path):
            res = util.__readProc__(['git', 'diff', '--color']  + flags + [ud['commit'], vd['commit']])

        print("These files are different:")
        print(res)

        # output_dir = "{}_{}".format(self.xp_state.EXPERIMENT_NAME, self.xp_state.outputDirectory)
        # u_nested_dir = os.path.join(output_dir, str(utag))
        # v_nested_dir = os.path.join(output_dir, str(vtag))
        #
        # u_files = {}
        # v_files = {}
        #
        #
        # for u_file in os.listdir(u_nested_dir):
        #     x = u_file.split('.')
        #     name = '.'.join(x[0:-1])
        #     ext = str(x[-1])
        #
        #     parts = name.split('_')
        #
        #     u_files['_'.join(parts[0:-1]) + ext] = u_file
        #
        # for v_file in os.listdir(v_nested_dir):
        #     x = v_file.split('.')
        #     name = '.'.join(x[0:-1])
        #     ext = str(x[-1])
        #
        #     parts = name.split('_')
        #
        #     v_files['_'.join(parts[0:-1]) + ext] = v_file
        #
        # and_files = set(u_files.keys()) & set(v_files.keys())
        #
        # print()

        # print("BINARY DIFF")
        # print("\n".join(["{} in {} but not in {}".format(u_files[each], utag, vtag) for each in u_files if each not in v_files]))
        # print("\n".join(["{} in {} but not in {}".format(v_files[each], vtag, utag) for each in v_files if each not in u_files]))
        # for f in and_files:
        #     print(util.__readProc__(['diff', os.path.join(u_nested_dir, u_files[f]), os.path.join(v_nested_dir, v_files[f])]))



class FlorFrame(pd.DataFrame):

    def __init__(self, artifacts, **kwargs):
        warnings.filterwarnings('ignore')
        super().__init__(**kwargs)
        self.___columns = kwargs['columns']
        self.___artifacts = artifacts
        warnings.filterwarnings('default')


    def cube(self, aggregation_dict: Dict):
        # cube_columns = [i for i in self.___columns if i not in self.___artifacts]

        for kee in aggregation_dict:
            assert kee not in self.___artifacts, "Cannot aggregate an artifact"

        utag = self.___columns[0]

        cube_columns = [i for i in self.___columns[1:] if i not in aggregation_dict.keys()]

        combinations = []
        for i in range(len(cube_columns)):
            combinations += list(itertools.combinations(cube_columns, i+1))

        dataframes = []

        for element in combinations:
            # dataframes.append(self.groupby([utag] + list(element))[pulled_art].mean().to_frame().reset_index())
            dataframes.append(self.groupby([utag] + list(element)).agg(aggregation_dict).reset_index())

        # reduce(lambda x, y: x.append(y).reset_index(drop=True), dataframes)

        # print("cube_columns: {}".format(cube_columns))
        out_df = reduce(lambda x, y: x.append(y).reset_index(drop=True), dataframes)[list(aggregation_dict.keys()) + [utag,] + cube_columns]
        out_df[cube_columns] = out_df[cube_columns].fillna(value='ALL')
        return out_df


