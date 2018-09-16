#!/usr/bin/env python3

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from flor.experiment_graph import ExperimentGraph
    from flor.stateful import State

import git
import tempfile
import os
from shutil import copytree
from shutil import rmtree
from shutil import move
from shutil import copy2 as copy

class Versioner:
    """
    Responsible for putting Literal and Code artifacts in ~/flor.d
    """

    def __init__(self, eg: 'ExperimentGraph', xp_state: 'State'):
        self.eg = eg
        self.xp_state = xp_state
        self.original = os.getcwd()
        self.versioning_dir = None

    def __move_starts__(self):
        for start in self.eg.starts:
            #TODO: Generalize. What if we have non-python scripts or files
            #TODO: Will need to generalize by typing flor Artifacts, propagate to ArtifactLight
            if type(start).__name__[0:len('Artifact')] == 'Artifact':
                start = start.loc
                if 'py' == start.split('.')[-1]:
                    copy(start, os.path.join(self.versioning_dir, start))
        copy('experiment_graph.pkl', os.path.join(self.versioning_dir, 'experiment_graph.pkl'))
        copy(self.xp_state.florFile, os.path.join(self.versioning_dir, self.xp_state.florFile))

    def __git_commit__(self, mode):
        os.chdir(self.versioning_dir)
        if mode == 'initial':
            repo = git.Repo.init(os.getcwd())
            repo.git.add(A=True)
            repo.index.commit('initial commit')
        else:
            repo = git.Repo(os.getcwd())
            repo.git.add(A=True)
            repo.index.commit('incremental commit')
        os.chdir(self.original)


    def save_commit_evnet(self):
        self.versioning_dir = os.path.join(self.xp_state.versioningDirectory, self.xp_state.EXPERIMENT_NAME)
        if os.path.exists(self.versioning_dir):
            with tempfile.TemporaryDirectory() as tempdir:
                move(os.path.join(self.versioning_dir, '.git'), tempdir)
                #TODO: optimize, this remove results in redundant copy
                rmtree(self.versioning_dir)
                os.mkdir(self.versioning_dir)
                self.__move_starts__()
                move(os.path.join(tempdir, '.git'), os.path.join(self.versioning_dir, '.git'))
            self.__git_commit__('incremental')
        else:
            if not os.path.exists(self.xp_state.versioningDirectory):
                os.mkdir(self.xp_state.versioningDirectory)
            os.mkdir(self.versioning_dir)
            self.__move_starts__()
            self.__git_commit__('initial')
