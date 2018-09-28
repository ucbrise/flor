#!/usr/bin/env python3

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from flor.experiment_graph import ExperimentGraph
    from flor.stateful import State

import os
from shutil import copy2 as copy


class Organizer:
    """
    Responsible for organizing some Artifacts and all Literals
    """
    def __init__(self, version, eg: 'ExperimentGraph', xp_state: 'State'):
        self.eg = eg
        self.xp_state = xp_state
        self.output_dir = "{}_{}".format(self.xp_state.EXPERIMENT_NAME, self.xp_state.outputDirectory)
        self.nested_dir = os.path.join(self.output_dir, version)

    def __create_output_directory__(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        os.mkdir(self.nested_dir)


    def __move_artifacts__(self):
        artifacts = self.eg.get_artifacts_reachable_from_starts()
        assert(all([type(art).__name__ == 'ArtifactLight' for art in artifacts]))

        intermediary_artifacts = filter(lambda art: art.parent, artifacts)

        for art in intermediary_artifacts:
            os.rename(art.get_isolated_location(), os.path.join(self.nested_dir, art.get_isolated_location()))
        copy('experiment_graph.pkl', os.path.join(self.nested_dir, 'experiment_graph.pkl'))
        if os.path.exists('output.gv'):
            os.rename('output.gv', os.path.join(self.nested_dir, 'output.gv'))
        if os.path.exists('output.gv.pdf'):
            os.rename('output.gv.pdf', os.path.join(self.nested_dir, 'output.gv.pdf'))


    def run(self):
        self.__create_output_directory__()
        self.__move_artifacts__()


    @staticmethod
    def is_valid_version(xp_state, version):
        # TODO: Check git instead, more reliable
        if version is None:
            return True
        output_dir = "{}_{}".format(xp_state.EXPERIMENT_NAME, xp_state.outputDirectory)
        nested_dir = os.path.join(output_dir, str(version))
        return not os.path.exists(nested_dir)

    @staticmethod
    def resolve_location(xp_state, version, loc):
        if version is None:
            raise ValueError("Must have valid version/utag")
        output_dir = "{}_{}".format(xp_state.EXPERIMENT_NAME, xp_state.outputDirectory)
        nested_dir = os.path.join(output_dir, str(version))
        file_names = filter(lambda s: (loc.split('.')[0].split('_')
                                            == s.split('.')[0].split('_')[0:-1]), os.listdir(nested_dir))
        return [os.path.join(nested_dir, f) for f in file_names]