#!/usr/bin/env python3


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from flor.stateful import State

import os
import pandas as pd
import cloudpickle

from flor.shared_object_model.resource import Resource
from flor.data_controller.organizer import Organizer

class Artifact(Resource):

    def __init__(self, loc: str,
                 parent, name,
                 version: str,
                 identifier: str,
                 xp_state: 'State'):
        """
        Initialize an Artifact
        :param loc: the path of the artifact
        :param parent: Action that generates this parent
        :param manifest: May be getting deprecated soon
        :param xp_state: The global state of the experiment version
        """
        super().__init__(parent, xp_state)
        self.loc = loc
        self.version = version
        self.identifier = identifier
        self.name = name

        self.xp_state.artifactNameToObj[self.loc] = self
        self.max_depth = 0

    def getLocation(self):
        return self.loc

    def get_name(self):
        return self.name

    def get_shape(self):
        return "box"

    def plot(self, rankdir=None):
        """Deprecated: Saving for backward compatability"""
        return super().__plot__(self.name, "box", rankdir)

    def flor_plan(self, rankdir=None):
        return super().__plot__(self.name, "box", rankdir)

    def pull(self, label=None):
        try:
            super().__pull__(self, label)
        except AssertionError as e:
            print(e)

    def peek(self):
        """
        Interactive method
        Read artifact (TODO: entirely for  now, sample later)
        Take a sample
        return the value
        TODO: Re-enable for derived artifacts, with bound src_literals
        TODO: What's the read strategy? Handling multiple extensions (json, csv, pkl, etc.)
        TODO: What's the sampling strategy?
        :return:
        """
        if self.parent is not None:
            raise NotImplementedError("Peek not currently available for derived artifacts.")
        file_path = self.resolve_location()
        file_name = os.path.relpath(file_path)
        extension = file_name.split('.')[-1]
        if extension == 'csv':
            return pd.read_csv(file_path, nrows=100)
        elif extension == 'json':
            return pd.read_json(file_path)
        elif extension == 'pkl':
            with open(file_path, 'rb') as f:
                out = cloudpickle.load(f)
            return out
        else:
            raise NotImplementedError("Unsupported extension to peek: {}".format(extension))



    def resolve_location(self):
        """
        TODO: Location needs to be resolvable from Ground metadata
        Case: name known but original location unknown
        Case: Run 1: Derived artifact; Run 2: Source Artifact
        :return:
        """
        if self.xp_state.pre_pull:
            if self.version is None or self.parent is not None:
                return self.loc

            file_names = Organizer.resolve_location(self.xp_state, self.version, self.loc)

            if len(file_names) > 1:
                raise NameError("Ambiguous specification: Which item do you want? Specify via Artifact.identifier: {}"
                                .format(file_names))
            elif len(file_names) < 1:
                raise FileNotFoundError("Invalid Artifact utag {} for artifact {}".format(self.version, self.name))

            file_name = file_names[0]
            return file_name
        else:
            if self.xp_state.pull_write_version is None:
                raise RuntimeError("Must derive Artifact '{}' before locating it".format(self.name))
            file_names = Organizer.resolve_location(self.xp_state, self.xp_state.pull_write_version, self.loc)

            return [os.path.abspath(f) for f in file_names]



