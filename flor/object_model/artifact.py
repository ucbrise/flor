#!/usr/bin/env python3


import os
import pandas as pd
import cloudpickle

from flor.shared_object_model.resource import Resource

class Artifact(Resource):

    def __init__(self, loc: str,
                 parent, name,
                 version: str,
                 identifier: str,
                 xp_state):
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
        return super().__plot__(self.loc, "box", rankdir)

    def pull(self, utag=None):
        super().__pull__(self, utag)

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
        file_name = os.path.basename(file_path)
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

        if self.version is None or self.parent is not None:
            return self.loc
        output_dir = "{}_{}".format(self.xp_state.EXPERIMENT_NAME, self.xp_state.outputDirectory)
        nested_dir = os.path.join(output_dir, self.version)
        file_names = list(filter(lambda s: (self.loc.split('.')[0].split('_')
                          == s.split('.')[0].split('_')[0:-1]), os.listdir(nested_dir)))
        if len(file_names) > 1:
            raise NameError("Ambiguous specification: Which item do you want? Specify via Artifact.identifier: {}"
                            .format(file_names))
        elif len(file_names) < 1:
            raise FileNotFoundError("Invalid Artifact utag {} for artifact {}".format(self.version, self.name))

        file_name = file_names[0]

        return os.path.join(nested_dir, file_name)



