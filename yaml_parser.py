#!/usr/bin/env python
from pprint import PrettyPrinter
import yaml
pp = PrettyPrinter(indent=4)
with open("Makefile.yml", 'r') as stream:
    try:
        pp.pprint(yaml.load(stream))
    except yaml.YAMLError as exc:
        print(exc)

