#!/usr/bin/env python
from pprint import PrettyPrinter
import yaml

with open("Makefile.yml", 'r') as stream:
    try:
        makefile = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

pipeline = makefile["make"]
for target in pipeline.keys():
	lhs = target
	colon = " : "
	if "tasks" in pipeline[target]:
		tasks = " ".join(pipeline[target]["tasks"])
	else:
		tasks = ""
	if "meta-data" in pipeline[target]:
		metadata = " ".join(pipeline[target]["meta-data"])
	else:
		metadata = ""
	if "models" in pipeline[target]:
		models = " ".join(pipeline[target]["models"])
	else:
		models = ""
	if "data" in pipeline[target]:
		data = " ".join(pipeline[target]["data"])
	else:
		data = ""
	if "script" in pipeline[target]:
		script = pipeline[target]["script"]
	else:
		script = ""

	rhs = " ".join([elm for elm in [tasks, metadata, models, data, script] if len(elm) > 0])
	print(lhs + colon + rhs)

	if "args" in pipeline[target]:
		args = " ".join(pipeline[target]["args"])
	else:
		args = ""
	
	dollar0 = " ".join([elm for elm in [script, args] if len(elm) > 0])

	if "recipe" in pipeline[target]:
		if isinstance(pipeline[target]["recipe"], list):	
			recipe = " && ".join(pipeline[target]["recipe"])
		else:
			recipe = pipeline[target]["recipe"]
	else:
		recipe = ""

	recipe = recipe.replace("$0", dollar0)
	if len(recipe) > 0:
		print("\t" + recipe)
	print

print(".PHONY : " + " ".join(makefile["phony"]))
