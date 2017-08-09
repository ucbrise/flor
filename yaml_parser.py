#!/usr/bin/env python
import yaml

with open("Makefile.yml", "r") as stream:
    try:
        makefile = yaml.load(stream)
    except yaml.YAMLError as exc:
        print exc

if "make" not in makefile: 
	raise AssertionError("A hash or dictionary with the name 'make' is not defined.")

# buffer for saving printable strings
print_table = []
pipeline = makefile["make"]

valid_dependency_types = set([
	'tasks',
	'script',
	'models',
	'meta-data',
	'args',
	'recipe',
	'test',
	'data'
	])

used_dependency_types = set([])
for target in pipeline.keys():
	used_dependency_types |= set(pipeline[target].keys())
invalid_dependency_types = used_dependency_types - valid_dependency_types
if invalid_dependency_types:
	raise AssertionError("Use of invalid dependency types: " + str(invalid_dependency_types))

"""
This function takes an Array (possibly of arrays)
and outputs a string, s.t. each element is seperated
by &&

For example:
["rm -f *.pkl *.txt"] => "rm -f *.pkl *.txt"
["source activate py36", ["python $0"], "source deactivate"] 
	=> "source activate py36 && python $0 && source deactivate"

"""
def flatten(recipe):
	for idx, val in enumerate(recipe):
		if isinstance(val, list):
			recipe[idx] = flatten(val)
	return " && ".join(recipe)

"""
Outputs a string of all the dependencies
separated by space.
"""
def get_rhs(dependencies):
	
	if "tasks" in dependencies:
		tasks = " ".join(dependencies["tasks"])
	else:
		tasks = ""
	
	if "meta-data" in dependencies:
		metadata = " ".join(dependencies["meta-data"])
	else:
		metadata = ""
	
	if "models" in dependencies:
		models = " ".join(dependencies["models"])
	else:
		models = ""
	
	if "data" in dependencies:
		data = " ".join(dependencies["data"])
	else:
		data = ""
	
	if "script" in dependencies:
		script = dependencies["script"]
	else:
		script = ""

	return " ".join([elm for elm in [tasks, metadata, models, data, script] if len(elm) > 0])

"""
Outputs the recipe to generate the target.
Substitutes $0 with the script provided
"""
def get_recipe(dependencies):
	
	if "script" in dependencies:
		script = dependencies["script"]
	else:
		script = ""

	if "args" in pipeline[target]:
		args = " ".join(pipeline[target]["args"])
	else:
		args = ""

	dollar0 = " ".join([elm for elm in [script, args] if len(elm) > 0])

	if "recipe" in pipeline[target]:
		recipe = flatten(pipeline[target]["recipe"])
	else:
		recipe = ""


	if "test" in pipeline[target]:
		recipe += " && test -f " + " ".join(pipeline[target]["test"])

	return recipe.replace("$0", dollar0)

# pipeline.keys() is a list of targets to write to the Makefile
for target in pipeline.keys():
	lhs = target
	colon = " : "
	rhs = get_rhs(pipeline[target])
	
	print_table.append(lhs + colon + rhs + "\n")

	recipe = get_recipe(pipeline[target])
	
	if len(recipe) > 0:
		print_table.append("\t" + recipe + "\n")
	
	print_table.append("\n")

if "phony" in makefile:
	print_table.append(".PHONY : " + " ".join(makefile["phony"]) + "\n")

with open("Makefile", "w") as f:
	for line in print_table:
		f.write(line)
