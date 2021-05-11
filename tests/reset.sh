#! /bin/bash

for file in ./training/*; do
	git restore "${file}" 2> /dev/null;
done

