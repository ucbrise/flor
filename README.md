Flor
=====

Build, configure, run, and reproduce experiments with Flor.

## What is Flor?
Flor (formerly known as [Jarvis](https://github.com/ucbrise/jarvis)) is a system with a declarative DSL embedded in python for managing the workflow development phase of the machine learning lifecycle. Flor enables data scientists to describe ML workflows as directed acyclic graphs (DAGs) of *Actions* and *Artifacts*, and to experiment with different configurations by automatically running the workflow many times, varying the configuration. To date, Flor serves as a build system for producing some desired artifact, and serves as a versioning system that enables tracking the evolution of artifacts across multiple runs in support of reproducibility.

## How do I install it?

1. **Clone or download this repository.**
2. **Create a Virtual Environment.**
 * Download anaconda [here](https://www.anaconda.com/download/).
 * Create a new environment using the following command:

 `conda create --name [env_name]`

 * Activate your new environment using: 
 `source activate [env_name]`

 * Once you have activated your environment, run the command: 

 `pip install -r requirements.txt`

 * Be sure that you have GraphViz by running the following command:

 `brew install graphviz`

 *Note: Whenever you work with Flor, make sure that this environment is active. For more information about environments, please read [this guide](https://conda.io/docs/user-guide/tasks/manage-environments.html).*

3. **Installing Ray (Flor Dependency)**

 * Run the following Brew Commands:

 `brew update`
 `brew install cmake pkg-config automake autoconf libtool boost wget`

 *Note: Linux users should use linuxbrew. See installation details in the **Additional Linux Instructions** section.*

 * Run the following Pip Commands:

 `pip install numpy funcsigs click colorama psutil redis flatbuffers cython --ignore-installed six`
`conda install libgcc`
 `pip install git+https://github.com/ray-project/ray.git#subdirectory=python`

 * If the command above fails, then run the following instead:
`pip3 install ray`

4. **Add Flor to Bash Profile**
 * Open Bash Profile using the following MacOS command:
`vim  ~/.bash_profile`

	*Note: linux machines should use `vim ~/.bashrc`*

 * Add the path to your flor directory in your Bash Profile. Here is an example of a command to add:

 `export PYTHONPATH="$PYTHONPATH:/Users/Sona/Documents/Jarvis/flor"`

5. **Installing Ground**

 * Download the zip file containing the latest version of Ground [here](https://github.com/ground-context/ground/releases) and unzip it.
 
 * Create a file called `script.sh` within the unzipped *ground-0.1.2/db* folder. 

 * Inside `script.sh`, add a path to the *ground-0.1.2/db/postgres_setup.py* file followed by `ground ground drop`. Here is an example of the lines to add to `script.sh`:

 `
 #!/bin/bash 
 `
 `
 python3 /Users/Sona/ground-0.1.2/db/postgres_setup.py 
 ground ground drop
 `


 * Open your bash profile using the following MacOS command:

 `vim  ~/.bash_profile`

 *Note: linux machines should use `vim ~/.bashrc`*

  * Add the following command to your Bash Profile:
	`alias startground='bash /Users/Sona/ground-0.1.2/db/myscript.sh && bash /Users/Sona/ground-0.1.2/bin/ground-postgres'`

 * To update your Bash Profile, run the following command:
`source ~/.bash_profile`

	*Note: linux users should use `source ~/.bashrc`*
 * Linux machines may require several more steps to get ground working. See **Additional Linux Instructions**.

* **Installing Grit and Client**

 * Flor requires ground-context in order to operate properly. Go to [Ground Context Repo](https://github.com/ground-context) and download/install the Grit and Client repos. Downloading the zip files and unzipping them is sufficient. 

 * Open Bash Profile using the following MacOS command:
`vim  ~/.bash_profile`

 * Modify the bash_profile to include paths to these directories like the following commands:

 `export PYTHONPATH="$PYTHONPATH:/Users/Sona/Documents/Jarvis/grit-master/python/"`
`export PYTHONPATH="$PYTHONPATH:/Users/Sona/Documents/Jarvis/client-master/python/"`

 * To update your Bash Profile, run the following command:
`source ~/.bash_profile`

* **Starting Ground**

 * In order to use Flor, you will need to start Ground. In your environment, run the following command:
 `startground`


For examples on how to write your own flor workflow, please have a look at:
```
examples/twitter.py -- classic example
examples/plate.py -- multi-trial example
```

Make sure you:
1. Import `flor`
2. Initialize a `flor.Experiment`
2. set the experiment's `groundClient` to 'git'.

Once you build the workflow, call `pull()` on the artifact you want to produce. You can find it in `~/flor.d/`.

If you pass in a non-empty `dict` to `pull` (see `lifted_twitter.py`), the call will return a pandas dataframe with literals and requested artifacts for the columns, and different trials for the rows.

## Additional Linux Instructions
 * **Installing linuxbrew**
 	* Use the command:
 		`sudo apt install linuxbrew-wrapper`
 	* After installation, follow the additional steps to make sure dependencies are installed and brew is added to path.

 		`Sudo apt-get install build-essential
		Echo ‘export PATH=”/home/linuxbrew/.linuxbrew/bin:$PATH”’ >>~/.bash_profile
		Echo ‘export MANPATH=”/home/linuxbrew/.linuxbrew/share/man:$MANPATH”’ >>~/.bash_profile
		Echo ‘export INFOPATH=”/home/linuxbrew/.linuxbrew/share/info:$INFOPATH”’ >>~/.bash_profile
		PATH=”/home/linuxbrew/.linuxbrew/bin:$PATH”
		Brew install gcc`
 * **Bash Profile**
 	* Where specified, linux machines should use `~/.bashrc` instead of `~/.bash_profile`

 * **Additional Ground Instructions**
 	* Postgres is needed for ground to run. Use the following command to download and install the latest version of postgres.
 		`sudo apt install postgresql`

 	* Open `pg_hda.conf`, which can be located in `/etc/postgresql/<version>/main/`.
 		* under the section `"local" is for Unix domain socket connections only`, set METHOD to `trust`

 		* under IPv4 local connections, add an entry that looks like:
 			`host      ground     ground    127.0.0.1/32    trust`

 		* comment out all lines below the section 
 		`Allow replication connections from localhost...`

 	* Create the `ground` user and `ground` database.
 		* Create the user by using the command 
 		`sudo -u postgres createuser ground`

 		* Create a database by opening postgres with 
 		`sudo -i -u postgres` 
 		and issuing the command 
 		`createdb ground`
 	* You should now be able to enter the `ground-0.1.2/db` directory and issue the `startground` command.

## Note on data

The dataset used in some of our examples has [migrated](https://drive.google.com/drive/folders/1kKtBETmx0bY2_mT9M6PlyPgvGYzBz-sn?usp=sharing).

## Example program
Contents of the `examples/plate.py` file:
```python
import flor

with flor.Experiment('plate_demo') as ex:

	ex.groundClient('ground')

	ones = ex.literal([1, 2, 3], "ones")
	ones.forEach()

	tens = ex.literal([10, 100], "tens")
	tens.forEach()

	@flor.func
	def multiply(x, y):
	    z = x*y
	    print(z)
	    return z

	doMultiply = ex.action(multiply, [ones, tens])
	product = ex.artifact('product.txt', doMultiply)

product.pull()
product.plot()
```
On run produces:
```shell
10
20
30
100
200
300
```

## Motivation
Flor should facilitate the development of auditable, reproducible, justifiable, and reusable data science workflows. Is the data scientist building the right thing? We want to encourage discipline and best practices in ML workflow development by making dependencies explicit, while improving the productivity of adopters by automating multiple runs of the workflow under different configurations. 

## Features
* **Simple and Expressive Object Model**:  The Flor object model consists only of *Actions*, *Artifacts*, and *Literals*. These are connected to form dataflow graphs.
* **Data-Centric Workflows**: Machine learning applications have data dependencies that obscure traditional abstraction boundaries. So, the data "gets everywhere": in the models, and the applications that consume them. It makes sense to think about the data carefully and specifically. In Flor, data is a first-class citizen.
* **Artifact Versioning**: Flor uses git to automatically version every Artifact (data, code, etc.) and Literal that is in a Flor workflow. 
* **Artifact Contextualization**: Flor uses [Ground](http://www.ground-context.org/) to store data about the context of *Artifacts*: their relationships, their lineage. Ground and git are complementary services used by Flor. Together, they enable experiment reproduction and replication. 
* **Parallel Multi-Trial Experiments**: Flor should enable data scientists to try more ideas quickly. For this, we need to enhance speed of execution. We leverage parallel execution systems such as [Ray](https://github.com/ray-project/ray) to execute multiple trials in parallel.
* **Visualization and Exploratory Data Analysis**: To establish the fitness of data for some particular purpose, or gain valuable insights about properties of the data, Flor will leverage visualization techniques in an interactive environment such as Jupyter Notebook. We use visualization for its ability to give immediate feedback and guide the creative process.


## License
Flor is licensed under the [Apache v2 License](https://www.apache.org/licenses/LICENSE-2.0).
