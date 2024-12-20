# General imports
import os
import shutil
import pkg_resources
import subprocess
import math
import textwrap
import pymol
from pymol import cmd
from Bio.PDB import PDBParser
import math
import numpy as np
from requests import get


class Martini:
    """
    A class to create a Martini project for a protein.

    Attributes:
        pdb_file (str): The path to the PDB file.
        project_name (str): The name of the project.
        path_ff (str): The path to the force field files.
        path_input_models (str): The path to the input models directory.
        path_output_models (str): The path to the output models directory.
        path_scripts (str): The path to the scripts directory.
        cg_model_name (str): The name of the coarse-grained model file.
        residues_per_chain (dict): The number of residues per chain in the PDB file.

    Returns:
        A Martini object.

    Methods:
        setProteinCGModel(residue_orientation, strength_conf, overwrite): Sets the coarse-grained (CG) model for the protein.
        setSolventCGModel(ion_molarity, membrane, box_dimensions, z_membrane_shift, charge, overwrite): Sets the coarse-grained (CG) model for the solvent.
        setUpMartiniSimulation(queue, ntasks, gpus, cpus, temperature, replicas, trajectory_checkpoints, simulation_time): Sets up the Martini simulation.
    """

    def __init__(self, pdb_file, project_name: str | None) -> None:
        """
        Initializes a Martini object.

        Parameters:
            pdb_file (str): The path to the PDB file.
            project_name (str | None): The name of the project. If None, the project name will be derived from the PDB file name.
        """

        def _get_number_residues_per_chain():
            """
            Get the number of residues per chain in the PDB file.
            """

            # Load the PDB structure
            parser = PDBParser()
            structure = parser.get_structure("structure", self.pdb_file)

            residues_per_chain = {}

            # Iterate over all chains and count the residues
            for model in structure:
                for i, chain in enumerate(model):
                    residues_per_chain[int(i)] = len(list(chain.get_residues()))

            return residues_per_chain

        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"File not found: {pdb_file}")

        self.pdb_file: str = pdb_file
        self.working_dir: str = os.path.dirname(pdb_file)
        self.base_pdb: str = os.path.basename(pdb_file).split(".")[0]
        self.pdb_oriented: str = f"{self.base_pdb}_oriented.pdb"

        if project_name is None:
            self.project_name: str = self.base_pdb
        else:
            self.project_name: str = project_name

        self.path_ff: str = f"{self.project_name}/FF"
        self.path_input_models: str = f"{self.project_name}/input_models"
        self.path_output_models: str = f"{self.project_name}/output_models"
        self.path_scripts: str = f"{self.project_name}/scripts"
        self.cg_model_name: str = ""
        self.residues_per_chain: dict = _get_number_residues_per_chain()
        self.begin_end_residue_per_chain: dict = {}

    def _deindent_file(self, input_file: str, output_file: str):
        """
        Removes indentation from lines in the input file and writes the modified lines to the output file.

        Parameters:
        - input_file (str): The path to the input file.
        - output_file (str): The path to the output file.
        """

        with open(input_file, "r") as infile:
            lines = infile.readlines()  # Read all lines from the input file

        # Remove indentation for lines with indentation
        dedented_lines = [
            line.lstrip() if line.startswith((" ", "\t")) else line for line in lines
        ]

        with open(output_file, "w") as outfile:
            # Write the modified lines to the output file
            outfile.writelines(dedented_lines)

    def _createFolderTree(self) -> None:
        """
        Creates the necessary folder tree for the Martini project.
        This function creates the following directories:
        - The project directory with the name specified in `self.project_name`.
        - The `ff` directory inside the project directory, which contains the Martini force field files.
        - The `scripts` directory inside the project directory, which contains additional scripts.
        - The `input_models` directory, which is used to store input models.
        The function also copies the contents of the `ff` and `scripts` directories from the installed package
        to the corresponding directories in the project directory. Additionally, it copies the `pdb_file` to
        the `input_models` directory.
        """

        # Get the paths to the 'ff' and 'scripts' directories from the installed package
        ff_path = pkg_resources.resource_filename(__name__, "ff")
        scripts_path = pkg_resources.resource_filename(__name__, "scripts")

        os.makedirs(self.project_name, exist_ok=True)
        os.makedirs(self.path_ff, exist_ok=True)
        os.makedirs(self.path_input_models, exist_ok=True)

        shutil.copytree(ff_path, f"{self.path_ff}/martini", dirs_exist_ok=True)
        shutil.copytree(
            scripts_path, f"{self.project_name}/scripts", dirs_exist_ok=True
        )
        shutil.copy(self.pdb_file, self.path_input_models)

    def setProteinCGModel(
        self,
        residue_orientation: list[tuple[str, int]] | None = None,
        strength_conf: float = 700,
        strength_el: float = 0.5,
        overwrite: bool = False,
        maxwarn: int = 20,
        verbose: bool = True,
    ) -> None:
        """
        This method prepares and sets up a coarse-grained model for a protein using the MARTINI force field.
        It optionally orients the protein structure based on specified residues and generates the CG model file.

            residue_orientation (list[tuple[str, int]] | None, optional):
                A list of tuples specifying the chain and residue numbers to orient the structure.
                If None, the first and last residues of the first chain are used. Defaults to None.
            strength_conf (float, optional):
                The strength of the elastic network used for the CG model. Defaults to 700.
            strength_el (float, optional):
                Elastic bond lower cutoff: F = Fc if rij < lo (default: 0.5)
            overwrite (bool, optional):
                Whether to overwrite the existing CG model file if it already exists. Defaults to False.
            maxwarn (int, optional):
                The maximum number of warnings allowed during the martinize2 process. Defaults to 20.
            verbose (bool, optional):
                Whether to print detailed information during the process. Defaults to True.

        Raises:
            Exception: If the residues selected for orientation are not exactly two.
            Exception: If the distance between the selected residues is too small.

        Notes:
            - This method uses Biopython to parse PDB files and PyMOL for molecular visualization and manipulation.
            - The martinize2 tool is used to generate the CG model.
            - Ensure that the required dependencies (Biopython, PyMOL, martinize2) are installed and properly configured.
        """

        def _find_first_and_last_residue() -> list[tuple[str, int]]:
            """
            Reads the PDB file and extracts the first and last residue from the first chain using Biopython.

            Returns:
                list[tuple[str, int]]: A list containing a tuple of the first and last residue with chain and residue number.
            """
            # Parse the PDB file
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("structure", self.pdb_file)

            # Get the first model (some PDB files have multiple models)
            model = structure[0]

            # Initialize variables to hold the first and last residues
            first_residue = None
            last_residue = None

            # Iterate through chains and residues to get the first chain and its residues
            for chain in model:
                # Get all residues from the first chain
                residues = list(chain.get_residues())

                # Check if the chain has any residues
                if len(residues) > 0:
                    # Get the first and last residue numbers
                    first_residue = (str(chain.id), int(residues[0].id[1]))
                    last_residue = (str(chain.id), int(residues[-1].id[1]))

                # Break after processing the first chain
                break

            return [first_residue, last_residue]

        def _orient_residues(residues_list: list[tuple[str, int]]) -> None:
            """
            Load a molecular structure, select the alpha carbons (CA) of specific residues,
            orient the structure around those residues, and save the result to a file.

            Parameters:
            residues_list (list): List of tuples containing chain and residue number, e.g., [('A', 32), ('B', 4)].
            """

            # Initialize PyMOL
            pymol.finish_launching(["pymol", "-qc"])
            cmd.load(self.pdb_file)

            # Create the selection string for alpha carbons of the specified residues
            selection = " or ".join(
                [
                    f"chain {chain} and resi {resi} and name CA"
                    for chain, resi in residues_list
                ]
            )
            cmd.select("z_selection", selection)

            # Get the coordinates of the selected alpha carbons
            model = cmd.get_model("z_selection")

            if len(model.atom) != 2:
                pymol.cmd.quit()
                raise Exception("RotationError: You must select exactly two residues.")

            # Get the coordinates of the two selected alpha carbons
            coord1 = model.atom[0].coord
            coord2 = model.atom[1].coord

            # Calculate the midpoint between the two alpha carbons
            midpoint = [(coord1[i] + coord2[i]) / 2 for i in range(3)]

            # Calculate the vector between the two alpha carbons
            vector = [coord2[i] - coord1[i] for i in range(3)]

            # Check if the distance between the two alpha carbons is too small
            length = (sum([v**2 for v in vector])) ** 0.5
            if length < 1e-3:
                pymol.cmd.quit()
                raise Exception(
                    "RotationError: The two residues are too close to each other."
                )

            # Normalize the vector
            unit_vector = [v / length for v in vector]

            # Find the angle between the vector and the z-axis (dot product with [0, 0, 1])
            dot_product = unit_vector[
                2
            ]  # Dot product with z-axis unit vector [0, 0, 1]
            dot_product = max(
                min(dot_product, 1.0), -1.0
            )  # Clamp to avoid domain errors
            angle = math.acos(dot_product) * 180 / math.pi  # Angle in degrees

            # Skip rotation if the vector is already aligned with the z-axis
            if abs(angle) < 1e-3:
                print("The structure is already aligned with the z-axis.")
            else:
                # Compute the axis of rotation (cross product of the vector and z-axis)
                axis = np.cross(unit_vector, [0, 0, 1]).tolist()

                # If the cross product is near zero, no rotation is needed
                if np.linalg.norm(axis) < 1e-6:
                    print(
                        "The vector is already aligned or anti-aligned with the z-axis."
                    )
                else:
                    # Move the selection to the origin (translate to midpoint)
                    translation_vector = [-midpoint[i] for i in range(3)]
                    cmd.translate(translation_vector, "all")

                    # Apply the rotation to align the vector with the z-axis
                    cmd.rotate(axis, angle, "all")

            # Save the newly oriented structure
            cmd.save(self.pdb_oriented)

            # Clean up and exit PyMOL
            cmd.delete("all")
            pymol.cmd.quit()

        # Create folder tree if it does not exist
        if not os.path.exists(self.project_name):
            self._createFolderTree()

        original_dir = os.getcwd()
        cg_pdb_path = f"{self.path_input_models}/{self.base_pdb}_cg.pdb"

        # Check if the file exists and if it should be overwritten
        if not os.path.exists(cg_pdb_path) or overwrite:
            # Change directory to path_input_models
            os.chdir(self.path_input_models)

            if residue_orientation is None:
                residue_orientation = _find_first_and_last_residue()
                if verbose:
                    print(
                        f"Residue orientation not provided. Using the first and last residues: {residue_orientation} to orient in the z-axis. Please consider passing a list with the residues to be oriented."
                    )
            else:
                if verbose:
                    print(
                        f"Residues used to orient in the z-axis: {residue_orientation}."
                    )

            _orient_residues(residue_orientation)

            # Prepare the command as a list of arguments
            command = [
                "martinize2",
                "-f",
                f"{self.pdb_oriented}",
                "-x",
                f"{self.base_pdb}_cg.pdb",
                "-o",
                "topol.top",
                "-ff",
                "martini3001",
                "-scfix",
                "-cys",
                "auto",
                "-p",
                "backbone",
                "-elastic",
                "-ef",
                str(strength_conf),
                "-el",
                f"{str(strength_el)}",
                "-eu",
                "0.9",
                "-maxwarn",
                str(maxwarn),
            ]

            # Run the command using subprocess
            if verbose:
                print(f"Running martinize2 with command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)

            # Check if the command was successful
            if result.returncode != 0:
                print(f"Error running martinize2: {result.stderr}")
                print(f"Output martinize2: {result.stdout}")

        else:
            if verbose:
                print(
                    f"CG model file '{cg_pdb_path}' already exists. Use 'overwrite=True' to regenerate the CG model."
                )

        # Set the name of the coarse-grained model file
        self.cg_model_name = f"{self.base_pdb}_cg.pdb"

        os.chdir(original_dir)

    def setSolventCGModel(
        self,
        ion_molarity: float = 0.15,
        membrane: bool = False,
        box_dimensions: list = [20, 20, 10],
        z_membrane_shift: float = 0,
        charge: int = 0,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Sets the coarse-grained (CG) model for the solvent.

        Parameters:
            ion_molarity (float, optional): The salt concentration in molarity. Defaults to 0.15.
            membrane (bool, optional): Whether a membrane is present. Defaults to False.
            box_dimensions (list, optional): The box dimensions for the CG model. Defaults to [20, 20, 10].
            overwrite (bool, optional): Whether to overwrite the existing CG model file if it already exists. Defaults to False.
        """

        # Define the path
        cg_model_path = os.path.join(self.path_input_models, "system.gro")
        original_dir = os.getcwd()

        # Check if the file exists and if overwrite is False
        if os.path.isfile(cg_model_path) and not overwrite:
            print(
                f"CG model file '{cg_model_path}' already exists. Use 'overwrite=True' to regenerate the CG model."
            )
            return

        # Proceed to generate a new CG model file if overwrite is True or the file does not exist.
        print(f"Generating a new CG model at '{cg_model_path}'...")

        # Change directory to path_input_models
        os.chdir(self.path_input_models)

        # Prepare common command parts
        command = [
            "insane",
            "-f",
            self.cg_model_name,
            "-o",
            "system.gro",
            "-p",
            "system.top",
            "-pbc",
            "square",
            "-box",
            f"{box_dimensions[0]},{box_dimensions[1]},{box_dimensions[2]}",
            "-sol",
            "W",
            "-salt",
            str(ion_molarity),
        ]

        # Add membrane options
        if membrane:
            command += [
                "-u",
                "POPC",
                "-l",
                "POPC",
                "-center",
                "-dm",
                str(z_membrane_shift),
            ]

        if charge != 0:
            command += ["-charge", str(charge)]

            if verbose:
                print(
                    "If the protein is not centered in the z-axis, consider using the z_membrane_shift parameter (with overwrite=True) to shift the membrane."
                )

        # Run the command using subprocess
        if verbose:
            print(f"Running insane with command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode != 0:
            print(f"Error running insane: {result.stderr}")
            print(f"Output insane {result.stdout}")

        if verbose:
            print("\n\n -- WARNING -- \n\n")
            print(
                "Insane might not capture correctly the charge of the system.\nIf the GROMACS simulation fails because of this, rerun this\nmethod with the parameter charge adjusted to the .err file\nfrom GROMACS."
            )
            print(
                "Make sure to make the simulation box suffienlty large. Otherwise you might encounter that the largest distance between excluded atoms is larger than 1.4."
            )

        # Set the name of the coarse-grained system file
        self.cg_system_name = "system.gro"
        os.chdir(original_dir)

    def setUpMartiniSimulation(
        self,
        queue: str = "acc_debug",
        ntasks: int = 1,
        gpus: int = 1,
        cpus: int = 20,
        temperature: float = 298.15,
        replicas: int = 4,
        trajectory_checkpoints: int = 10,
        simulation_time: int = 10000,
        time_step: float = 0.02,
        pymol_visualization: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Sets up the Martini simulation by performing the following steps:
        1. Modifies the topology files by replacing a specific include line with a new block of includes.
        2. Generates Gromacs (Gmx) files using the 'gmx make_ndx' and 'gmx genrestr' commands.
        3. Modifies the Gromacs (Gmx) script files based on the given simulation parameters.
        4. Generates the output directory tree for the Martini project.
        5. Generates the run file for the Martini project.

        Parameters:
        - queue (str): The queue to submit the job to. Default is 'acc_debug'.
        - ntasks (int): The number of tasks to run. Default is 1.
        - gpus (int): The number of GPUs to use. Default is 1.
        - cpus (int): The number of CPUs per task. Default is 20.
        - replicas (int): The number of replicas to run. Default is 4.
        - trajectory_checkpoints (int): The number of trajectory checkpoints. Default is 10.
        - simulation_time (int): The total simulation time in picoseconds (ns). Default is 10000 ns.
        - temperature (float): The temperature of the simulation in Kelvin (K). Default is 298.15 K.
        """

        def _modifyTopologyFiles():
            """
            Modifies the topology files by replacing a specific include line with a new block of includes.
            This function performs the following steps:

            1. Reads the content of the original file specified by `file_path`.
            2. Replaces the specific include line with the new block of includes.
            3. Writes the updated content back to the file specified by `file_path`.
            4. Renames the original file to ".system.top.bak".
            5. Renames the updated file to "system.top".
            6. Removes the file "topology.top".
            Note: This function assumes that the current working directory is set to `self.path_input_models`.
            """

            number_of_molecules = len(
                [
                    file
                    for file in os.listdir(self.path_input_models)
                    if file.endswith(".itp")
                ]
            )

            header = """#include "../FF/martini/martini_v3.0.0.itp"
            #include "../FF/martini/martini_v3.0.0_ions_v1.itp"
            #include "../FF/martini/martini_v3.0.0_phospholipids_v1.itp"
            #include "../FF/martini/martini_v3.0.0_solvents_v1.itp"\n
            """

            protein_substitution = ""
            for i in range(0, number_of_molecules):
                header += f"""#include "molecule_{i}.itp"
                """
                if i == number_of_molecules - 1:
                    protein_substitution += f"molecule_{i}       1"
                else:
                    protein_substitution += f"molecule_{i}       1\n"

            # Path to the file you want to modify
            original_dir = os.getcwd()
            os.chdir(self.path_input_models)
            file_path = "system.top"

            # Read the content of the original file
            with open(file_path, "r") as file:
                content = file.read()

            # Replace the specific include line with the new block
            updated_content_nb = content.replace('#include "martini.itp"', header)
            updated_content_mol = updated_content_nb.replace(
                "Protein          1", protein_substitution
            )
            updated_content_na = updated_content_mol.replace("NA+", "NA ")
            updated_content = updated_content_na.replace("CL-", "CL ")

            # Write the updated content back to the file (or a new file)
            with open("system.top.tmp", "w") as updated_file:
                updated_file.write(updated_content)

            self._deindent_file("system.top.tmp", "system.top.tmp")

            os.rename("system.top", ".system.top.bak")
            os.rename("system.top.tmp", "system.top")
            if os.path.exists("topol.top"):
                os.remove("topol.top")

            os.chdir(original_dir)

            if verbose:
                print(
                    f"Modified the topology file (system.top) in the directory: {self.path_input_models}"
                )

        def _generateGmxFiles():
            """
            Generates Gromacs (Gmx) files using the 'gmx make_ndx' and 'gmx genrestr' commands.
            """

            # Change directory to path_input_models
            original_dir = os.getcwd()
            os.chdir(self.path_input_models)

            # Command to run 'gmx make_ndx'
            make_ndx_command = [
                "gmx",
                "make_ndx",
                "-f",
                "system.gro",
                "-o",
                "index.ndx",
            ]

            # Piped input for 'gmx make_ndx'
            make_ndx_input = "1 | 13\n17 | 14\nq\n"

            # Run 'gmx make_ndx' with input provided through stdin using Popen
            process_make_ndx = subprocess.Popen(
                make_ndx_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process_make_ndx.communicate(input=make_ndx_input)

            # Check if 'gmx make_ndx' command was successful
            if process_make_ndx.returncode != 0:
                print(f"Error running gmx make_ndx: {stderr}")
            else:
                print(f"gmx make_ndx ran successfully: {stdout}")

            # Return to the original directory
            os.chdir(original_dir)

        def _modifyGmxScripts(
            temperature: float,
            trajectory_checkpoints: int,
            simulation_time: int,
            time_step: float,
        ) -> None:
            """
            Modifies the Gromacs (Gmx) script files in the specified directory based on the given simulation parameters.

            This function performs the following steps:
            1. Modifies the simulation time in the 'md' script file based on the given simulation parameters.
            2. Modifies the simulation temperature in the .mdp files found in the specified directory.
            """

            def _modify_simulation_time(
                simulation_time, time_step, trajectory_checkpoints
            ):
                """
                Modifies the simulation time in the 'md' script file based on the given simulation parameters.
                The function performs the following steps:
                1. Calculate the number of steps based on the given logic.
                2. Find the 'md' script file in the specified directory.
                3. Read the content of the file.
                4. Modify the 'nsteps' line.
                5. Write the modified content back to the file.
                """

                # Calculate number_of_steps based on the given logic
                simulation_time_in_pico = simulation_time * 1000
                value_to_assess = simulation_time_in_pico / (
                    trajectory_checkpoints * time_step
                )

                # Check if the division is exact
                if simulation_time_in_pico % trajectory_checkpoints == 0:
                    number_of_steps = int(value_to_assess)
                else:
                    number_of_steps = math.trunc(value_to_assess) + 1
                    print(
                        f"The total simulation time will be: {number_of_steps * (1000*trajectory_checkpoints)}ps"
                    )

                # Find the md script file in the specified directory
                md_file = [
                    x for x in os.listdir(self.path_scripts) if x.startswith("md")
                ][0]
                path_to_md = os.path.join(self.path_scripts, md_file)

                # Read the content of the file
                with open(path_to_md, "r") as file:
                    lines = file.readlines()

                # Modify the 'nsteps' line
                for i, line in enumerate(lines):
                    if line.startswith("nsteps"):
                        # Replace the nsteps value with the new number_of_steps
                        lines[i] = (
                            f"nsteps                  = {number_of_steps}     ; {simulation_time}ns\n"
                        )
                    elif line.startswith("dt"):
                        lines[i] = (
                            f"dt                      = {time_step}          ; {time_step*1000} fs\n"
                        )

                # Write the modified content back to the file
                with open(path_to_md, "w") as file:
                    file.writelines(lines)

            def _modify_simulation_temperature(temperature):
                """
                Modifies the simulation temperature in the .mdp files found in the specified directory.

                Parameters:
                temperature (float): The new temperature value to set in the .mdp files in Kelvin (K).
                """

                # Find the .mdp files in the specified directory
                files_to_modify = [
                    os.path.join(self.path_scripts, x)
                    for x in os.listdir(self.path_scripts)
                    if x.endswith(".mdp")
                ]

                for file in files_to_modify:
                    # Generate a temporary file path for the modified file
                    new_file_path = file.split(".")[0] + "_tmp.mdp"

                    # Read the content of the original file
                    with open(file, "r") as original_file:
                        lines = original_file.readlines()

                    # Modify the 'ref_t' line
                    for i, line in enumerate(lines):
                        if line.startswith("ref_t"):
                            # Replace the ref_t value with the new temperature
                            lines[i] = (
                                f"ref_t                           = {temperature}          {temperature}        ; reference temperature, one for each group, in K\n"
                            )
                            break  # Exit the loop once we've made the modification

                    # Write the modified content to the new temporary file
                    with open(new_file_path, "w") as new_file:
                        new_file.writelines(lines)

                    # Rename the original file to a backup file (with .bak extension)
                    bak_file_path = f"{file}.bak"
                    os.rename(file, bak_file_path)

                    # Rename the new temporary file to the original filename
                    os.rename(new_file_path, file)

            # Modify the simulation time
            _modify_simulation_time(simulation_time, time_step, trajectory_checkpoints)
            # Modify the simulation temperature
            _modify_simulation_temperature(temperature)
            if verbose:
                print(
                    f"Modified the Gmx script files in the directory: {self.path_scripts}"
                )

        def _generateOuptutDirectoryTree(replicas: int):
            """
            Generates the output directory tree for the Martini project.
            This function creates the following directories:
            - The `output_models` directory, which is used to store output models.
            - Inside the `output_models` directory, it creates subdirectories for each replica.
            """

            os.makedirs(self.path_output_models, exist_ok=True)

            for i in range(replicas):
                os.makedirs(f"{self.path_output_models}/{i}", exist_ok=True)

        def _generateRunFile(
            queue: str,
            ntasks: int,
            gpus: int,
            cpus: int,
            replicas: int,
            trajectory_checkpoints: int,
        ):
            """
            Generate a run file for a job submission system.

            Args:
                queue (str): The queue to submit the job to. Should be either 'acc_debug' or 'acc_bscls'.
                ntasks (int): The number of tasks to allocate for the job.
                gpus (int): The number of GPUs to allocate for the job.
                cpus (int): The number of CPUs per task to allocate for the job.
                replicas (int): The number of replicas to generate in the main body of the run file.
                trajectory_checkpoints (int): The number of trajectory checkpoints to generate for each replica.
            """

            if verbose:
                print(f"Generating the run file...")

            if queue not in ["acc_debug", "acc_bscls"]:
                raise Exception(
                    "InvalidQueue: this value should either be acc_debug or acc_bscls."
                )
            else:
                if queue == "acc_debug":
                    time = "02:00:00"
                elif queue == "acc_bscls":
                    time = "48:00:00"

            # Dedent the header section to remove all initial indentation
            header = textwrap.dedent(
                f"""#!/bin/bash
                #SBATCH --job-name={self.project_name}
                #SBATCH --qos={queue}
                #SBATCH --time={time}
                #SBATCH --ntasks {ntasks}
                #SBATCH --gres gpu:{gpus}
                #SBATCH --account=bsc72
                #SBATCH --cpus-per-task {cpus}
                #SBATCH --array=1-{replicas}
                #SBATCH --output={self.project_name}_%a_%A.out
                #SBATCH --error={self.project_name}_%a_%A.err

                module load cuda
                module load nvidia-hpc-sdk/23.11
                module load gromacs/2023.3

                export SRUN_CPUS_PER_TASK=${{SLURM_CPUS_PER_TASK}}
                export SLURM_CPU_BIND=none
                export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
                export GMX_ENABLE_DIRECT_GPU_COMM=1
                export GMX_GPU_PME_DECOMPOSITION=1

                GMXBIN="mpirun --bind-to none -report-bindings gmx_mpi"
            """
            )

            main_body = """"""

            for i in range(1, replicas + 1):

                # Generate trajectory checkpoints and dedent each generated part
                trajectory_checks = textwrap.dedent(
                    f"""
                    ${{GMXBIN}} grompp -f md.mdp -c ../npt/npt3.gro -r ../npt/npt3.gro -p ../../../input_models/system.top -o prot_md_1.tpr -n ../../../input_models/index.ndx -maxwarn 1
                    ${{GMXBIN}} mdrun -v -deffnm prot_md_1
                """
                )

                # NPT variable restraints
                restraints = "cd ../../../input_models\n"
                for k in self.residues_per_chain.keys():
                    restraints += (
                        f"sed -i '/\\[ position_restraints \\]/,/#endif/ "
                        f"{{s/\\([0-9]\\+[ \\t]\\+[0-9]\\+[ \\t]\\+\\)1000 1000 1000/\\1500 500 500/g}}' "
                        f"molecule_{k}.itp\n"
                    )

                restraints += f"cd ../output_models/{i-1}/npt"

                # MD checkpoints
                for j in range(1, trajectory_checkpoints):
                    trajectory_checks += textwrap.dedent(
                        f"""
                    ${{GMXBIN}} grompp -f md.mdp -c prot_md_{j}.gro -t prot_md_{j}.cpt -p ../../../input_models/system.top -o prot_md_{j+1}.tpr -n ../../../input_models/index.ndx -maxwarn 1
                    ${{GMXBIN}} mdrun -v -deffnm prot_md_{j+1}
                    """
                    )

                # Add the rest of the main body, dedenting each section
                main_body += textwrap.dedent(
                    f"""

                    # ----------------------------------------------------------  {i}  ------------------------------------------------------------

                    if [[ $SLURM_ARRAY_TASK_ID = {i} ]]; then
                    cd {self.project_name}/output_models/{i-1}

                    #
                    # EM
                    #
                    mkdir -p em
                    cp ../../scripts/em.mdp em
                    cd em
                    ${{GMXBIN}} grompp -f em.mdp -c ../../../input_models/system.gro -p ../../../input_models/system.top -o em.tpr -v -maxwarn 1
                    ${{GMXBIN}} mdrun -v -deffnm em
                    cd ..

                    #
                    # NVT
                    #
                    mkdir -p nvt
                    cp ../../scripts/nvt*.mdp nvt
                    cd nvt

                    # time step = 0.001 ps
                    ${{GMXBIN}} grompp -f nvt1.mdp -c ../em/em.gro -r ../em/em.gro -p ../../../input_models/system.top -o nvt1.tpr -n ../../../input_models/index.ndx -maxwarn 1
                    ${{GMXBIN}} mdrun -v -deffnm nvt1

                    # time step = 0.002 ps
                    ${{GMXBIN}} grompp -f nvt2.mdp -c nvt1.gro -r nvt1.gro -p ../../../input_models/system.top -o nvt2.tpr -n ../../../input_models/index.ndx -maxwarn 1
                    ${{GMXBIN}} mdrun -v -deffnm nvt2

                    # time step = 0.004 ps
                    ${{GMXBIN}} grompp -f nvt3.mdp -c nvt2.gro -r nvt2.gro -p ../../../input_models/system.top -o nvt3.tpr -n ../../../input_models/index.ndx -maxwarn 1
                    ${{GMXBIN}} mdrun -v -deffnm nvt3

                    # time step = 0.01 ps
                    ${{GMXBIN}} grompp -f nvt4.mdp -c nvt3.gro -r nvt3.gro -p ../../../input_models/system.top -o nvt4.tpr -n ../../../input_models/index.ndx -maxwarn 1
                    ${{GMXBIN}} mdrun -v -deffnm nvt4
                    cd ..

                    #
                    # NPT
                    #
                    mkdir -p npt
                    cp ../../scripts/npt.mdp npt
                    cd npt

                    # restraint 1000
                    ${{GMXBIN}} grompp -f npt.mdp -c ../nvt/nvt4.gro -r ../nvt/nvt4.gro -p ../../../input_models/system.top -o npt1.tpr -n ../../../input_models/index.ndx -maxwarn 1
                    ${{GMXBIN}} mdrun -v -deffnm npt1

                    # restraint 500
                    {restraints}

                    ${{GMXBIN}} grompp -f npt.mdp -c npt1.gro -r npt1.gro -p ../../../input_models/system.top -o npt2.tpr -n ../../../input_models/index.ndx -maxwarn 1
                    ${{GMXBIN}} mdrun -v -deffnm npt2

                    # restraint 0
                    {restraints.replace("500", "0").replace("1000","500")}

                    ${{GMXBIN}} grompp -f npt.mdp -c npt2.gro -r npt2.gro -p ../../../input_models/system.top -o npt3.tpr -n ../../../input_models/index.ndx -maxwarn 1
                    ${{GMXBIN}} mdrun -v -deffnm npt3
                    cd ..

                    #
                    # MD
                    #
                    mkdir -p md
                    cp ../../scripts/md.mdp md
                    cd md
                    {trajectory_checks}
                    cd ..
                    fi
                """
                )

            # Combine header and main body, write the output to slurm_array.sh
            file_text = header + main_body

            with open("slurm_array.sh", "w") as file:
                file.write(file_text)

            # Dedent the file to remove all initial indentation
            self._deindent_file("slurm_array.sh", "slurm_array.sh")

        def _generatePymolVisualization() -> None:
            """
            Generates a PyMOL visualization script and saves it to a file.
            The script loads a molecular system, hides all objects, shows spheres,
            and selects specific non-protein residues with transparency settings.
            """

            # Define the path where the file will be saved
            path_pymol_visualization = os.path.join(
                self.path_input_models, "cg_viz.pml"
            )

            # PyMOL commands to write to the file
            pymol_commands = [
                "load system.gro",
                "hide all",
                "show spheres",
                r"select non_protein, resn NA\++CL-+POPC+W",  # Escape '+' for proper PyMOL selection
                "set sphere_transparency, 0.8, non_protein",
            ]

            # Open the file in write mode and write the PyMOL commands
            with open(path_pymol_visualization, "w") as file:
                for command in pymol_commands:
                    file.write(f"{command}\n")

            print(
                f"PyMOL visualization script saved to {path_pymol_visualization}.\nIt is recommended to look at the system before launching simulations.\n\t$> pymol cg_viz.pml"
            )

        _modifyTopologyFiles()  # Modify the topology file
        _generateGmxFiles()  # Generate Gmx files
        _modifyGmxScripts(
            temperature, trajectory_checkpoints, simulation_time, time_step
        )  # Modify the Gmx scripts
        _generateOuptutDirectoryTree(replicas)  # Generate the output directory tree
        _generateRunFile(
            queue, ntasks, gpus, cpus, replicas, trajectory_checkpoints
        )  # Generate the run file

        if pymol_visualization:
            _generatePymolVisualization()  # Generate the PyMOL visualization script
