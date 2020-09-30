The sources in this module are from the PADME and DeepChem projects.

The folder `/ext_src` stores the C source code for calculating the number of swapped pairs, which results in a module named `swapped` 
that is called by `cindex_measure.py`.

To construct the module `swapped`, please run `python ./setup.py` with the `jova/metrics` folder as the current folder. This results
in a `.so` file. Please compile the code to get your own version of the `.so` file.
