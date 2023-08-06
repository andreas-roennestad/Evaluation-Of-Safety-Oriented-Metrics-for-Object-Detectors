# Evaluation of Safety-Oriented Metrics for Object Detectors
The Python and Jupyter Notebook files provided in this repository implement all methods and functionality applied in 
my Master's Thesis "Evaluation of Safety-Oriented Metrics for Object Detectors".
The thesis was written for the Department of Computer Science (IDI) at the Norwegian University of Science and Technology (NTNU).

Part of this repository builds on the [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit/blob/master/LICENSE.txt), 
which also uses the Apache 2.0 license. Specifically, the directories `metrics_models/eval/` and `metrics_models/utils/` extends the 
corresponding directories of the nuScenes devkit. Furthermore, it extends the modified nuScenes devkit provided in 
[Object criticality model](https://github.com/AndreaCeccarelli/metrics_model), by Andrea Ceccarelli and Leonardo Montecchi.

##  Scope of the repository
The experimental work performed as part of the thesis involved the comparison of two approaches for the 
safety-oriented evaluation of object detectors, namely [PKL](https://github.com/nv-tlabs/planning-centric-metrics) and 
the [Object Criticality Model](https://github.com/AndreaCeccarelli/metrics_model).
For this purpose, functionality for evaluation with the two metrics and the analysis of the resulting data is implemented. Below, a brief description of each of the relevant files and directories provided are given:

./metrics_model/compute_APCRIT-ANONYMIZED.ipynb:
This notebook is arguably the most important one, implementing the evaluation of detector predictions with relevant metrics. This can be edited to include more models or to change the parameters related to this evaluation (including injected faults, plotting the predictions, etc.).

./metrics_model/analyze_samples.ipynb:
This notebook implements a series of different tools for analysing the metric data produced by evaluation. Running this requires the datasets of metric results produced by running compute_APCRIT-ANONYMIZED.ipynb, to be produced.

./metrics_model/save_sample.ipynb:
This notebook is a helper file for generating lists of nuScenes sample tokens of samples whose corresponding metric data over detector predictions will be analysed. While the JSON files found in ./results contain the list of sample tokens used for the experiments of this thesis (for recreation), this allows for creating custom randomly sampled datasets of metric results over the nuScenes datasets. 

./metrics_model/statistical_analysis.ipynb:
This notebook computes and plots correlation coefficients and confidence intervals for constrained datasets of metric results. The data from the experiments of this thesis was recorded by limiting the number of objects considered in each sample by means of filtering metric data in analyze_samples.ipynb.

./metrics_model/visualization_helpers.py:
This Python file implements functions for visualizing detector predictions and their criticalities.

./metrics_model/eval/ and ./metrics_model/utils/:
Represents the modified devkit. These directories replace the corresponding directories in the devkit.

NOTE: The files of ./metrics_model that are excluded in the above are files not in use, either files inherited from the parent repo [Object criticality model](https://github.com/AndreaCeccarelli/metrics_model) or automatically generated files. Additional details of each file can be found in the in-line documentation and comments.

./results: 
This is the directory where datasets of metric results are automatically stored upon creation. Furthermore, the JSON files specifying the samples selected for experiments are stored here. The JSON files present correspond to the samples used in this thesis. 
 not in use, either files inherited from the parent repo [Object criticality model](https://github.com/AndreaCeccarelli/metrics_model) or automatically generated files.

