# Neural network for quantization
This work was conducted in 2019 as part of research with ARNI at UCLA.
 
The following files are included:

The final report summarizing the research:
* finalreport.pdf

Two Jupyter notebooks, written for use on Google colab and requiring Google drive mounting access:
* manager.ipynb - Use this to configure and run the network. This has one filepath which indicates where a results text file will be output. This contains a cell with all of the content from manager.py (see below), as well as cells loading the required files for the CIFAR10 pretrained model.
* analysis.ipynb - Use this to analyze the sample result text files, which should be placed in the filepaths indicated.

Five .py files.
* manager.py - This is used to configure and run the network if you are not using notebooks; it is included for convenience and completeness. I only used it for MNIST and minimal CIFAR10 code verification before the actual trial due to resource limitations. If verifying code before a CIFAR10 trial and you don't have the Pytorch Playground pretrained model files, use an encoder like "CIFAR10 Tutorial".
* testandtrain.py - This handles operations for testing and training the models. 
* supermodel.py - This contains all modules and activation functions.
* datasets.py - This contains a loader function which loads and returns the desired dataset and some relevant handler data. If a pretrained model is being used, this will also load that model. If using this locally without the Pytorch Playground pretrained model files, you will need to comment out the "from utee import selector" line at the start.
* tools.py - This contains all other functions used by the network.

Three text files. The first two can be used to try out the analysis notebook and verify the filepath is supported.
* sampleresults1.txt
* sampleresults2.txt
* And this readme