# Overview
SEnSCA, an innovative framework for unraveling the intricate network of cell-cell communication mediated by ligand-receptor interactions, integrates state-of-the-art machine learning algorithms and a three-point estimation methodology based on single-cell RNA sequencing data. SEnSCA accurately inferred intercellular communication within human melanoma tissues. It is anticipated to dissect cellular crosstalk and signal pathways at single cell resolution.
![Overview_SEnSCA](https://github.com/wxw418/SE/assets/93773332/7fcefaee-5cf5-4bcb-a79f-5670862ae1e0)

# Environment
Install Python 3.9.16 for running this code. And these packages should be satisfied: 
* numpy=1.24.3
* pandas=1.5.3
* scikit-learn=1.2.2
* torch=2.0.1
* matplotlib=3.7.1

# Data
1. Ligand-receptor data is available at [Uniprot](https://www.uniprot.org/), [GEO](https://www.ncbi.nlm.nih.gov/geo/).
2. Feature extraction website at [iFeature](https://ifeature.erc.monash.edu/).

# Usage
* Step 1: Begin by running the code to identify reliable negative samples, with the option to manually adjust the pre value within the range of 0 to 1.
```
python Code/Kmeans.py
```
* Step 2: Execute the code using the default 5-fold cross-validation, and get the ligand-receptor pairs.
```
python Code/SEnSCA.py
```
* Step 3: Implement the **three-point estimation method**, incorporating [cell expression, expression product, specific expression].
(Note: the user-specified database only needs to replace the **LRI.csv** file and the corresponding format in the file.)
* Step 4: Finally, output the strength of cell-cell communication.

# Cell-cell communication tools for comparative analysis
[CellChat](https://github.com/sqjin/CellChat) 
[iTALK](https://github.com/Coolgenome/iTALK) 
[CellPhoneDB](https://github.com/Teichlab/cellphonedb) 
[NATMI](https://github.com/asrhou/NATMI) 
[CellComNet](https://github.com/plhhnu/CellComNet) 
[CellDialog](https://github.com/plhhnu/CellDialog) 
[Cellinker](http://www.rna-society.org/cellinker) 
[SingleCellSignalR](https://github.com/SCA-IRCM) 
[Connectome](https://msraredon.github.io/Connectome) 
[Cytotalk](https://github.com/tanlabcode/CytoTalk)
