# MEGACare

The data and source code for MEGACare: Knowledge-guided Multi-view Hypergraph Predictive Framework for Healthcare.
Related code and data will be published in a new repository after review.

## Setup

### 1. Create the rdkit conda environment
```python
conda create -c conda-forge -n MEGACare  rdkit  && conda activate MEGACare
```
### 2. Install dependecies
Install the required packages
```python
pip install rdkit-pypi, scikit-learn, dill, dnc
```

Finally, install other packages if necessary
```python
pip install [xxx] # any required package if necessary
```

### 3. Data
Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset (You may need to get the certificate)
```python
cd ./data
wget -r -N -c -np --user [account] --ask-password https://physionet.org/files/mimiciii/1.4/
```

Processing the data to get a complete records_final.pkl

Go into the folder and unzip three main files

```python
cd ./physionet.org/files/mimiciii/1.4
gzip -d PROCEDURES_ICD.csv.gz # Procedure information
gzip -d PRESCRIPTIONS.csv.gz  # Medication information
gzip -d DIAGNOSES_ICD.csv.gz  # Diagnosis information
```

### 4. Folder Specification
- ```data/```
    - Input:
        - **PRESCRIPTIONS.csv**
        - **DIAGNOSES_ICD.csv**
        - **PROCEDURES_ICD.csv**
        - **RXCUI2atc4.csv**
        - **drug-atc.csv**
        - **ndc2RXCUI.txt**
        - **drugbank_drugs_info.csv**
        - **drug-DDI.csv**
    - Output:
        - **atc3toSMILES.pkl**
        - **ADDI.pkl**
        - **SDDI.pkl**
        - **records_final.pkl**: we only provide the first 100 entries as examples here. We cannot distribute the whole MIMIC-III data.
        - **voc_final.pkl**
- ```src/```
    - Baselines:
        - **LR.py**
        - **CNN.py**
        - **RNN.py**
        - **GRAM.py**
        - **KAME.py**
        - **Dipole.py**
        - **RETAIN.py**
        - **GAMENet.py**
        - **SafeDrug.py**
        - **MICRON.py**
        - **COGNet.py**
        - **LEAP.py**
        - **Retain.py**
        - **processdata_new.py**
        - **Pre-trainMPNN.py**
        - **proposedmethod.py**
    - Setting file
        - **model.py**
        - **SafeDrug_model.py**
        - **COGNet_model.py**
        - **Statistic_ddi_rate_in_mimic.py**
        - **util.py**
        - **layer.py**

> The current statistics are shown below:

```
#patients  6,350
#clinical events  15,031
#diagnosis  1,958
#med(ATC-3rd)  132
#procedure 1,430
#avg of diagnoses  10.5089
#avg of medicines  11.1864
#avg of procedures  3.8436
#avg of vists  2.3672
#max of diagnoses  128
#max of medicines  64
#max of procedures  50
#max of visit  29
```

### 4. Tips
Welcome to contact me jialunwu96@163.com for any question.
