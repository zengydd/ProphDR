## ProphDR: AI-powered deep phenotyping reveals biochemical determinants in variable cancer drug response   
![image](https://github.com/zengydd/ProphDR/blob/main/IMG/Fig0_abstract.png)  

### ⭐ Highlight
* ProphDR is specifically designed to intuitively prophesy biochemical determinants in various drug response prediction.
* ProphDR allows customized multi-omics combinations inputs for users to try, flexible and scalable at any combinations: e.g. mutation-only or mutation+expression ...

ProphDR is specifically designed to intuitively prophesy biochemical determinants in various drug response prediction. 

  
💡ProphDR puns on "A PROfessor of PHenotyping makes PROPHecy for biochemical determinants triggering cancer Drug Response". 😆
### Model overview
ProphDR combines a variable and scalable gene-centric encoder with a visualizable drug-gene interaction extractor based on the attention mechanism, which allows customized experiments on dynamic multi-omics data, and most importantly, can explicitly reveal the associations among specific oncogenes, drug components, and responses.  
![model_overview](https://github.com/zengydd/ProphDR/blob/main/IMG/Fig1.png)

### Environment
* Install via conda.yml file (cuda11.8)
```
conda env create -f ProphDR.yml -n ProphDR
```
  
* Install manually  

```
conda create -n ProphDR python=3.7
#Please install pytorch according to your cuda version: https://pytorch.org/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge rdkit
pip install pubchempy
pip install scipy
pip install scikit-learn
pip install collections
pip install lifelines
```  

### Data  
* Processed data:  
Dataset description can be found in the file: Data_collect  
The pretrain data for k_bert and the processed data can be downloaded from this link: [Data_collect](https://drive.google.com/file/d/1CotLTH2z8iefNqYYC3E-03mpGPQ2anYm/view?usp=sharing)   
Please replace the "\ProphDR\Data_collect" with the downloaded "Data_collect"  

* Raw data can be downloaded from:  
https://www.cancerrxgene.org/downloads/bulk_download  
https://depmap.org/portal/download/all/  
https://cancer.sanger.ac.uk/cosmic

### Run ProphDR
* Preparation: Please put k_bert under Models: [k_bert](https://drive.google.com/file/d/1h-Ve-2fpYH6P8epfsHezJsxK2PRKFnc2/view?usp=drive_link)    
* Customized parameters:  
Here we show an example of using all 3 types of multi-omics data, for other inputs combinations or tasks, please edit the "exp_params".  
```
# @@@@@@@@@@@@@@@@@@@
# task: IC50 or binary
# method: only2, Iorio
# strat: TCGA_DESC, binary, None
# omics of 3: omic_encode
# omics combinations of 2: mut_cnv mut_exp exp_cnv
# single-omics: mut_encode, cnv_encode, exp_encode
exp_params = {
    'task': task,
    'method': method,
    'strat': None,
    'omic_dim': 3,
    'omic_f':omic_encode,
    }
```
#### Test
If you wanna run our trained model, please download ckpt: [ProphDR_ckpt](https://drive.google.com/file/d/15bzGyW5V6JNSypt4f86XJOPHnr_1OHmg/view?usp=drive_link)   
```
python run test.py
```

#### Train
```
python run train.py
```

### Phenotypic drug screening  
Phenotypic screening on your drugs:  
This function allows customized inputs, as long as following the format of INPUT "smiles" & "COSMIC_ID"  
Note that "COSMIC_ID" must be contained in the cell line dataset [Cell_line_info](https://github.com/zengydd/ProphDR/tree/main/data_collect).  which covers 30 cancer types
```
# Edit here:
smiles = ['C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1', 'CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1']
cosmic_id = [1659928]
```

```
python run pred.py
```

### Attention analysis visualization
We provide an example used in our case study: Gefitinib against NSCLC cell line HCC-827 
For important drug atoms visualization:
```
python run visualize.py
```
  
🌻 Please feel free to contact me if you have any questions: zengyundian@gmail.com   
