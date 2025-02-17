# EAGLE (a deep framework based on bipartitE grAph learninG for anomaLy dEtection)

When Bipartite Graph Learning Meets Anomaly Detection in Attributed Networks: Understand Abnormalities from Each Attribute (Peng, Z, Wang, Y, Lin, Q, Dong, B, & Shen, C, Neural Networks 2025): [https://doi.org/10.1016/j.neunet.2025.107194](https://doi.org/10.1016/j.neunet.2025.107194)

![image](https://github.com/zpeng27/EAGLE/blob/main/eagle.png)

The code is presented in an easy-to-understand pattern. You could further optimize it.

## Requirements 
numpy>=1.23.5  
scipy>=1.10.0  
torch>=1.12.1  
dgl>=0.9.0  
tensorboard>=2.11.0  
scikit-learn>=1.2.1  
tqdm>=4.64.1  
icecream>=2.1.3  
networkx>=2.8.4  
matplotlib>=3.7.1  
pandas>=1.5.3  

## Run Enron dataset
We provide Enron dataset for model evaluation.
```shell
python main.py --dataset enron --gpu 0
```

## Run your own dataset
 
1. Process adjacency matrix and feature matrix into `scipy.sparse.csr_matrix`.
2. Process node labels into `numpy` dense matrix.
3. Pack these three things above into a `dict` with key names: 
```python
   {'A': adj_matrix, 'X': features, 'gnd': labels}
```
4. Dump the `dict` into `./data/mydataset.pickle` with `pickle` module.
5. Run in shell:
```shell
python main.py --dataset mydataset --gpu 0
```

## Cite
Please cite our paper if you make advantage of EAGLE in your research:

```
@article{peng2025bipartite,
  title={When bipartite graph learning meets anomaly detection in attributed networks: Understand abnormalities from each attribute},
  author={Peng, Zhen and Wang, Yunfan and Lin, Qika and Dong, Bo and Shen, Chao},
  journal={Neural Networks},
  pages={107194},
  year={2025},
  publisher={Elsevier}
}
```
