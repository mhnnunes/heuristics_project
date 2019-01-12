# Heuristics and Metaheuristics for The MSSC Problem  

## Abstract  

The Euclidean MSSC (Minimum Sum-of-Squares Clustering) is a famous combinatorial optimization problem in the machine learning and computing theory literatures. Popularly known as "k-Means", due to the name of the first proposed heuristic for it, its input is a set of _n_ points scattered in an _m_-dimensional space, and a number _k_ of groups. The goal is to divide the _n_ points into _k_ groups, minimizing the euclidean distance between each of the _n_ points to the center of the group the point was assigned to. Efficient methods for solving this problem exist, and in average the algorithms for it are relatively fast. However, this problem has been proven to be _NP-Hard_ even for points in the plane. In this work we design and implement two novel heuristics and two novel metaheuristics for this problem, comparing our results to the literature. We show that our proposed methods are comparable to the state of the art in using three real and three synthetic datasets.  

The full text (in Portuguese) can be found [here](HeM_Projeto_Final.pdf).  

## Dependencies  

In order to run the code available in this directory, a few dependencies are necessary. We recommend using a virtual environment for running this code. To create a virtual environment, run the following code in your terminal.  


```{bash}  
python -m venv env_name
source env_name/bin/activate
```  


In order to install the dependencies, run the following code in your terminal:  

```{bash}  
pip3 install --user -f requirements.txt  
```  

## Authors  

[Matheus Henrique do Nascimento Nunes](https://github.com/mhnnunes)  
[Tarsila Bessa Nogueira Assunção](https://github.com/tarsilab)  
