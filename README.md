# Federated Tumor Segmentation Challenge 2021 - METU-FL Team Challenge Repository

This repository contains METU-FL team implementations for the Federated Tumor Segmentation Challenge 2021.   

The list of given and implemented functions for Task 1 is listed below.    

**Aggregation function** determines how the server side merges the collaborator model updates. 

|  Aggregation Function Name                  | Source                    | 
| ------------------------------------------- | --------------------------|
| weighted_average_aggregation                |  Given by FeTS initiative |
| fedNova_simplified                          |  METU-FL                  |
| make_aggregation_with_improved_nodes        |  METU-FL                  |
| FedAvgM                                     |  METU-FL                  |
| coordinatewise_median_aggregation           |  METU-FL                  | 

  
**Hyperparameter choice** function determines the parameters for each FL round.  

| Function Name                               | Source                    |   
| ------------------------------------------- |-------------------------- |      
| constant_hyper_parameters                   |  Given by FeTS initiative | 
| lrscheduling_hyper_parameters               |  METU-FL                  | 
| adaptive_epoch                              |  METU-FL                  |
| **adaptive_epoch_with_lr_scheduling**       |  METU-FL                  |     
  
**Collaborator choice** function determines which collaborators are chosen to train in each FL round. 

| Function Name                               |    Source                 | 
| ------------------------------------------- | ------------------------- |      
| **all_collaborators_train**                 |  Given by FeTS initiative |
| choose_random_nodes_with_faster_ones        |  METU-FL                  |
| random_collaborators_train                  |  METU-FL                  |
| random_single_collaborator_train            |  METU-FL                  |
   
**Validation metrics** show the metrics to be computed each FL round.    


| Function Name                               | Source                    | 
| ------------------------------------------- | ------------------------- |         
| channel_sensitivity                         |  Given by FeTS initiative |
| sensitivity                                 |  Given by FeTS initiative |
| specificity                                 |  Given by FeTS initiative |

### Our challenge combination is as below:

- **Hyperparameter choice function:** *adaptive_epoch_with_lr_scheduling*  
  adaptive_epoch_with_lr_scheduling is an adaption of AdaComm with a learning scheduling scheme. The number of epochs per round decays according to the decrease in initial loss, and the learning rate decays according to the performance metric (average DICE score).

```BibText
@inproceedings{MLSYS2019_c8ffe9a5,
 author = {Wang, Jianyu and Joshi, Gauri},
 booktitle = {Proceedings of Machine Learning and Systems},
 editor = {A. Talwalkar and V. Smith and M. Zaharia},
 pages = {212--229},
 title = {Adaptive Communication Strategies to Achieve the Best Error-Runtime Trade-off in Local-Update SGD},
 url = {https://proceedings.mlsys.org/paper/2019/file/c8ffe9a587b126f152ed3d89a146b445-Paper.pdf},
 volume = {1},
 year = {2019}
}
```

- **Collaborator choice function:** *all_collaborators_train*    
    All collaborators participate in each FL round.  
    
- **Aggregation function:** *fedAvgM*     
    Federated Averaging with server momentum uses accumulated gradients for the weight update.

```BibText
@article{hsu2019measuring,
  title={Measuring the effects of non-identical data distribution for federated visual classification},
  author={Hsu, Tzu-Ming Harry and Qi, Hang and Brown, Matthew},
  journal={arXiv preprint arXiv:1909.06335},
  year={2019}
}
```

**Validation metrics:** We used given validation default metrics but considering limited computational resources, we did not include Hausdorff Distance (include_validation_with_hausdorff = False).




