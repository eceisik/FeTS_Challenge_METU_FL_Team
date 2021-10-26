import os
import numpy as np

from fets_challenge import run_challenge_experiment
from fets_challenge.experiment import logger
from fets_challenge.spec_sens_code import brats_labels

#Our packages
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# AdaptiveEpoch variables
initial_loss = None 
initial_tau = 8    
initial_epoch = 1

isUsingWandb = False
hp_epoch_per_round = 1
hp_batches_per_round = None
hp_learning_rate = 5e-5

best_score = 0 
lr_scheduling_counter = 0
lr_scheduling_patience = 20
lr_scheduling_factor = 0.2
min_lr = 1e-6

training_nodes_runtimes = dict()      ## faster nodes variable
training_nodes_mean_times = dict()    ## faster nodes variable

## global variables
is_dice_score_calculated_before = False
previous_DICE_score = 0
collaborator_scores = pd.DataFrame()

selected_collaborators_and_round_pairs = []

#momentum parameters
isUsingMomentum = True
momentum = 0.9
aggregator_lr = 1.0
weight_speeds = {}

def channel_sensitivity(output, target):
    # computes TP/P for a single channel 

    true_positives = np.sum(output * target)
    total_positives = np.sum(target)

    if total_positives == 0:
        score = 1.0
    else:
        score = true_positives / total_positives
    
    return score


def channel_specificity(output, target):
    # computes TN/N for a single channel

    true_negatives = np.sum((1 - output) * (1 - target))
    total_negatives = np.sum(1 - target)

    if total_negatives == 0:
        score = 1.0
    else:
        score = true_negatives / total_negatives
        
    return score
   
    
def sensitivity(output, target):
    """"
    Calculates the average sensitivity across all of ET, TC, and WT.
    Args:
        Targets: numpy array of target values
        Predictions: numpy array of predicted values by the model
    """        
 
    # parsing model output and target into each of ET, TC, and WT arrays
    brats_val_data = brats_labels(output=output, target=target)
    
    outputs = brats_val_data['outputs']
    targets = brats_val_data['targets']
    
    output_enhancing = outputs['ET'] 
    target_enhancing = targets['ET']

    output_core = outputs['TC'] 
    target_core = targets['TC'] 

    output_whole = outputs['WT'] 
    target_whole = targets['WT']

    sensitivity_for_enhancing = channel_sensitivity(output=output_enhancing, 
                                                    target=target_enhancing)

    sensitivity_for_core = channel_sensitivity(output=output_core, 
                                               target=target_core)

    sensitivity_for_whole = channel_sensitivity(output=output_whole, 
                                                target=target_whole)

    return (sensitivity_for_enhancing + sensitivity_for_core + sensitivity_for_whole) / 3.0
    
    
def specificity(output, target):
    """"
    Calculates the average sensitivity across all of ET, TC, and WT.
    Args:
        Targets: numpy array of target values
        Predictions: numpy array of predicted values by the model
    """  
        
    # parsing model output and target into each of ET, TC, and WT arrays
    brats_val_data = brats_labels(output=output, target=target)
    
    outputs = brats_val_data['outputs']
    targets = brats_val_data['targets']

    
    output_enhancing = outputs['ET'] 
    target_enhancing = targets['ET']

    output_core = outputs['TC'] 
    target_core = targets['TC'] 

    output_whole = outputs['WT'] 
    target_whole = targets['WT']

    specificity_for_enhancing = channel_specificity(output=output_enhancing, 
                                                    target=target_enhancing)

    specificity_for_core = channel_specificity(output=output_core, 
                                               target=target_core)

    specificity_for_whole = channel_specificity(output=output_whole, 
                                                target=target_whole)

    return (specificity_for_enhancing + specificity_for_core + specificity_for_whole) / 3

#############################################################################################################################
# Collaborator Selection
#############################################################################################################################

def all_collaborators_train(collaborators,
                            db_iterator,
                            fl_round,
                            collaborators_chosen_each_round,
                            collaborator_times_per_round):
    
# add to collaborator choosing function for improved nodes
    global is_dice_score_calculated_before
    is_dice_score_calculated_before = False        
   
    tags_local = ('metric', 'validate_local')
    tags_agg = ("metric", "validate_local")
    if isUsingWandb:
        for record in db_iterator:            
            if (len(record["tags"]) == 3) \
            and (set(tags_local) < set(record['tags'])) \
            and (record["round"] == fl_round-1):
                wandb.log(
                {
                    "round"     : fl_round,
                    record["tags"][2]+" - "+record["tensor_name"]        : record["nparray"]
                })
            elif len(record["tags"])==2 and \
            set(tags_agg) == set(record['tags']) and \
            record["round"] == fl_round-1:                
                wandb.log(
                {
                    "round"     : fl_round,
                    "Aggregator - " + record["tensor_name"]        : record["nparray"]
                })
            elif record["round"] == (fl_round-1) \
            and record["tags"] == ("metric",) \
            and record["tensor_name"] == "loss":
                wandb.log(
                {
                    "round"     : fl_round,
                    record["tensor_name"]        : record["nparray"]
                })
        
    return collaborators

def random_collaborators_train(collaborators,
                            db_iterator,
                            fl_round,
                            collaborators_chosen_each_round,
                            collaborator_times_per_round):
    

    # add to collaborator choosing function for improved nodes
    global is_dice_score_calculated_before
    is_dice_score_calculated_before = False
    
    tags_local = ('metric','validate_local')
    tags_agg = ("metric", "validate_local")
    if isUsingWandb:
        for record in db_iterator:         
            if (len(record["tags"]) == 3) and (set(tags_local) < set(record['tags'])) and (record["round"] == fl_round-1):
                wandb.log(
                {
                    "round"     : fl_round,
                    record["tags"][2]+" - "+record["tensor_name"]        : record["nparray"]
                })
            elif len(record["tags"])==2 and \
            set(tags_agg) == set(record['tags']) and \
            record["round"] == fl_round-1:                
                wandb.log(
                {
                    "round"     : fl_round,
                    "Aggregator - " + record["tensor_name"]        : record["nparray"]
                })
    
    selected_collaborators = random.sample(collaborators, 4) # Hardcoded to 4 
    logger.warning("Selected Collaborators")
    logger.warning(selected_collaborators)
        
    return selected_collaborators

def random_single_collaborator_train(collaborators,
                            db_iterator,
                            fl_round,
                            collaborators_chosen_each_round,
                            collaborator_times_per_round):

    # ### add to collaborator choosing function for improved nodes
    global is_dice_score_calculated_before
    is_dice_score_calculated_before = False
    
    tags_local = ('metric','validate_local')
    tags_agg = ("metric", "validate_local")
    if isUsingWandb:
        for record in db_iterator:         
            if (len(record["tags"]) == 3) and (set(tags_local) < set(record['tags'])) and (record["round"] == fl_round-1):
                wandb.log(
                {
                    "round"     : fl_round,
                    record["tags"][2]+" - "+record["tensor_name"]        : record["nparray"]
                })
            elif len(record["tags"])==2 and \
            set(tags_agg) == set(record['tags']) and \
            record["round"] == fl_round-1:                
                wandb.log(
                {
                    "round"     : fl_round,
                    "Aggregator - " + record["tensor_name"]        : record["nparray"]
                })
    
    selected_collaborator = [random.choice(collaborators)]
    logger.warning("Selected Collaborator")
    logger.warning(selected_collaborator)
        
    return selected_collaborator

def choose_random_nodes_with_faster_ones(collaborators,
                            db_iterator,
                            fl_round,
                            collaborators_chosen_each_round,
                            collaborator_times_per_round):
    
# ### add to collaborator choosing function for improved nodes
    global is_dice_score_calculated_before
    is_dice_score_calculated_before = False
    
    global training_nodes_runtimes
    global training_nodes_mean_times
    
    ## select all collaborators for the first round 
    if fl_round==0:
        for nodes in collaborators:
            training_nodes_runtimes[nodes] = []
            training_nodes_mean_times[nodes] = 0
             
    else:
        ## update mean-runtime if there is training on this node
        previous_round_chosen_nodes = collaborators_chosen_each_round[fl_round-1]
        for chosen_nodes in previous_round_chosen_nodes:
            training_nodes_runtimes[chosen_nodes].append(collaborator_times_per_round[fl_round-1][chosen_nodes])
            training_nodes_mean_times[chosen_nodes] = np.mean(np.array(training_nodes_runtimes[chosen_nodes]))
             

        ## select random node and the other dudes which are faster 
        random_node = np.random.choice(collaborators)
        random_node_with_faster_ones = []
        
        for key, value in training_nodes_mean_times.items():
            if value <= training_nodes_mean_times[random_node]:
                random_node_with_faster_ones.append(key)
        
        collaborators= random_node_with_faster_ones        
   
    tags_local = ('metric','validate_local')
    tags_agg = ("metric", "validate_local")
    if isUsingWandb:
        for record in db_iterator:            
            if (len(record["tags"]) == 3) \
            and (set(tags_local) < set(record['tags'])) \
            and (record["round"] == fl_round-1):
                wandb.log(
                {
                    "round"     : fl_round,
                    record["tags"][2]+" - "+record["tensor_name"]        : record["nparray"]
                })
            elif len(record["tags"])==2 and \
            set(tags_agg) == set(record['tags']) and \
            record["round"] == fl_round-1:                
                wandb.log(
                {
                    "round"     : fl_round,
                    "Aggregator - " + record["tensor_name"]        : record["nparray"]
                })             
    
    
    return collaborators

#############################################################################################################################
# Aggregation
#############################################################################################################################

def fedNova_simplified(local_tensors,
             db_iterator,
             tensor_name,
             fl_round,
             collaborators_chosen_each_round,
             collaborator_times_per_round):
    
    if fl_round == 0:
        # Just apply FedAvg
        
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]               
        new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        
                       
        return new_tensor_weight  
    else:
        # Calculate aggregator's last value
        previous_tensor_value = None
        for record in db_iterator:
            if (record['round'] == (fl_round) 
                and record["tensor_name"] == tensor_name
                and record["tags"] == ("aggregated",)):
                previous_tensor_value = record['nparray']
                break
                
        deltas = [previous_tensor_value - t.tensor for t in local_tensors]
#         weight_values = [t.weight for t in local_tensors]
        grad_nova =  np.average(deltas, axis=0)
        
        new_tensor_weight = previous_tensor_value - aggregator_lr *grad_nova
        
        return new_tensor_weight
    
        
def weighted_average_aggregation(local_tensors,
                                 db_iterator,
                                 tensor_name,
                                 fl_round,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
    
    # basic weighted fedavg
    # here are the tensor values themselves
    tensor_values = [t.tensor for t in local_tensors]
    
    # and the weights (i.e. data sizes)
    weight_values = [t.weight for t in local_tensors]
    
    # so we can just use numpy.average
    return np.average(tensor_values, weights=weight_values, axis=0)

def coordinatewise_median_aggregation(local_tensors,
                                 db_iterator,
                                 tensor_name,
                                 fl_round,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
      
    tensor_values = [t.tensor for t in local_tensors]
    return np.median(tensor_values,  axis=0)

def make_aggregation_with_improved_nodes(local_tensors,
                        db_iterator,
                        tensor_name,
                        fl_round,
                        collaborators_chosen_each_round,
                        collaborator_times_per_round):
    """Aggregate tensors. This aggregator only evaluates collaborators that increased the performance metric after local tuning

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    
    global is_dice_score_calculated_before 
    global previous_DICE_score
    global collaborator_scores
    global selected_collaborators_and_round_pairs
    
    selected_tensors = []
    selected_weights = []
    
    metric_DICE_ET = 'performance_evaluation_metric_binary_DICE_ET'
    metric_DICE_TC = 'performance_evaluation_metric_binary_DICE_TC'
    metric_DICE_WT = 'performance_evaluation_metric_binary_DICE_WT'    
    
    collaborator_names = []
    for local_tensor in local_tensors:
        collaborator_names.append(local_tensor.col_name)        
    total_collaborator = len(local_tensors)
       
    if not is_dice_score_calculated_before:
 
        data = np.ones((total_collaborator, 3))*-1    
        collaborator_scores = pd.DataFrame(data=data, columns=[metric_DICE_ET, metric_DICE_TC, metric_DICE_WT])
        collaborator_scores.index = collaborator_names

        previous_aggregator_scores_DICE = np.array([])

        got_previous_score = False
        got_current_scores = False

        for record in db_iterator:
            
            # Get previous round aggregator score
            if fl_round == 0:
                got_previous_score = True                  
            elif record["round"] == (fl_round-1) \
            and "metric" in record["tags"] \
            and "validate_local" in record["tags"] \
            and len(record["tags"]) == 2 \
            and (record["tensor_name"] == metric_DICE_ET or \
            record["tensor_name"] == metric_DICE_TC or \
            record["tensor_name"] == metric_DICE_WT):
                previous_aggregator_scores_DICE = np.append(previous_aggregator_scores_DICE, record["nparray"])
                if len(previous_aggregator_scores_DICE) == 3:
                    logger.warning("previous_aggregator_scores_DICE is OK!!! ")
                    got_previous_score = True

            # Get local scores for the current round
            for local_tensor in local_tensors:
                tags = ('metric', 'validate_local')
                tags = set(tags + (local_tensor.col_name,))
                
                if tags <= set(record['tags']) \
                and record['round'] == fl_round \
                and (record["tensor_name"] == metric_DICE_ET \
                     or record["tensor_name"] == metric_DICE_TC \
                     or record["tensor_name"] == metric_DICE_WT):
                    collaborator_scores.loc[local_tensor.col_name, record["tensor_name"]] = record["nparray"]

                    if not (-1 in collaborator_scores.values):
                        logger.warning("Current collaborator_scores is OK!!! ")
                        got_current_scores = True

            # Exit loop if all information is collected
            if got_current_scores and got_previous_score:
                is_dice_score_calculated_before = True
                logger.warning("Previous and current scores are found, exiting db_iterator!")
                break
        
        if is_dice_score_calculated_before:

            if fl_round == 0:
                previous_DICE_score = 0
            else:
                previous_DICE_score = np.mean(previous_aggregator_scores_DICE)
            logger.warning("PREVIOUS DICE SCORE CALCULATED: " + str(previous_DICE_score))

            collaborator_scores["mean"] = collaborator_scores.mean(axis=1)

            if isUsingWandb:
                wandb.log({
                            "round"     : fl_round,
                            "Aggregator mean DICE score": previous_DICE_score
                        })
                for local_tensor in local_tensors:
                    wandb.log({
                            "round"     : fl_round,
                            local_tensor.col_name + " mean DICE": collaborator_scores.loc[local_tensor.col_name, "mean"]
                        })

                for local_tensor in local_tensors:
                    if collaborator_scores.loc[local_tensor.col_name, "mean"] > previous_DICE_score:
                        selected_collaborators_and_round_pairs.append([fl_round, int(local_tensor.col_name)])
                
                plt.scatter([item[0] for item in selected_collaborators_and_round_pairs],
                            [item[1] for item in selected_collaborators_and_round_pairs])
                plt.title("Selected collaboators for aggregation")
                plt.xlabel("round")
                plt.ylabel("Collaborators")
                wandb.log({str(fl_round)+" selected collaborators": plt})
                
                logger.warning("Round: "+str(fl_round)+" round x collaborator pairs")
                logger.warning(selected_collaborators_and_round_pairs)
        else:
            logger.warning("ERROR in calculating DICE Scores!")     
            exit()
            
#     burası hepsi için olacak
    for local_tensor in local_tensors:
        if collaborator_scores.loc[local_tensor.col_name, "mean"] > previous_DICE_score:
            selected_tensors.append(local_tensor.tensor)
            selected_weights.append(local_tensor.weight)
                
    if len(selected_tensors)==0:
        selected_tensors= [t.tensor for t in local_tensors]
        selected_weights= [t.weight for t in local_tensors]
        logger.warning("hepsi ortalamanın altında, hepsi seçildi")
        
    return np.average(selected_tensors, weights=selected_weights, axis=0)


def fedAvgM(local_tensors,
             db_iterator,
             tensor_name,
             fl_round,
             collaborators_chosen_each_round,
             collaborator_times_per_round):
    
    global weight_speeds
    global momentum
    global aggregator_lr
    
    if fl_round == 0:
        # Just apply FedAvg
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]               
        new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        
        
        if not (tensor_name in weight_speeds):
            weight_speeds[tensor_name] = np.zeros_like(local_tensors[0].tensor) # weight_speeds[tensor_name] = np.zeros(local_tensors[0].tensor.shape)
                       
        return new_tensor_weight        
    else:
        if tensor_name.endswith("weight") or tensor_name.endswith("bias"):
            # Calculate aggregator's last value
            previous_tensor_value = None
            for record in db_iterator:
                if (record['round'] == (fl_round) 
                    and record["tensor_name"] == tensor_name
                    and record["tags"] == ("aggregated",)): 
                    previous_tensor_value = record['nparray']
                    break

            if previous_tensor_value is None:
                logger.warning("Error in fedAvgM: previous_tensor_value is None")
                logger.warning("Tensor: " + tensor_name)

                # Just apply FedAvg       
                tensor_values = [t.tensor for t in local_tensors]
                weight_values = [t.weight for t in local_tensors]               
                new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        

                if not (tensor_name in weight_speeds):
                    weight_speeds[tensor_name] = np.zeros_like(local_tensors[0].tensor) # weight_speeds[tensor_name] = np.zeros(local_tensors[0].tensor.shape)

                return new_tensor_weight
            else:
                # compute the average delta for that layer
                deltas = [previous_tensor_value - t.tensor for t in local_tensors]
                weight_values = [t.weight for t in local_tensors]
                average_deltas = np.average(deltas, weights=weight_values, axis=0) 

                # V_(t+1) = momentum*V_t + Average_Delta_t
                weight_speeds[tensor_name] = momentum * weight_speeds[tensor_name] + average_deltas
                
                # W_(t+1) = W_t-lr*V_(t+1)
                new_tensor_weight = previous_tensor_value - aggregator_lr*weight_speeds[tensor_name]

                return new_tensor_weight
        else:
            # Just apply FedAvg       
            tensor_values = [t.tensor for t in local_tensors]
            weight_values = [t.weight for t in local_tensors]               
            new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)

            return new_tensor_weight


#############################################################################################################################
# Hyperparameter choice
#############################################################################################################################


def adaptive_epoch_with_lr_scheduling(collaborators,
                              db_iterator,
                              fl_round,
                              collaborators_chosen_each_round,
                              collaborator_times_per_round):
    
    #### adacomm variables
    global initial_tau
    global initial_loss
    global initial_epoch
    
    current_loss=None
    
    #### lr scheduling starts
    global hp_learning_rate
    global best_score
    global lr_scheduling_counter     
    
    metric_DICE_ET = 'performance_evaluation_metric_binary_DICE_ET'
    metric_DICE_TC = 'performance_evaluation_metric_binary_DICE_TC'
    metric_DICE_WT = 'performance_evaluation_metric_binary_DICE_WT'
    
    previous_aggregator_scores_DICE = np.array([])
    previous_DICE_score = 0
    
    if fl_round==0:
        epochs_per_round = initial_epoch
        learning_rate = hp_learning_rate
        
    elif fl_round==1:
        epochs_per_round = initial_tau
        for record in db_iterator:
            if record["round"] == (fl_round-1) \
                and record["tags"] == ("metric",) \
                and record["tensor_name"] == "loss": 
                initial_loss = record["nparray"]
            
            # Get previous round aggregator score
            if record["round"] == (fl_round-1) \
                and "metric" in record["tags"] \
                and "validate_local" in record["tags"] \
                and len(record["tags"]) == 2 \
                and (record["tensor_name"] == metric_DICE_ET or \
                record["tensor_name"] == metric_DICE_TC or \
                record["tensor_name"] == metric_DICE_WT):
                previous_aggregator_scores_DICE = np.append(previous_aggregator_scores_DICE, record["nparray"])
                if len(previous_aggregator_scores_DICE) == 3:                
                    previous_DICE_score = np.mean(previous_aggregator_scores_DICE)
                    logger.warning("Calculated mean aggregator score for round "+str(fl_round-1)+": "+str(previous_DICE_score))    
            
        if previous_DICE_score > best_score:
            best_score = previous_DICE_score
            lr_scheduling_counter = 0
        else:
            lr_scheduling_counter += 1

        if lr_scheduling_counter >= lr_scheduling_patience:
            hp_learning_rate = hp_learning_rate* lr_scheduling_factor
            lr_scheduling_counter = 0  
        learning_rate = hp_learning_rate                
    else:
        for record in db_iterator:
            if record["round"] == (fl_round-1) \
                and record["tags"] == ("metric",) \
                and record["tensor_name"] == "loss": 
                current_loss = record["nparray"]
                
            # Get previous round aggregator score
            if record["round"] == (fl_round-1) \
                and "metric" in record["tags"] \
                and "validate_local" in record["tags"] \
                and len(record["tags"]) == 2 \
                and (record["tensor_name"] == metric_DICE_ET or \
                record["tensor_name"] == metric_DICE_TC or \
                record["tensor_name"] == metric_DICE_WT):
                previous_aggregator_scores_DICE = np.append(previous_aggregator_scores_DICE, record["nparray"])
                if len(previous_aggregator_scores_DICE) == 3:                
                    previous_DICE_score = np.mean(previous_aggregator_scores_DICE)
                    logger.warning("Calculated mean aggregator score for round "+str(fl_round-1)+": "+str(previous_DICE_score))                
        
        current_tau = np.ceil( np.sqrt(current_loss/initial_loss)  * initial_tau)
        epochs_per_round = current_tau
        
        if previous_DICE_score > best_score:
            best_score = previous_DICE_score
            lr_scheduling_counter = 0
        else:
            lr_scheduling_counter += 1

        if lr_scheduling_counter >= lr_scheduling_patience:
            hp_learning_rate = hp_learning_rate* lr_scheduling_factor
            if hp_learning_rate < min_lr:
                hp_learning_rate = min_lr
            lr_scheduling_counter = 0      
        learning_rate = hp_learning_rate
        
    batches_per_round = hp_batches_per_round        
    
    if isUsingWandb:
        wandb.log(
                {
                    "round"       : fl_round,
                    "tau"         : epochs_per_round,
                    "current loss": current_loss,
                    "initial loss": initial_loss,
                    "lr"          : learning_rate
                })
            
    return (learning_rate, epochs_per_round, batches_per_round)


def adaptive_epoch(collaborators,
                              db_iterator,
                              fl_round,
                              collaborators_chosen_each_round,
                              collaborator_times_per_round):
    
    global initial_tau
    global initial_loss
    global initial_epoch
    
    current_loss=None
    
    if fl_round==0:
        epochs_per_round = initial_epoch        
      
    elif fl_round==1:
        epochs_per_round = initial_tau
        for record in db_iterator:
            if record["round"] == (fl_round-1) \
            and record["tags"] == ("metric",) \
            and record["tensor_name"] == "loss": 
                initial_loss = record["nparray"]
                break
    else:
        for record in db_iterator:
            if record["round"] == (fl_round-1) \
            and record["tags"] == ("metric",) \
            and record["tensor_name"] == "loss": 
                current_loss = record["nparray"]
                break
        
        current_tau = np.ceil( np.sqrt(current_loss/initial_loss)  * initial_tau)
        epochs_per_round = current_tau
    
    if isUsingWandb:
        wandb.log(
                {
                    "round"     : fl_round,
                    "tau"       : epochs_per_round,
                    "current loss": current_loss,
                    "initial loss": initial_loss                   
                })
    
    batches_per_round = hp_batches_per_round
    learning_rate = hp_learning_rate
    return (learning_rate, epochs_per_round, batches_per_round)

def constant_hyper_parameters(collaborators,
                              db_iterator,
                              fl_round,
                              collaborators_chosen_each_round,
                              collaborator_times_per_round):    
    
    epochs_per_round = hp_epoch_per_round
    batches_per_round = hp_batches_per_round
    learning_rate = hp_learning_rate
    return (learning_rate, epochs_per_round, batches_per_round)


def lrscheduling_hyper_parameters(collaborators,
                              db_iterator,
                              fl_round,
                              collaborators_chosen_each_round,
                              collaborator_times_per_round):
    
    global hp_learning_rate
    global best_score
    global lr_scheduling_counter
    
    metric_DICE_ET = 'performance_evaluation_metric_binary_DICE_ET'
    metric_DICE_TC = 'performance_evaluation_metric_binary_DICE_TC'
    metric_DICE_WT = 'performance_evaluation_metric_binary_DICE_WT'
    
    previous_aggregator_scores_DICE = np.array([])
    previous_DICE_score = 0
    for record in db_iterator:
        # Get previous round aggregator score
        if record["round"] == (fl_round-1) \
        and "metric" in record["tags"] \
        and "validate_local" in record["tags"] \
        and len(record["tags"]) == 2 \
        and (record["tensor_name"] == metric_DICE_ET or \
        record["tensor_name"] == metric_DICE_TC or \
        record["tensor_name"] == metric_DICE_WT):
            previous_aggregator_scores_DICE = np.append(previous_aggregator_scores_DICE, record["nparray"])
            if len(previous_aggregator_scores_DICE) == 3:                
                previous_DICE_score = np.mean(previous_aggregator_scores_DICE)
                logger.warning("Calculated mean aggregator score for round "+str(fl_round-1)+": "+str(previous_DICE_score))
                break
     
    if previous_DICE_score > best_score:
        best_score = previous_DICE_score
        lr_scheduling_counter = 0
    else:
        lr_scheduling_counter += 1
    
    if lr_scheduling_counter >= lr_scheduling_patience:
        hp_learning_rate = hp_learning_rate* lr_scheduling_factor
        if hp_learning_rate < min_lr:
            hp_learning_rate = min_lr
        lr_scheduling_counter = 0
    
    epochs_per_round = hp_epoch_per_round
    batches_per_round = hp_batches_per_round
    learning_rate = hp_learning_rate
    
    if isUsingWandb:
        wandb.log({
            "round": fl_round,
            "lr": learning_rate
        })
    
    return (learning_rate, epochs_per_round, batches_per_round)

#################################################################################################################################
# Experiment Setup
#################################################################################################################################

aggregation_function = fedAvgM
choose_training_collaborators = all_collaborators_train
training_hyper_parameters_for_round = adaptive_epoch_with_lr_scheduling
validation_functions = [('sensitivity', sensitivity), ('specificity', specificity)]

include_validation_with_hausdorff = False
institution_split_csv_filename = 'partitioning_2.csv'
brats_training_data_parent_dir = 'your_path'

db_store_rounds = 2
device = 'cuda'
rounds_to_train = 300
save_checkpoints = True
challenge_metrics_validation_interval = 1     # if make_aggregation_with_improved_nodes is used it should be 1
restore_from_checkpoint_folder = None

if isUsingWandb:
    wandb.init(project="fets2021-2", entity="metu_fets", save_code=True)
    wandb.run.name = os.path.basename(__file__)[:-3] + "_" + wandb.run.name.split("-")[2]
    wandb.run.save()

    config = wandb.config
    config.aggregator = aggregation_function.__name__
    config.collaborator = choose_training_collaborators.__name__
    config.hyperparameter = training_hyper_parameters_for_round.__name__
    
    config.partitioning = institution_split_csv_filename
    config.epoch_per_round = hp_epoch_per_round
    config.batches_per_round = hp_batches_per_round
    config.learning_rate = hp_learning_rate 
    config.aggregator_lr = aggregator_lr
    config.momentum = momentum
    config.db_store = db_store_rounds
    config.rounds = rounds_to_train

scores_dataframe = run_challenge_experiment(
    aggregation_function=aggregation_function,
    choose_training_collaborators=choose_training_collaborators,
    training_hyper_parameters_for_round=training_hyper_parameters_for_round,
    validation_functions=validation_functions,
    include_validation_with_hausdorff=include_validation_with_hausdorff,
    institution_split_csv_filename=institution_split_csv_filename,
    brats_training_data_parent_dir=brats_training_data_parent_dir,
    db_store_rounds=db_store_rounds,
    rounds_to_train=rounds_to_train,
    device=device,
    challenge_metrics_validation_interval=challenge_metrics_validation_interval,
    save_checkpoints=save_checkpoints,
    restore_from_checkpoint_folder = restore_from_checkpoint_folder)

print(scores_dataframe)