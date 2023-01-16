import numpy as np
import torch
import torch.nn.functional as F
import tqdm

def np_sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z

def generate_score(regularization, mag, reward_bound=10.0):
    sigmoid_range = 6
    if(regularization=='sigmoid'):
        val = np_sigmoid(-sigmoid_range+sigmoid_range*mag*2)*reward_bound-5
    elif (regularization=='linear'):
        if(mag>1.0):
            val = -1*reward_bound + (2.0-mag)*2*reward_bound
        else:
            val = -1*reward_bound + (mag*2*reward_bound)
    elif 'exp' in regularization:
        split_words = regularization.split('-')
        if 'n' in split_words[-1]:
            slope = -int(split_words[-1][1:])
        else:
            slope = int(split_words[-1])
        p = (2*reward_bound)/(np.exp(slope)-1)
        q = -1*reward_bound*(1+np.exp(slope))/(np.exp(slope)-1)
        val = q+p*np.exp(slope*mag)
    return val
        

def rank_pal_auto(div: str, agent_samples, expert_samples, reward_func, reward_optimizer,  device, regularization='vanilla', reward_bound=10.0, epochs=1, max_iterations=None, state_dim=None,batch_size=1024):
    ''' 
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    sA, _, _ = agent_samples
    _, T, d = sA.shape
    sA = torch.FloatTensor(sA)
    sE = torch.FloatTensor(expert_samples).reshape(-1,T, d)
    loss_fn = torch.nn.MSELoss()
    intermediate_samples = np.arange(0.0,1.1,0.2) #np.arange(-0.4,1.4,0.2)
    

    # Generate dataset
    traj_dataset = None
    label_dataset = None

    for mag  in intermediate_samples:
        sE_sample = sE[np.random.choice(sE.shape[0],size=sA.shape[0])]
        sM = sA + mag*(sE_sample-sA) # trajectory mixup
        sM_vec = sM
        val = generate_score(regularization, mag, reward_bound)
        if traj_dataset is None:
            traj_dataset =sM_vec
            label_dataset = np.ones((sM_vec.shape[0],sM_vec.shape[1]))*int(val)
        else:
            traj_dataset = np.concatenate((traj_dataset,sM_vec),axis=0)
            labels =np.ones((sM_vec.shape[0],sM_vec.shape[1]))*int(val)
            label_dataset = np.concatenate((label_dataset,labels),axis=0)


    state_dataset = traj_dataset.reshape(-1,d)
    state_label_dataset = label_dataset.reshape(-1)
    idx = np.arange(state_dataset.shape[0])
    holdout_size = int(0.1*(state_dataset.shape[0]))
    train_dataset = state_dataset[idx[holdout_size:],:]
    train_label = state_label_dataset[idx[holdout_size:]]

    holdout_dataset = state_dataset[idx[:holdout_size],:]
    holdout_label = state_label_dataset[idx[:holdout_size]]

    if max_iterations is not None:
        max_epoch_since_update = max_iterations
    elif epochs==-1:
        max_epoch_since_update = 5
        best = None
    else:
        max_epoch_since_update = epochs
        best = np.inf
    
    epoch_since_update = 0
    iterations = 0
    
    pbar = tqdm.trange(100, desc="Training reward")
    for epoch in pbar: 
        idx = np.arange(train_dataset.shape[0])
        np.random.shuffle(idx)
        for i in range(0,idx.shape[0],batch_size):
            train_x = train_dataset[idx[i:min(i+batch_size,idx.shape[0])]]
            label_x = train_label[idx[i:min(i+batch_size,idx.shape[0])]].reshape(-1,1)
            t1 = reward_func.r(torch.FloatTensor(train_x).to(device))
            loss_val= loss_fn(t1,torch.FloatTensor(label_x).to(device))
            reward_optimizer.zero_grad()
            loss_val.backward()
            reward_optimizer.step()
            iterations+=1
            if max_iterations is not None and iterations>=max_iterations:
                break
        if (max_iterations is not None and iterations>=max_iterations):
                break
        holdout_loss = 0
        for i in range(0,holdout_dataset.shape[0],batch_size):
            holdout_x = holdout_dataset[i:min(i+batch_size,holdout_dataset.shape[0])]
            holdout_label_x = holdout_label[i:min(i+batch_size,holdout_dataset.shape[0])].reshape(-1,1)
            t1 = reward_func.r(torch.FloatTensor(holdout_x).to(device))
            loss_val= loss_fn(t1,torch.FloatTensor(holdout_label_x).to(device))
            holdout_loss+= loss_val.item()*(min(i+batch_size,holdout_dataset.shape[0])-i)
        holdout_loss/= (holdout_dataset.shape[0]+1)
        pbar.set_description("Holdout loss: {}".format(holdout_loss))
        if(best is None or ((best-holdout_loss)/best>0.01)):
            epoch_since_update = 0
            best = holdout_loss
        else:
            epoch_since_update+=1
        if(epoch_since_update>max_epoch_since_update):
            break
        

    return  loss_val.item() 



def rank_ral_auto(div: str, cum_agent_samples, expert_samples, reward_func, reward_optimizer, device, regularization='vanilla', epochs=1,max_iterations=None,batch_size = 1024):
    sA = cum_agent_samples
    _, T, d = sA.shape
    sA_vec = sA.reshape(-1,d)
    sE = expert_samples.reshape(-1,T,d)
    loss_fn = torch.nn.MSELoss()
    intermediate_samples = np.arange(0.0,1.1,0.2) 
    K = intermediate_samples.shape[0]
    sigmoid_range = 6
    if max_iterations is not None:
        epochs = max_iterations
    
    pbar = tqdm.trange(epochs, desc="Training reward")
    idx = np.arange(sA_vec.shape[0]*intermediate_samples.shape[0])
    np.random.shuffle(idx)
    total_dataset_size = idx.shape[0]
    training_idx = idx[:-int(0.1*total_dataset_size)]
    holdout_idx = idx[-int(0.1*total_dataset_size):]
    iterations = 0
    for epoch in pbar:
        for i in range(0,training_idx.shape[0],batch_size):
            batch_idx = training_idx[i:min(i+batch_size,training_idx.shape[0])]
            traj_id = batch_idx//(T*K)
            time_step = (batch_idx%(T*K))//K
            interp = (batch_idx% K).reshape(-1)
            interp =  intermediate_samples[interp.reshape(-1)]
            current_batch_size = min(i+batch_size,training_idx.shape[0]) - i
            
            sA_sample = np.take_along_axis(sA[traj_id],time_step.reshape(-1,1,1),axis=1).reshape(current_batch_size,-1)
            sE_sample = np.take_along_axis(sE[np.random.choice(sE.shape[0],size=current_batch_size)],time_step.reshape(-1,1,1),axis=1).reshape(current_batch_size,-1)
            # sE_sample = sE[np.random.choice(sE.shape[0],size=sA.shape[0])][:,time_step,:]
            sM_sample = sA_sample + interp.reshape(-1,1)*(sE_sample-sA_sample)
            # val = generate_score(regularization,mag)
            if(regularization=='sigmoid'):
                val = np_sigmoid(-sigmoid_range+sigmoid_range*interp*2)*10-5
            elif (regularization=='linear'):
                val = -10 + (interp>1.0)*(2.0-interp)*20 + (interp<=1.0)*(interp*20)
            elif 'exp' in regularization:
                split_words = regularization.split('-')
                if 'n' in split_words[-1]:
                    slope = -int(split_words[-1][1:])
                else:
                    slope = int(split_words[-1])
                p = 20/(np.exp(slope)-1)
                q = -10*(1+np.exp(slope))/(np.exp(slope)-1)
                val = q+p*np.exp(slope*interp)
            train_x = sM_sample
            label_x = val.reshape(-1,1)
            t1 = reward_func.r(torch.FloatTensor(train_x).to(device))
            loss_val= loss_fn(t1,torch.FloatTensor(label_x).to(device))
            reward_optimizer.zero_grad()
            loss_val.backward()
            reward_optimizer.step()
            iterations+=1
            if (max_iterations is not None and iterations>=max_iterations):
                break
        if (max_iterations is not None and iterations>=max_iterations):
                break
    
    torch.cuda.empty_cache()

    return  loss_val.item() 


