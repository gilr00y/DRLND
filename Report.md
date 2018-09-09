# Report

_Note:_ The code in this project is largely adapted from the implementation of the LunarLander exercise (Lesson 2, Part 6).  This is the only code that was re-used or "copied" from any other source, and determined to be allowed due to the suggestion in the "Not sure where to start" section of the project page:

_Adapt the code from the exercise to the project, while making as few modifications as possible._

If the use of this code is problematic or prohibited in any way, please let me know.

### Learning Algorithm

The algorithm is a straightforward implementation of DQN as described in the [Deep Mind paper published in Nature](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

This conceptually follows on table-based Q-Learning, wherein an agent infers future rewards by "looking ahead" at each 

q_local_idx = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather( 1, q_local_idx)

        # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
    
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 


### Tuning

To achieve the performance described below, certain hyperparameters for the algorithm had to be tuned to find a satisfactory result.

These hyperparameters are described as follows:

**Replay Buffer Size**

`BUFFER_SIZE = int(1e5)` 

The replay buffer size is the number of agent experiences available in the buffer for sampling during the training stage of the fully-connected MLP network. Thus a larger value ensures that the network is learning from experiences outisde of its own recent history, but as this buffer is stored in memory, it is resource-constrained.  For this project, run on 2014 Macbook Pro, a buffer size of 10^5 experiences was sufficient; however, it proved too large and had to be constrained to 10^4 for the VisualBanana environment, which is based on pixels and, as such, a greater state size.

**Batch Size**

`BATCH_SIZE = 64`

This is the number of samples taken from the Replay Buffer each time the fully-connected is trained. The Udactiy DQN exercise code initialized the batch size at `64`, and it was left unchanged for this project.

**Gamma - Discount Factor**

`GAMMA = 0.99`

This is the time-discount factor, which is applied to the expected action-value of the subsequent 

**Tau**

`TAU = 1e-3`

 soft update of target parameters

**Learning Rate**

`LR = 5e-4`

learning rate

**Update Frequency** 
`UPDATE_EVERY = 4`

how often to update the network

### Architecture

**Fully-Connect MLP Layers**



### Performance

### Future Improvements

#### Double DQN

For the double-DQN algorithm, only a few changes are necessary. A prototype implementation of this is present in a Jupyter Notebook in the project directory labeled `Navigation-DoubleDQN.ipynb`.

#### Dueling DQN



#### Prioritized Experience Replay

