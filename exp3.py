import numpy as np
import pandas as pd 

class EXP3:
    def __init__(self,df,gamma = 0.1):
        ''' 
        Init a instance for exp3 bandit

        Parameters
        ---------
        df: data to backtest, must be in desired reward format (0-1)
        gamma: egalitarianism factor, controls EE threshold
        '''
        self.data = df.copy(deep = True)
        self.k = df.shape[1] # number of arms
        self.weights = np.array([1.0] * self.k) # initial weights
        self.gamma = gamma
    
    def get_prob(self):
        '''
        get a probilities distribution using weights
        '''
        return [(1-self.gamma)*self.weights[i]/self.weights.sum() + self.gamma/self.k for i in range(self.k)]
    
    def draw(self,Pt):
        '''
        draw an action At given a probilities distribution
        '''

        return np.random.choice([i for i in range(self.k)], 1,
              p=Pt)[0]
    
   
    def get_reward(self,At,t):
        '''
        get the actual reward given an iloc positio (time step, At)
        '''
        return 1 if self.data.iloc[t,At] > self.data.loc[t].mean() else 0
    
    def get_value(self,t):
        '''
        This mimics the actual reward received when ditributed our
        funds proportionally to the weights

        This doesn't affect the bandit decision but will be stored
        for evaluation purpose
        '''
        weights = np.exp(self.weights)/np.exp(self.weights).sum()
        return sum(weights * self.data.iloc[t])
    
    def get_bah_value(self,t):
        '''
        This mimics the actual reward received when ditributed our
        funds equally at all time step

        This doesn't affect the bandit decision but will be stored
        for evaluation purpose
        '''
        weights = [1/self.k] * self.k
        return sum(weights * self.data.iloc[t])
        
        
    def step(self,t):
        ''' 
        One single step function

        1. calculate the probability distribution
        2. draw an action
        3. calculated the reward
        4. get esitimated reward
        5. update the weights
        6. return the value (actual backtest return received)

        '''

        Pt = self.get_prob()      
        At = self.draw(Pt)
        reward = self.get_reward(At,t)
        xt = reward/Pt[At]
        self.weights[At] *= np.exp(self.gamma*xt/self.k)
        return self.get_value(t)
    
    def backtest(self,max_steps):  
        ''' 
        backtest on the data given and record the performance
        '''
        values = []
        bah = []
        for i in range(max_steps):
            value = self.step(i)
            values.append(value)
            bah_value = self.get_bah_value(i)
            bah.append(bah_value)
            
        self.data['exp3'] = values
        self.data['bah'] = bah