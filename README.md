# Cooperative multi-agent reinforcement learning for high-dimensional nonequilibrium control


Official implementation:  

**Cooperative multi-agent reinforcement learning for high-dimensional nonequilibrium control**
Shriram Chennakesavalu and Grant M. Rotskoff
https://arxiv.org/abs/2111.06875


**Abstract**: Experimental advances enabling high-resolution external control create new opportunities to produce materials with exotic properties. In this work, we investigate how a multi-agent reinforcement learning approach can be used to design external control protocols for self-assembly. We find that a fully decentralized approach performs remarkably well even with a "coarse" level of external control. More importantly, we see that a partially decentralized approach, where we include information about the local environment allows us to better control our system towards some target distribution. We explain this by analyzing our approach as a partially-observed Markov decision process. With a partially decentralized approach, the agent is able to act more presciently, both by preventing the formation of undesirable structures and by better stabilizing target structures as compared to a fully decentralized approach.

#### Installing prerequisites (using conda)

```
conda env create -f environment.yml -n marldesign
conda activate marldesign
```

Possible --centralize_approach values are ("plaquette", "all", "grid_n"), where 1 < n < region_num/2

#### Sample training commands
```
python train.py --active --centralize_states --centralize_approach plaquette
python train.py --active --centralize_rewards --centralize_approach all
python train.py --centralize_rewards --centralize_states --centralize_approach grid_1
```

#### Sample testing commands
```
python test.py --active --num_samples 10  --centralize_states --centralize_approach plaquette
python test.py --active --num_samples 10 --centralize_rewards --centralize_approach grid_1
python test.py --centralize_rewards --num_samples 10 --centralize_states --centralize_approach grid_2
```
