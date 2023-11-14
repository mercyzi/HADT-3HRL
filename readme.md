
### 在HRL那篇论文的开源代码基础上写的，无效的代码量比较多，时间久了我没删

### 因为四个实验数据集的模型分层结构不同，我当初是写了四个版本，这个版本是dxy的好像。对于machine机器，dxy和muzhi数据集都是一个单DQN；mz10和synthetic数据集的疾病分别是分了5组和9组。

### classifire文件夹应该没用

### data文件夹是数据集依次为：dxy、muzhi、mz10、synthetic，这四个数据集。 mz4没用


### dialogue_system是主要代码：

--agent：里面agent_hrl_joint2.py是模型的智能体；agent_rule.py是医生模拟器；其余应该用处不大

--dialogue_manager：里面dialogue_manager_hrl.py是对话管理，它控制着用户、智能体、对话状态转移

--intrinsic_rewards：这个应该是我创建的一个，里面有icm、ride、rnd三种内在奖励方法，说实话没啥用

--memory：无用

--policy_learning：里面是网络的模型，ppo_torch.py对应PPO；dqn_torch.py对应DQN

--res：无用，纯实验结果垃圾

--reward_shaping：无用

--run：run是主main函数的运行文件、running_steward是主main函数运行的核心

--state_tracker：对话状态转移

--user_simulator：user用户模拟器，原作者写的太复杂，我懒得改了

--utils：画图的，

dialogue_system/disease_classifier.py：诊断模型。



### 训练稍微有点繁琐，对于一个数据集，我基本是先训练一个诊断模型；
### 再训练machine，取症状召回率效果最高的保存：若两层DQN（比如mz10和synthetic数据集），先各自训练多个reactor，再固定多个reactor训练activator；若一层DQN，直接训就完了
### 最后训练master


