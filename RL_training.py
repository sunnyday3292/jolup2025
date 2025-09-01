# ------------------------- RL 학습 -------------------------
state_dim=4
action_dim=3
class DQN(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(state_dim,64),nn.ReLU(),
                              nn.Linear(64,64),nn.ReLU(),
                              nn.Linear(64,action_dim))
    def forward(self,x): return self.fc(x)

buffer = deque(maxlen=10000)
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(),lr=1e-3)
gamma=0.99
epsilon=1.0
epsilon_decay=0.995
epsilon_min=0.05
batch_size=32

detector=LaneChangeDetector()
env = DrivingEnv(frames, yolo_model, detector)

# 간단 학습 루프 (예시) # naive
num_episodes=3
for ep in range(num_episodes):
    state=env.reset()
    done=False
    total_reward=0
    while not done:
        if random.random()<epsilon: action=env.action_space.sample()
        else:
            with torch.no_grad():
                q=policy_net(torch.tensor(state,dtype=torch.float32))
                action=int(torch.argmax(q).item())
        next_state,reward,done,_ = env.step(action)
        buffer.append((state,action,reward,next_state,done))
        state=next_state
        total_reward+=reward
        
        if len(buffer)>=batch_size:
            batch=random.sample(buffer,batch_size)
            s,a,r,ns,d=zip(*batch)
            s=torch.tensor(s,dtype=torch.float32)
            a=torch.tensor(a)
            r=torch.tensor(r,dtype=torch.float32)
            ns=torch.tensor(ns,dtype=torch.float32)
            d=torch.tensor(d,dtype=torch.float32)
            q_values=policy_net(s).gather(1,a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                target_q=r+gamma*(1-d)*target_net(ns).max(1)[0]
            loss=nn.MSELoss()(q_values,target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epsilon=max(epsilon*epsilon_decay,epsilon_min)
    target_net.load_state_dict(policy_net.state_dict())
    print(f"Episode {ep+1} | Total Reward: {total_reward}")
