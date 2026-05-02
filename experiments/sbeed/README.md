# SBEED Experiments

This folder is for notebooks and scripts that use the reusable implementation in
`src/rl_methods/sbeed`.

Minimal online discrete example. Each `solver.run(...)` call starts from a
fresh replay buffer:

```python
from rl_methods.sbeed import (
    DiscreteMDPSpec,
    SBEEDSolver,
    TabularStateActionFeatures,
    TabularStateFeatures,
)

mdp_spec = DiscreteMDPSpec(
    n_states=N,
    n_actions=A,
    gamma=gamma,
    value_features=TabularStateFeatures(N),
    rho_features=TabularStateActionFeatures(N, A),
    x0=x_0,
)

solver = SBEEDSolver(
    spec=mdp_spec,
    buffer_mode="fifo",      # or "growing"
    max_buffer_size=5_000,   # required for FIFO
    batch_size=64,
    lambda_entropy=0.01,
    eta=1.0,
    ridge=1e-6,
    lr_value=1e-2,
    lr_policy=1e-2,
    device=DEVICE,
    seed=SEED,
)

def reward_fn(s, a, sp):
    return 1.0 if int(sp) == goal_grid else 0.0

pi = solver.run(
    transition_fn=next_state,   # may return just next_state
    reward_fn=reward_fn,
    episodes=100,
    collect_per_episode=20,
    updates_per_episode=10,
    initial_collect_steps=50,   # uniform warmup
    start_state=x_0,
    behavior="policy",
    epsilon=0.0,
    terminal_states={goal_grid},
    tqdm_print=True,
    verbose=True,
    log_every=10,
)
```
