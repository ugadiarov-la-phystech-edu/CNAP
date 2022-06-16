# CNAP
Continuous Neural Algorithmic Planner

To run main_classic.py for OpenAI's classic control suite:

```
python3 main_classic.py --env=mountaincar-continuous --seed=1 --graph_type=erdos-renyi --gnn_steps=1 --value_loss_coef=1 --lr=0.001 --include_transe=True --include_executor=True --action_bins=10 --device=cuda
```

To run main_mujoco.py for MuJoCo suite:

```
python3 main_mujoco.py --env=halfcheetah --seed=1 --use_gae=True --lr_decay=True --save_model=True --enable_time_limit=True --num_total_train_steps=1000000 --include_executor=True --gnn_steps=1 --num_neighbours=11 --cat_method=encoder_cat_executor --sample_method=manual_gaussian --record_video=True --save_dir=logs --device=cuda
```

To see a full list of arguments: arguments.py 
