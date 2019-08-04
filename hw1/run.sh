#env_name='Ant-v2'
#env_name='HalfCheetah-v2'
#env_name='Hopper-v2'
#env_name='Humanoid-v2'
#env_name='Reacher-v2'
env_name='Walker2d-v2'

#python run_expert.py experts/$env_name.pkl $env_name --num_rollouts=20 --save_data
#python behavioral_cloning.py --env_name=$env_name --mode=train
python behavioral_cloning.py --env_name=$env_name --mode=run