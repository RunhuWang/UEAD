TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/home/fqz02/Works/WangRunhu/demo03/navsim_workspace/navsim/spmultiplanner_99_spmll.ckpt
CACHE_PATH=/home/fqz02/Works/WangRunhu/demo02/navsim_workspace/exp/metric_cache

python /home/fqz02/Works/WangRunhu/demo02/navsim_workspace/DiffusionDrive/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=spmultiplanner_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=spmultiplanner_agent_V1 \
metric_cache_path=$CACHE_PATH \
