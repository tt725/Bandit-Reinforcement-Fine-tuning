export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

TRAIN_DATA_DIR=/root/autodl-tmp/new-verl/rlft-data/gsm8k-result/gsm8k/train.parquet
VAL_DATA_DIR=/root/autodl-tmp/new-verl/rlft-data/gsm8k-result/gsm8k/test.parquet
MODEL_DIR=/root/autodl-tmp/new-verl/pretraining-model/Qwen2.5-0.5B-Instruct


python3 -m verl.trainer.main_ppo_replay \
    algorithm.adv_estimator=asyttgrpo \
   +data.window_size=8 \
    data.train_files=$TRAIN_DATA_DIR \
    data.val_files=$VAL_DATA_DIR \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_DIR \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
   +algorithm.norm_adv_by_std_in_grpo=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='RLFT-V3.1' \
    trainer.experiment_name='asyttgrpo_0512_1_8_Qwen0.5B_math' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=100 \
    trainer.total_epochs=1000 $@ > asyttgrpo_0512_1_8_Qwen0.5B_math.log 2>&1 &