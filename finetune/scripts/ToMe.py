import subprocess
import os

def activate_conda_env(env_name):
    activate_command = f"conda activate {env_name}"
    try:
        print(f"Activating conda environment: {env_name}")
        subprocess.run(activate_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while activating the conda environment: {e}")

# 切换到目标工作目录
os.chdir("/usr/data/yuchang/yk/ToMe-main/DNABERT_2-main/finetune")

# activate_conda_env("DNABERT")

# 定义数据路径和学习率
data_path = "/usr/data/yuchang/yk/DNABERT_2/DNABERT_2-main"
lr = "3e-5"

print(f"The provided data_path is {data_path}")
print(f"Current working directory is {os.getcwd()}")  # 打印当前工作目录

# 定义种子
seeds = [42]

# 定义数据类型
data_types = [
    "H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me3", "H3K9ac", "H4", "H4ac"
]

# 定义模型配置文件、保存路径等参数
model_name_or_path = "zhihan1996/DNABERT-2-117M"
output_dir = "output/DNABERT2_ToMe"
save_steps = 200
eval_steps = 200
warmup_steps = 50
logging_steps = 100000
num_train_epochs = 3
per_device_train_batch_size = 4
per_device_eval_batch_size = 8
gradient_accumulation_steps = 1
fp16 = True
overwrite_output_dir = True
find_unused_parameters = False

# 定义函数来运行命令
def run_command(command):
    try:
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the command: {e}")

# 迭代种子和数据类型，构造并运行命令
for seed in seeds:
    for data in data_types:
        # 构建命令
        run_name = f"DNABERT2_{lr}_EMP_{data}_seed{seed}"
        command = f"python train.py " \
                  f"--model_name_or_path {model_name_or_path} " \
                  f"--data_path {os.path.join(data_path, 'GUE', 'EMP', data)} " \
                  f"--kmer -1 " \
                  f"--run_name {run_name} " \
                  f"--seed {seed} " \
                  f"--model_max_length 128 " \
                  f"--per_device_train_batch_size {per_device_train_batch_size} " \
                  f"--per_device_eval_batch_size {per_device_eval_batch_size} " \
                  f"--gradient_accumulation_steps {gradient_accumulation_steps} " \
                  f"--learning_rate {lr} " \
                  f"--num_train_epochs {num_train_epochs} " \
                  f"--fp16 " \
                  f"--save_steps {save_steps} " \
                  f"--output_dir {output_dir} " \
                  f"--evaluation_strategy steps " \
                  f"--eval_steps {eval_steps} " \
                  f"--warmup_steps {warmup_steps} " \
                  f"--logging_steps {logging_steps} " \
                  f"--overwrite_output_dir {overwrite_output_dir} " \
                  f"--log_level info " \
                  f"--find_unused_parameters {find_unused_parameters}"

        # 运行命令
        run_command(command)
