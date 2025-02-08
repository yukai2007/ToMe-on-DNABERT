# ToMe-on-DNABERT
**此项目仅用于科研实训的成果展示。**

https://cn.overleaf.com/read/hpwmrnfxqtpg#d2e743

出于方便考虑，这边并不会详细介绍如何复现 ToMe on BERT 的流程。大致的方法是：先下载 DNABERT2 模型以及 GUE 测试集（https://github.com/MAGICS-LAB/DNABERT_2/ 这里有详细的介绍），然后找到其中的 `bert_layers.py` 和 `bert_padding.py` 这两个文件，用此项目中的两个对应文件替换之。`/fintune` 文件夹里的那些文件同理。之后在本地安装 tome 这个库（此项目中已经给出了 `setup` 文件），便可以运行 `/scripts` 里面的脚本文件。

+ `run_dnabert2_with_r=0.sh` 是采用 ToMe on DNABERT 架构，但是不进行 token 合并；
+ `run_dnabert2_without_ToMe.sh` 是原始 DNABERT 架构；
+ `run_dnabert2_with_r=0.sh` 是采用 ToMe on DNABERT 架构，进行 token 合并。

我自己电脑上的测试结果如下：
