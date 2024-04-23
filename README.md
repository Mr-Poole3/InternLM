# 第七课作业及笔记

## 使用 OpenCompass 评测 internlm2-chat-1_8b 模型在 C-Eval 数据集上的性能。

### 创建环境

```python
studio-conda -o internlm-base -t opencompass
source activate opencompass
git clone -b 0.2.4 https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

![image-20240423125230786](C:\Users\28402\AppData\Roaming\Typora\typora-user-images\image-20240423125230786.png)

**如果pip install -e .安装未成功,请运行:**

```python
pip install -r requirements.txt
```

### 解压评测数据

![image-20240423145407881](C:\Users\28402\AppData\Roaming\Typora\typora-user-images\image-20240423145407881.png)

### 查看支持的数据集和模型

![image-20240423145740569](C:\Users\28402\AppData\Roaming\Typora\typora-user-images\image-20240423145740569.png)

### 启动测评

![image-20240423151704909](C:\Users\28402\AppData\Roaming\Typora\typora-user-images\image-20240423151704909.png)

# 笔记

本文介绍了OpenCompass平台，该平台发布了大模型开源开放评测体系，旨在为大语言模型和多模态模型提供一站式评测服务。OpenCompass提供了丰富的模型支持和功能，包括70多个数据集和约40万题的模型评测方案。该平台具有全面的能力维度设计，包括语言、知识、理解、推理和安全等多个能力维度的评测。此外，OpenCompass还提供了客观评测和主观评测两种方式来评估模型的性能。文章还介绍了模型的客观评测和主观评测的具体步骤和指导。最后，文章讨论了数据污染评估和大海捞针测试两种评估方法，以进一步评估大模型的能力和性能。

## 关键要点

1. OpenCompass 2.0 是一个大模型开源开放评测体系，可用于为大语言模型、多模态模型等提供一站式评测服务。
2. 其主要特点是开源可复现、全面的能力维度、丰富的模型支持、分布式高效评测和多样化评测范式。
3. OpenCompass 提供了丰富的功能支持自动化地开展大语言模型的高效评测。
4. 它的设计思路是从通用人工智能的角度出发，结合学术界的前沿进展和工业界的最佳实践，提出一套面向实际应用的模型能力评价体系。
5. 在评测过程中，采用了提示词工程和语境学习进行客观评测，并使用真实人类专家的主观评测与基于模型打分的主观评测相结合的方式开展模型能力评估。1
