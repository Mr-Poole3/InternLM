# -
# 第一节课视频笔记

## **模型分为专用模型和通用大模型；**

专用模型：针对特定的任务的模型，但是一个模型只能解决一种任务（例：人脸识别）

通用模型：一个模型对应多种任务和多种模态（例：chat-gpt）

就好比普通的阿尔法狗就只能下象棋所以是专用模型，但是如果它能够一边下象棋，一边和你进行聊天对话，那么它就被称之为通用模型。

## **大模型的本质：回归语言模型建模**

**自回归语言模型**

第一次听到自回归语言模型（Autoregressive LM）这个词。我们知道一般的语言模型都是从左到右计算某个词出现的概率，但是当我们做完型填空或者阅读理解这一类NLP任务的时候词的上下文信息都是需要考虑的，而这个时候只考虑了该词的上文信息而没有考虑到下文信息。所以，反向的语言模型出现了，就是从右到左计算某个词出现的概率，这一类语言模型称之为自回归语言模型。像坚持只用单向Transformer的GPT就是典型的自回归语言模型，也有像ELMo那种拼接两个上文和下文LSTM的变形自回归语言模型。

**自编码语言模型**

自编码语言模型（Autoencoder LM）这个名词毫无疑问也是第一次听到。区别于上面所述，自回归语言模型是根据上文或者下文来预测后一个单词。那不妨换个思路，我把句子中随机一个单词用[mask]替换掉，是不是就能同时根据该单词的上下文来预测该单词。我们都知道Bert在预训练阶段使用[mask]标记对句子中15%的单词进行随机屏蔽，然后根据被mask单词的上下文来预测该单词，这就是自编码语言模型的典型应用。

**新一代数据清洗过滤技术**

1. 多维度数据价值评估

   基于文本质量、信息质量、信息密度等维度对数据价值进行综合评估与提升

2. 高质量语料驱动的数据富集

   利用高质量语料的特征从物理世界、互联网以及语料库中进一步富集更多类似语料

3. 有针对性的数据补齐

   针对性补充语料，重点加强世界知识数理、代码等核心能力

## 如何应用

通过接入InternLM2大模型实现例如智能客服，聊天助手等AI智能应用。

## 大模型微调

1. 增量续训

   可以让模型学习到垂直领域知识（书籍，代码等）

2. 有监督微调

   可以令模型理解各种指令并进行对话（高质量的对话，问答数据等）
