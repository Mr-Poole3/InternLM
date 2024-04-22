# 第五课作业复现及笔记

## 环境部署

### 安装环境

```python
studio-conda -t lmdeploy -o pytorch-2.1.2
```
![Snipaste_2024-04-22_22-37-01](https://github.com/Mr-Poole3/InternLM/assets/112788987/4a6326e1-8d0a-4f67-a281-b4775ed81824)

![2](https://github.com/Mr-Poole3/InternLM/assets/112788987/e602084e-b95b-49dd-9cef-2c116b43a1f1)


### 安装0.3版本的lmdeploy。

![3](https://github.com/Mr-Poole3/InternLM/assets/112788987/3da7632e-6a71-4bb8-9200-6b3b9e77c53c)


### 新建pipeline_transformer.py文件

![4](https://github.com/Mr-Poole3/InternLM/assets/112788987/0f2fdef7-722b-444a-920b-01152b620454)


### 测试模型

![5](https://github.com/Mr-Poole3/InternLM/assets/112788987/7c26df38-3a27-4bf4-ab62-2bd217e2e69c)


![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/6c8e3735-98b1-428d-9480-d3933c2b859d)


可以看出，这个推理也是占用了巨大内存

## 使用LMDeploy与模型对话

![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/de65d254-3ac2-4359-a9c6-4ab15e62d93a)


### 以命令行方式与InternLM2-Chat-1.8B大模型对话

![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/19ba8f5c-0e4c-4440-a037-de215db7d243)


## 笔记

本文介绍了LMDeploy环境部署的步骤和方法，包括创建开发机、conda环境，安装LMDeploy，以及使用LMDeploy进行模型对话、模型量化和服务部署等。文章还介绍了如何使用LMDeploy运行视觉多模态大模型和第三方大模型，并比较了LMDeploy与Transformer库的推理速度差异。

### 关键要点

1. LMDeploy环境部署包括创建开发机和安装LMDeploy。
2. 使用LMDeploy与模型对话，可以使用Transformer库运行模型并使用LMDeploy与模型对话。
3. LMDeploy模型量化包括设置最大KV Cache缓存大小和使用W4A16量化。
4. LMDeploy服务包括启动API服务器和命令行客户端连接API服务器。
5. Python代码集成包括Python代码集成运行1.8B模型和向TurboMind后端传递参数。
