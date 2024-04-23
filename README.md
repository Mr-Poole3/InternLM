# 作业六作业及笔记

## 基础作业

### 完成 Lagent Web Demo 使用，并在作业中上传截图。

#### 安装环境

```python
mkdir -p /root/agent
studio-conda -t agent -o pytorch-2.1.2
cd /root/agent
conda activate agent
git clone https://gitee.com/internlm/lagent.git
cd lagent && git checkout 581d9fb && pip install -e . && cd ..
git clone https://gitee.com/internlm/agentlego.git
cd agentlego && git checkout 7769e0d && pip install -e . && cd ..
```
![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/21532ef0-25ed-4eef-9969-6b81fd2e0a48)



#### 安装lagent和agentlego

```
cd /root/agent
conda activate agent
git clone https://gitee.com/internlm/lagent.git
cd lagent && git checkout 581d9fb && pip install -e . && cd ..
git clone https://gitee.com/internlm/agentlego.git
cd agentlego && git checkout 7769e0d && pip install -e . && cd ..
```

![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/383ae793-e20b-41d5-8099-1031cf116ade)

![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/d0f8d1de-c9e8-4abc-9a7b-4f53b5d32373)

#### Lagent：轻量级智能体框架

启动一个 api_server

![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/8fc18a78-e1f8-4f11-9147-9741e3f0652f)

![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/20501d97-ec05-4dbf-9f10-a53d27ca7ef3)

### AgentLego：组装智能体“乐高”

#### 下载demo

```python
cd /root/agent
wget http://download.openmmlab.com/agentlego/road.jpg
conda activate agent
pip install openmim==0.3.9
mim install mmdet==3.3.0
```

![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/3b0e9adf-fddc-4a39-ae7d-18d31fc1113b)


#### 推理图片

进行推理。在等待 RTMDet-Large 权重下载并推理完成后，我们就可以看到如下输出以及一张位于 /root/agent 名为 road_detection_direct.jpg 的图片

```python
python /root/agent/direct_use.py
```
![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/fda1afe3-745e-47e3-b3da-d81c1dcb052d)

![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/9c979228-e527-4896-87e5-c71bfb6f8aed)

![image](https://github.com/Mr-Poole3/InternLM/assets/112788987/16662e20-3d97-41da-b248-7439b4010144)

# 笔记

本文介绍了AgentLego的使用方法，包括直接使用AgentLego作为智能体工具以及自定义工具两种方式。在直接使用AgentLego作为智能体工具的例子中，我们首先下载demo文件，然后安装目标检测工具所需的依赖，接着通过touch命令创建直接使用目标检测工具的python文件，并在文件中编写代码进行推理。在自定义工具的例子中，我们创建了一个调用MagicMakerAPI进行图像生成的工具，并将其注册到工具列表中。最后，我们在两个终端中分别启动LMDeploy服务和AgentLego的WebUI，以体验自定义工具的效果。

具体步骤包括：

1. 修改相关文件，例如使用 Mim 工具来安装所需的依赖。
2. 使用 LMDeploy 部署，可以通过修改配置文件和启动服务来实现。
3. 使用 AgentLego WebUI，可以通过启动服务和上传图片来使用工具。

作为智能体工具则需要先修改算法库的相关配置，例如修改模型 URL 和 Agent name，然后再配置工具并使用 AgentLego WebUI。此外，还可以使用 AgentLego 提供的自定义工具接口来构建自己的工具。

总的来说，AgentLego 提供了一个方便易用的工具集，可以帮助开发者快速搭建智能体应用，并支持自定义工具的开发。
