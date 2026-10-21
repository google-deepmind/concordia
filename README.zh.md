# Concordia

*面向生成式社会模拟的智能体开源库 (A library for generative social simulation)*

<!-- GITHUB -->
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[![Python](https://img.shields.io/pypi/pyversions/gdm-concordia.svg)](https://pypi.python.org/pypi/gdm-concordia)
[![PyPI version](https://img.shields.io/pypi/v/gdm-concordia.svg)](https://pypi.python.org/pypi/gdm-concordia)
[![PyPI tests](../../actions/workflows/pypi-test.yml/badge.svg)](../../actions/workflows/pypi-test.yml)
[![Tests](../../actions/workflows/test-concordia.yml/badge.svg)](../../actions/workflows/test-concordia.yml)
[![Examples](../../actions/workflows/test-examples.yml/badge.svg)](../../actions/workflows/test-examples.yml)
<!-- /GITHUB -->

[技术报告 (Tech Report)](https://arxiv.org/abs/2312.03664) | [设计模式 (Design Pattern)](https://arxiv.org/abs/2507.08892) | [代码速查表 (Code Cheat Sheet)](CHEATSHEET.md)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

## 关于 Concordia (About)

Concordia 是一个用于构建和运行生成式智能体基模型（generative agent-based models）的开源库，旨在模拟实体在接地的物理、社会或数字真实环境中的复杂交互。它采用了一种灵感源自桌面角色扮演游戏（TRPG）的交互范式：由一个被称为 **主持人 (Game Master, GM)** 的特殊实体来模拟玩家实体所交互的整个环境。实体用自然语言描述其意图行动，而 GM 则将这些意图转化为合理的结果，例如在模拟世界中检验行动的物理合理性。

Concordia 支持广泛的应用领域，包括社会科学研究、AI 安全与伦理、认知神经科学、计算经济学、用于个性化的合成数据生成，以及通过模拟使用对真实服务进行性能评测。

Concordia 需要接入标准的大语言模型 (LLM) API，并可按需与外部应用程序和服务进行无缝集成。

## 运行机制 (How it Works)

模拟系统由 **实体 (Entities)** 构成。每个实体由提供具体行为与状态的 **组件 (Components)** 组成，如指令、记忆与行动选择。**预制体 (Prefabs)** 是预构建好的实体模板。通过实例化预制体，覆写实体名称、指令或目标等参数，其标准工厂方法将根据这些配置自动构建对应组件。

**引擎 (Engine)** 负责组织与调度执行。在每次 `run_loop` 调用中，实体被赋予玩家（Player）或主持人（Game Master）角色。引擎请求玩家实体做出行动决策，并要求主持人实体提供观测反馈并结算这些行动。这些角色是在该次调用中的动态职责分配，而非实体固有的类型类别。

建议从 [GUI 与 CLI 实体组件指南](concordia/command_line_interface/README.md) 入手，学习如何检查组件、编辑模拟环境并观察其智能体行为。

## 目录结构 (Folder Structure)

*   **[`concordia/prefabs`](concordia/prefabs/README.md)**：实体及其组件的预组装起始配置模板。
*   **[`concordia/components`](concordia/components/README.md)**：实体的模块化构建积木，包括记忆系统与感知模块。
*   **[`concordia/environment`](concordia/environment/README.md)**：包含决定回合制（turn-taking）、并发行动（simultaneous action）或异步行动（asynchronous action）的引擎接口。
*   **[`concordia/document`](concordia/document/README.md)**：用于管理 LLM 提示词与上下文组装的实用工具。
*   **[`concordia/language_model`](concordia/language_model/README.md)**：LLM 接入支持与 API 封装层。
*   **[`examples/`](examples/)**：快速上手教程与示例模拟场景。

## 教程资源 (Tutorials)

1.  侧重于 GUI 编辑器与 CLI 界面交互的教程：[Concordia GUI 与 CLI 教程](https://concordia-tutorial.pages.dev/)。
2.  YouTube 深度视频讲座：[Concordia: Building Generative Agent-Based Models](https://youtu.be/2FO5g65mu2I?si=TSk7XTk4gCaadEDs)，搭配 [`examples/tutorial.ipynb`](examples/tutorial.ipynb) 运行。

## 安装指南 (Installation)

[Concordia 已在 PyPI 发布](https://pypi.python.org/pypi/gdm-concordia)，可通过以下命令安装：

```shell
pip install gdm-concordia
```

安装完成后，即可在自己的代码中执行 `import concordia`。

## 开发指南 (Development)

### Codespace 云端开发环境

参与 Concordia 源码开发最便捷的方式是使用我们预先配置好的 [GitHub Codespace](https://github.com/features/codespaces) 开发环境。

它提供了经过严格验证、可复现的开发工作流，最大程度减少了依赖管理的复杂度。我们强烈建议所有向 Concordia 提交的 Pull Request 都通过该工作流准备。

### 本地手动配置 (Manual setup)

如果您希望在自己的本地开发环境中对 Concordia 源码进行二次开发，需要自行管理安装与依赖环境。

例如，您可以通过以下步骤进行可编辑安装（Editable installation）：

1.  克隆 Concordia 仓库：

    ```shell
    git clone -b main https://github.com/google-deepmind/concordia
    cd concordia
    ```

2.  创建并激活 Python 虚拟环境：

    ```shell
    python -m venv venv
    source venv/bin/activate
    ```

3.  安装 Concordia 开发依赖：

    ```shell
    pip install --editable .[dev]
    ```

4.  运行测试以验证安装：

    ```shell
    pytest --pyargs concordia
    ```

5.  安装所需的其他语言模型依赖，例如：

    ```shell
    pip install .[google]
    pip install --requirement=examples/requirements.in
    ```

    请注意在此阶段，您可能会遇到某些底层依赖不支持您当前开发环境的情况，需要进行相应的依赖排查与适配。

## 自带大模型 API (Bring your own LLM)

Concordia 需要接入大语言模型 API。任何支持文本采样的 LLM API 均可搭配使用，模拟结果的质量取决于所选模型的能力。此外，您还需要提供用于联想记忆（Associative memory）的文本向量模型（Text embedder）。任何固定维度的 Embedding 均可使用，建议选用适用于句子相似度或语义检索的 Embedding 模型。

## 典型应用示例 (Example usage)

以下是一个极具代表性的社会模拟示例：4 位好友被大雪困在一家酒馆中，其中两人因一起汽车碰撞事故发生争端。

智能体基于 March 和 Olsen（2011）的经典推理框架构建，该理论指出人类通常通过回答三个核心问题来决定自己的行为：

1. 这是怎样的一种情境？
2. 我是一个怎样的人？
3. 像我这样的人，在这样的情境下会怎么做？

以下示例中的智能体完整实现了这三道核心思考逻辑：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/concordia/blob/main/examples/tutorial.ipynb)

## 论文引用 (Citing Concordia)

如果在研究中使用了 Concordia，请引用以下学术论文：

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

```bibtex
@article{vezhnevets2023generative,
  title={Generative agent-based modeling with actions grounded in physical,
  social, or digital space using Concordia},
  author={Vezhnevets, Alexander Sasha and Agapiou, John P and Aharon, Avia and
  Ziv, Ron and Matyas, Jayd and Du{\'e}{\~n}ez-Guzm{\'a}n, Edgar A and
  Cunningham, William A and Osindero, Simon and Karmon, Danny and
  Leibo, Joel Z},
  journal={arXiv preprint arXiv:2312.03664},
  year={2023}
}
```

## 免责声明 (Disclaimer)

这不是一款得到 Google 官方支持的产品 (This is not an officially supported Google product)。

如需完整的交互式构建演练以及详细的编辑器/CLI 功能覆盖，请参阅 [GUI 与 CLI 实体组件指南](concordia/command_line_interface/README.md)。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月21日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
