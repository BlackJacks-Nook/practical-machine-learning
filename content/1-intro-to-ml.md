# Introduction to Machine Learning


:::{objectives}

- Provide a general overview of ML and why we need ML.
- Trace historical evolution from AI to ML and DL.
- Explain relationship between AI, ML, DL.
- Explore why should use ML and list representative real-world applications of ML.
:::


:::{instructor-note}

- 15 min teaching
- 5 min exercises/discussion
:::


## 1. What is ML


ML is a field of computer science that studies algorithms and techniques enabling computers to learn patterns from data and make predictions or decisions **without being explicitly programmed**, thereby automating solutions to complex problems that are difficult to solve using conventional programming methods.
- In **conventional programming**, programmer explicitly codes logic (rules) to transform inputs (data) into outputs (answers), making it suitable for well-defined, rule-based tasks.
- In **machine learning**, system learns logic (rules) from data and answers, making it ideal for complex, pattern-based tasks where explicit rules are hard to define.
- Choice between them depends on problem, data availability, and complexity.


:::{figure} ./images/fundamentals/classic-programming-vs-ml.jpg
:align: center
:width: 80%

Figure: [Classic programming *vs.* machine learning](https://twimlai.com/resources/kubernetes-for-mlops/).
:::



## 2. Why ML?


ML is transforming how we solve complex problems in real world by enabling systems to learn directly from data, rather than relying on explicitly programmed rules. In many real-world scenarios, such as medical diagnosis, stock market prediction, or NLP, the relationships between inputs and outputs are too complex or dynamic to define manually. ML models can uncover hidden patterns and make accurate predictions or decisions, making them essential tools in fields like healthcare, finance, transportation, and cybersecurity.

Another crucial advantage of ML is its ability to adapt and improve over time as more data becomes available. Unlike traditional rule-based systems that require constant manual updates, ML models can retrain and adjust themselves to new data, trends, or anomalies, ensuring that the system stays relevant and effective. For example, in fraud detection, ML algorithms can evolve as fraud tactics change, providing a stronger defense compared to static rules that may become outdated. This adaptability makes ML particularly powerful in dynamic, real-time environments where traditional programming methods fall short.

In addition, ML empowers automation of complex tasks that were previously dependent on human expertise and intuition. From voice recognition in virtual assistants to autonomous driving, ML algorithms can process vast amounts of unstructured data such as text, images, and audio, which are traditionally challenging for computers to handle. By enabling machines to "learn" from experience and improve their performance over time, ML not only enhances productivity but also opens new frontiers for innovation across industries, creating smarter systems that can make meaningful contributions to society.


## 3. Relation with AI and DL


:::{figure} ./images/fundamentals/relationship-ai-ml-dl.png
:align: center
:width: 50%

*Figure: [Relationship between AI, ML,and DL](https://carpentries-lab.github.io/deep-learning-intro/).*
:::


AI is broadest field, encompassing any technique that enables computers to mimic human intelligence, such as reasoning, problem-solving, perception, and decision-making. AI includes a wide range of approaches, from rule-based systems (like expert systems) to modern data-driven methods. It aims to create systems that can perform tasks that typically require human intelligence, such as playing chess, recognizing images, or understanding language.

ML is a subset of AI that focuses on algorithms and models that learn patterns from data to make predictions or decisions **without being explicitly programmed**. ML is one of primary ways to achieve AI. It enables systems to improve performance over time by learning from experience (data) rather than relying solely on hardcoded rules. ML includes various techniques like supervised learning (*e.g.*, regression, classification), unsupervised learning (*e.g.*, clustering, dimensionality reduction), and reinforcement learning.

DL is a specialized subset of ML, and it leverages artificial neural networks (NNs) inspired by human brain to tackle tasks like image recognition, speech processing, and natural language understanding. DL excels in handling unstructured data (*e.g.*, images, audio, text) and requires significant computational power and large datasets for training.


## 4. ML in a Bigger Picture


ML has been evolving in stages of increasing complexity in following four clearly differentiated steps.
- 1st model of ML involved **rule-based decisions** and a simple level of data-based algorithms that includes in itself, and as a prerequisite, all possible ramifications and decision rules, implying that all possible options will be hardcoded into model beforehand by an expert in field.
	- This structure was implemented in majority of applications developed since first programming languages appeared in 1950.
	- Main data type and function being handled by this kind of algorithm is Boolean, as it exclusively dealt with yes or no decisions.
- During 2nd developmental stage of **statistical reasoning**, we started to let probabilistic characteristics of data have a say, in addition to previous choices set up in advance.
	- This better reflects fuzzy nature of real-world problems, where outliers are common and where it is more important to take into account nondeterministic tendencies of data than rigid approach of fixed questions.
	- This discipline adds to mix of mathematical tools elements of Bayesian probability theory.
		- Methods pertaining to this category include curve fitting (usually of linear or polynomial), which has common property of working with numerical data.
- **ML stage** is realm in which we are going to be working throughout this book, and it involves more complex tasks than simplest Bayesian elements of previous stage.
	- Most outstanding feature of ML algorithms is that they can generalize models from data but models are capable of generating their own feature selectors, which aren't limited by a rigid target function, as they are generated and defined as training process evolves.
	- Another differentiator of this kind of model is that they can take a large variety of data types as input, such as speech, images, video, text, and other data susceptible to being represented as vectors.
- **AI** is last step in scale of abstraction capabilities that, in a way, include all previous algorithm types, but with one key difference: AI algorithms are able to apply learned knowledge to solve tasks that had never been considered during training.
	- Types of data with which this algorithm works are even more generic than types of data supported by ML, and they should be able, by definition, to transfer problem-solving capabilities from one data type to another, without a complete retraining of model.
	- In this way, we could develop an algorithm for object detection in black and white images and model could abstract knowledge to apply model to color images.
- Following diagram represents these four stages of development towards real AI applications.


:::{figure} /images/fundamentals/ml-in-big-picture.png
:align: center
:width: 80%

*Figure: ML in a big picture.*
:::


## 5. Three Booms and Two Winters of AI


AI从20世纪中叶至今经历了**三次高潮(起)和两次低谷(落)**
- **第一次起(1956年-1970年左右)**
	- 背景：1956年达特茅斯会议被认为是AI正式诞生
		- 那时研究者提出"让机器像人一样思考"的愿景
	- 代表进展：符号主义(Symbolic AI)、逻辑推理、简单搜索算法（如井字棋、积木世界）
	- 原因：计算机刚出现，人们对其潜力抱有极大期待，初期的小规模演示让大家误以为"通用智能"指日可待
- **第一次落(1970年代中后期)**
	- 原因
	- 算力不足：硬件能力远远达不到复杂推理的需求
	- 算法局限：符号主义方法只能处理玩具问题，无法扩展到现实世界
	- 资金削减：美国和英国政府减少了对AI资助(英国"莱特希尔报告"影响很大)
- **第二次起(1980年代中期-1990年代初)**
	- 背景：专家系统(Expert Systems)兴起，基于规则的知识库开始在工业和医学中应用
	- 代表进展：XCON系统(帮助配置计算机系统)、逻辑推理在企业中获得实际收益
	- 原因：商业应用推动，大量投资涌入，AI被认为终于要走向实用
- **第二次落(1990年代中后期)**
	- 原因：
	- 知识获取瓶颈：专家系统需要人工输入海量规则，难以维护和扩展
	- 系统脆弱：只在特定场景下有效，无法适应变化
	- 过度宣传：炒作过高预期，实际效果不佳，导致`第二次AI寒冬`
- **第三次起(2006年至今)**
	- 背景：深度学习崛起，结合大数据和GPU算力带来突破
	- 代表进展：
		- 2006年Hinton提出深度置信网络(DBN)
		- 2012年AlexNet在ImageNet大赛上击败传统方法
		- 之后语音识别、图像识别、自然语言处理全面爆发
	- 原因：
		- 算力飞跃(GPU、TPU)
		- 大数据驱动
		- 算法改进(深度神经网络、反向传播优化)
		- 产业推动(互联网巨头+开源框架)
- **总结**
	- 第一次落：符号主义方法 + 算力不足
	- 第二次落：专家系统的知识获取瓶颈 + 脆弱性 + 过度炒作
	- 第三次起：数据、算力、算法三要素齐备，推动深度学习爆发


:::{figure} ./images/fundamentals/three-booms-two-winters-of-ai.jpg
:align: center
:width: 80%

*Figure: Three booms and two winters of AI.*
:::


## 6. ML Applications


ML is used across a wide range of industries and real-world problems in healthcare, finance, CV, NLP, transportation, manufacturing industry, retail, and cybersecurity. Below are key categories of problems that can be applied using ML/DL.

| Application area | Example use Cases |
| :--------------: | :---------------: |
| Healthcare | Disease prediction & diagnosis, <br>medical image analysis, drug discovery |
| Finance | Fraud detection, credit scoring, algorithmic trading
| Retail & e-commerce | Product recommendations, customer segmentation, <br>demand forecasting |
| Transportation & autonomous systems | Self-driving cars, traffic prediction, route optimization |
| Natural language processing (NLP) | Chatbots and virtual assistants, sentiment analysis, <br>language translation |
| Manufacturing & industry | Predictive maintenance, quality control, <br>supply chain optimization |
| Computer Vision | Facial recognition, object detection, image classification |
