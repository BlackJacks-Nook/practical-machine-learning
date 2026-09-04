# Fundamentals of Machine Learning


:::{objectives}

- Describe representative types of ML (supervised, unsupervised, reinforcement learnings, *etc*.).
- Explain general workflow of a ML project.
- Introduce representative ML libraries.
:::


:::{instructor-note}

- 20 min teaching
- 10 min exercises/discussion
:::


## 1. Types of ML


ML can be broadly categorized into three main types depending on how models learn from input data and nature of input data they process.


:::{figure} ../images/fundamentals/ml-three-types.png
:align: center
:width: 80%

Figure: Three main types of ML. Main approaches include classification and regression under supervised learning and clustering under unsupervised learning. RL enhance model performance by interacting with environment. Coloured dots and triangles represent training data. Yellow stars represent new data which can be predicted by trained model. This figure was taken from paper [Machine Learning Techniques for Personalised Medicine Approaches in Immune-Mediated Chronic Inflammatory Diseases: Applications and Challenges](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2021.720694/full).
:::


### 1.1 Supervised learning


In supervised learning, model is trained on a *labeled* dataset, where each input is paired with a corresponding output (label).
- This approach is called **supervised** because learning process is guided by presence of labels, which act like a teacher providing correct answers during training.
- Goal is to learn a mapping from inputs to outputs to make predictions on new, unseen data.
- Supervised learning has two subtypes and representative examples of these two subtypes in real-word problems are listed below.
    - **Classification** is used when target variable is categorical (predicting discrete categories), such as email spam detection (spam/ham), image recognition (cat/dog), and medical diagnosis (disease/no disease), *etc*.
    - **Regression** is applied when target variable is continuous (predicting continuous values), such as predicting house prices based on features like size, location, and age.


### 1.2 Unsupervised learning


In unsupervised learning, model works with *unlabeled* data, which means that dataset contains only input features without associated labels.
- Goal is to identify hidden patterns, structures, or relationships within data without explicit guidance on what to predict.
    - In lots of real-world scenarios, labels may not always be available, and collecting labeled data can be expensive, time-consuming, or even impossible.
	- In such cases, we have to use unsupervised learning to uncover patterns and structure in data.
- Unsupervised learning is essential for analyzing vast amounts of raw data generated in real-world applications, from scientific research to business intelligence.
	- Its significance can be seen across several key areas.
	- **Exploratory Data Analysis** (EDA) can reveal natural groupings, trends, and correlations that might otherwise remain hidden, providing a crucial first step in any data-driven investigation.
	- **Anomaly Detection**
		- Unsupervised learning is vital for maintaining security and operational integrity.
		- By modeling "normal" behavior, algorithms can identify unusual patterns, such as fraudulent financial transactions, network intrusions, or rare mechanical failures, without needing labeled examples of every type of anomaly.
	- **Feature Engineering** and **Representation Learning**
		- Methods like PCA can compress data into its most informative components, reducing noise and improving efficiency and performance of downstream supervised models.
- Unsupervised learning also has two main subtypes, and representative examples of these two subtypes in real-word problems are listed below.
    - **Clustering** (grouping similar data points together): customer segmentation in marketing (grouping users by behavior), image segmentation (grouping similar pixels), *etc*.
    - **Dimensionality reduction** (simplifying data by reducing features while preserving important information): compressing high-dimensional data (*e.g.*, reducing image features for faster processing), anomaly detection, *etc*.


<details>
<summary><strong><mark>Challenges in unsupervised learning</mark></strong></summary>

:::{note}

- A major challenge in unsupervised learning is evaluating whether algorithm learned something useful.
	- Unsupervised learning algorithms are usually applied to data that does not contain any label information, so we don’t know what right output should be.
	- Therefore, it is very hard to say whether a model "did well".
		- *i.e.*, our hypothetical clustering algorithm could have grouped together all pictures that show faces in profile and all full-face pictures.
- As a consequence, unsupervised algorithms are used often in an exploratory setting, when a data scientist wants to understand data better, rather than as part of a larger automatic system.
- Another common application for unsupervised algorithms is as a preprocessing step for supervised algorithms.
	- Learning a new representation of data can sometimes improve accuracy of supervised algorithms, or can lead to reduced memory and time consumption.
- Before we start with "real" unsupervised algorithms, we will briefly discuss some simple preprocessing methods that often come in handy.
	- Even though preprocessing and scaling are often used in tandem with supervised learning algorithms, scaling methods don’t make use of supervised information, making them unsupervised.
:::
</details>


### 1.3 Reinforcement learning


Model (agent) learns by interacting with an environment.
- It takes actions, receives feedback (rewards or penalties), and learns a strategy (policy) to maximize long-term rewards.
- Representative examples of RL in real-word problems include game-playing AI (*e.g.*, AlphaGo), robot navigation, autonomous driving, *etc*.


### 1.4 Other subtypes


In addition to supervised and unsupervised learning, there are other important paradigms in ML.
- **Semi-supervised learning** bridges gap between supervised and unsupervised learning by using a small amount of labeled data together with a large amount of unlabeled data, helping models learn more effectively when labeling is expensive or time-consuming (*e.g.*, medical image analysis).
- **Self-supervised learning** is a form of unsupervised learning where model generates its own labels from data -- typically for pretraining models on tasks like image or language understanding, enabling them to learn robust representations without explicit labels (*e.g.*, predicting next word in a sentence, and filling in missing image patches).
- **Transfer learning** involves applying knowledge from a pretrained model, trained on a large, general dataset, to a new, related task, significantly reducing training time and data requirements (*e.g.*, fine-tuning a speech recognition model for a new dialect).

These techniques expand capabilities and versatility of ML across data-limited or computationally constrained environments.


## 2. ML Workflow


A ML workflow is a structured approach for developing, training, evaluating, and deploying ML models. It typically involves several key phases, including data collection, preprocessing, model training and evaluation, and finally, deployment to production.

Below is a graphical representation of a general workflow to train a model using ML algorithms. Detailed steps may vary depending on specific datasets and tasks.


:::{figure} ../images/fundamentals/ML-workflow.png
:align: center
:width: 100%

Figure: ML workflow.
:::


### 2.1 Business understanding


**Problem definition** (business understanding) is first and most critical phase of any ML project. It sets direction, scope, and goals forentire project.
- We should understand problem domain: what is real-world problem we are trying to solve? are we predicting, classifying, or grouping data? (*e.g.*, predict house prices, detect spam emails, cluster customers).
- We should determine if ML is appropriate solution for problem.
- We then should identify expected outputs: what will ML model produce? (*e.g.*, a number, a label, or a probability).
- We define type of ML task (*e.g.*, classification and regression tasks for supervised learning, clustering, dimensionality reduction for unsupervised learning, and decision-making tasks for RL).


<details>
<summary><strong><mark>Project Setup</mark></strong></summary>

:::{admonition} **Project Setup** is to set up programming/development environment for project

- Hardware requirements (CPU, SSD, GPU, cloud platforms, *etc.*).
- Software requirements (programming languages and libraries, ML/DL frameworks, and development tools, IDEs, Git/Docker, *etc*).
- Project structure: organize project for clarity and scalability.
- A typical ML project structure looks like this:

```
  ML_Project/
  ├── data/                 # raw and processed data
  │   ├── raw/              # original, unprocessed data
  │   ├── processed/        # cleaned, preprocessed data
  ├── notebooks/            # jupyter notebooks for EDA & modeling
  ├── src/                  # source code
  │   ├── utils/            # utility functions (*e.g.*, metrics, logging)
  │   ├── preprocessing.py  # data cleaning script  
  │   └── train.py          # model training script
  ├── models/               # trained model files (*e.g.*, .pkl, .h5)
  ├── tests/                # unit and integration tests
  ├── README.md             # project overview and setup instructions
  ├── requirements.txt      # project dependencies
  ├── config.yaml           # configuration file for hyperparameters and paths
```
:::
</details>


### 2.2 Data collection, preparation and processing


In ML, data collection, preparation and processing are crucial steps that significantly affect performance of a model. High-quality, well-processed data leads to better predictions, while poor data can result in unreliable models.
- **Data collection**: Gather necessary data from various sources (*e.g.*, databases, APIs (twitter, linkedin, *etc.*), or manual collection), and ensure that data is representative and sufficient for problem.
- **Data preparation** refers to activities performed to make raw data ready for analysis or modeling.
	- It focuses on data quality, structure, and usability.
	- Goal of this step is to produce a clean, consistent, and well-organized dataset.
	- Typical tasks include data cleaning (handling missing values (drop, impute, or predict), removing duplicates or irrelevant data, correcting errors and inconsistencies), data formatting and restructuring (converting data types, reshaping tables), feature selection, data splitting, and basic feature creation, and data labeling and annotation.
- **Data processing** refers to systematic transformation of data to extract information, patterns, or features.
	- It focuses on computation and transformation rather than cleanliness.
	- Goal of this step is to transform prepared data into a form suitable for analysis, modeling, or decision-making.
	- Typical tasks include data transformation (normalization and standardization, encoding categorical variables, aggregation and filtering), feature engineering (creating new features, dimensionality reduction), signal, image, or text processing (filtering signals, tokenizing text, extracting image features), batch or streaming processing.


:::{admonition} **One-sentence summary for typical ML pipeline**

- data preparation: making raw data clean, consistent, and usable.
    - data collection, data cleaning, feature/label separation, train/validation/test split, data formatting and labeling
- data processing: transforming prepared data into informative representations
    - normalization/standardization, encoding categorical variables, feature extraction, dimensionality reduction, data augmentation
:::


### 2.3 Model training, evaluation and assessment, and optimization


**Model selection and training** refer to process of choosing an appropriate model architecture and training it to learn patterns from data to solve a specific task. It involves selecting appropriate algorithms (*e.g.*, linear/nonlinear models, tree-based models, NN-based models, *etc*) based on problem type and data characteristics, configuring model hyperparameters, training constructed model on training dataset, and making predictions on testing dataset.


**Model evaluation and assessment** refers to a systematic process of measuring and analyzing a model's performance to determine its effectiveness in solving a specific task. 
- It involves using metrics and techniques to quantify how well a model performs and how reliably it generalizes to unseen data, identifies patterns, and meets desired objectives, typically using a test dataset separate from training data.
- Evaluation focuses on performance measurement using appropriate metrics, while assessment emphasizes comparative analysis, robustness, and decision-making about whether a model is acceptable for deployment or further improvement.
- Goal is not only to quantify predictive accuracy but also to understand a model’s strengths, limitations, and suitability for a given task.


Below are common evaluation metrics by task types:

| Task types | Evaluation metrics |
| :--------: | :----------------: |
| Classification | Accuracy, precision, recall, F1-score, ROC-AUC, *etc.* |
| Regression | Mean Squared Error (MSE), Mean Absolute Error (MAE), <br>Root Mean Squared Error (RMSE), R-squared, *etc.* |
| Clustering | Silhouette score (measures how similar a sample is to its own cluster compared to other clusters) for data without ground truth, Adjusted Rand Index (ARI) for data with ground truth |
| Dimensionality reduction | Reconstruction-based metrics (Reconstruction Error, Mean Squared Reconstruction Error), Structure preservation metrics (Trustworthiness, Continuity, Neighborhood Preservation Ratio) |


Representative techniques and processes for assessment include:
- **Comparison with baselines**: Comparing model performance against simple baselines (*e.g.*, random guessing, linear models) to ensure meaningful improvement.
- **Train-validation-test split**: Divide data into training (model learning), validation (hyperparameter tuning), and test (final evaluation) sets to prevent overfitting.
- **Cross-validation**: Use k-fold cross-validation to assess model stability across multiple data subsets.
- **Robustness testing**: Evaluate performance under noisy, adversarial, or out-of-distribution data.
- **Time-aware validation** is used to respect temporal order for time-dependent data. 
- Beyond numerical metrics, assessment may also involve error analysis, confusion matrix inspection, learning curves, and statistical significance testing to compare competing models.


**Model optimization** is process of systematically improving a ML model’s performance by adjusting factors that influence how it learns and generalizes. This involves selecting the right model type, tuning hyperparameters (like learning rate, number of layers, or batch size), choosing effective training algorithms, and applying regularization techniques to prevent overfitting. It also includes refining input features, ensuring high-quality data, and aligning loss functions with task’s goals. Ultimate aim is to find combination of model design, training strategy, and data preparation that maximizes performance on unseen data, balancing accuracy, efficiency, and robustness.


<details>
<summary><strong><mark>Representative model optimization strategies</mark></strong></summary>

:::{hint}

- Hyperparameter Tuning
    - Adjusting parameters that control model learning but are not learned from data, *e.g.*, learning rate, batch size, number of hidden layers, tree depth.
    - Methods: Grid search, random search, Bayesian optimization, Hyperband, population-based training.
- Regularization Techniques
    - Methods to prevent overfitting and improve generalization.
    - Examples: L1/L2 penalties, dropout, early stopping, label smoothing, data augmentation, noise injection.
- Neural Network Architecture & Algorithm Optimization
    - Designing effective network structures and selecting training strategies.
    - Includes: number/depth of layers, width of layers, activation functions, normalization layers (BatchNorm, LayerNorm), skip connections, choice of optimizer (SGD, Adam, RMSProp).
- Ensemble Methods
    - Combining multiple models to improve robustness and predictive performance.
    - Examples: Bagging (Random Forest), Boosting (Gradient Boosting, XGBoost, LightGBM, CatBoost), Stacking/Blending.
- Feature Engineering
    - Creating, selecting, or transforming features to improve model performance.
    - Examples: Dimensionality reduction (PCA, autoencoders), interaction features, feature selection (filter, wrapper, embedded methods).
- Loss Function & Evaluation Optimization
    - Choosing or customizing loss functions aligned with the task objective.
    - Examples: Cross-entropy, focal loss, Huber loss, weighted losses for imbalanced data; metric-driven optimization like AUC, F1-score.
- Data-Level Optimization
    - Improving data quality and representation.
    - Examples: Normalization, standardization, handling missing values, outlier treatment, balanced sampling.
- Computational & System-Level Optimization
    - Enhancing efficiency and scalability of model training.
    - Examples: Mixed precision training, parallelism (data/model), memory optimization, kernel fusion (GPU/CUDA).
:::
</details>


### 2.4 Model deployment, documentation, and maintenance


The lifecycle of a ML model extends beyond training to encompass deployment, documentation, and ongoing maintenance, ensuring model delivers reliable value in real-world applications.
- **Model deployment** involves integrating trained model into production environments, such as web services, mobile apps, or enterprise systems, so it can generate predictions on new, unseen data.
	- Deployment also requires attention to scalability, latency, reliability, and security, ensuring that model can handle production workloads efficiently.
	- Techniques such as containerization (Docker), model serving frameworks (TensorFlow Serving, TorchServe), and cloud platforms (AWS, Azure, GCP) are commonly used to streamline deployment and ensure reproducibility.
- **Documentation** is essential for transparency, reproducibility, and collaboration.
	- It includes recording model’s design decisions, data preprocessing steps, feature selection methods, hyperparameter configurations, training procedures, evaluation metrics, and known limitations.
	- Good documentation allows other developers, data scientists, or stakeholders to understand how model works, reproduce results, and troubleshoot issues effectively.
- **Model maintenance** is an ongoing process of monitoring and updating a deployed model to ensure its continued performance over time.
	- ML models can degrade due to changing data distributions, concept drift, or evolving business requirements.
	- Maintenance activities include monitoring performance metrics, detecting drift in data or concept, retraining models with new data, updating hyperparameters or architectures, and addressing issues such as bias or ethical concerns.
	- Automating monitoring pipelines and establishing clear update schedules are critical to maintaining reliable and accurate predictions in production environments.

Together, these stages form a continuous, structured process that maximizes utility, reliability, and longevity of ML models in production.


## 3. ML Libraries


There are numerous libraries available for ML that simplify process of building, training, and deploying models.
- These libraries provide ready-to-use implementations of algorithms, tools for data preprocessing, utilities for model evaluation, and even frameworks for large-scale DL.
- Some are optimized for beginners with easy-to-use APIs, while others are designed for high performance and flexibility, making them suitable for research and production environments alike.
- By leveraging these libraries, practitioners can focus more on solving problems and less on reinventing basic components.


### 3.1 Scikit-learn


**Scikit-learn = SciPy Toolkits**, is a widely-used, open-source Python library designed for **classical ML**, providing a variety of ready-to-use implementations of supervised and unsupervised algorithms and tools for tasks, such classification, regression, clustering, and dimensionality reduction, through a simple and **consistent interface**.
- It is built upon **SciPy stack**, which involves NumPy, SciPy, Matplotlib, Pandas, and it is designed for ease of use, making it ideal for beginners and rapid prototyping.
    - Other such common libraries are *scikit-image* and *statsmodels*.
- Scikit-learn was initially started in 2007 as a Google Summer of Code project by David Cournapeau, who has also been involved in development of NumPy and SciPy.
	- By 2010, more developers were starting to get involved, and 1st public release was made in Feb. 2010.
- It supports supervised learning (*e.g.*, SVM, decision trees, random forests), unsupervised learning (*e.g.*, k-means, PCA), and semi-supervised learning, with robust tools for data preprocessing, model evaluation, and hyperparameter tuning via `GridSearchCV`.
- Scikit-learn excels in handling small to medium-sized datasets and includes utilities for data preprocessing, model evaluation, hyperparameter tuning, and pipeline construction.
    - However, it lacks support for DL and GPU acceleration, limiting its scalability for large datasets or complex NN tasks.
- All of scikit-learn's implementations run in memory on a single machine.
	- There are solutions being developed that allow scikit-learn to scale to multiple machines, such as `Dask`.
	- Many scikit-learn algorithms allow parallel execution using `joblib`, which natively provides thread-based and process-based parallelism.
	- Dask can scale these joblib-backed algorithms out to a cluster of machines by providing an alternative joblib backend.


<details>
<summary><strong><mark>Three primary APIs</mark></strong></summary>

:::{note}

Scikit-learn is designed in a way to have similar interfaces across functionalities, and it is organized around three primary APIs, estimator, predictor, and transformer.
- **Estimators** are core interface implemented by classification, regression, clustering, and dimensionality reduction methods.
	- An estimator is initialized from hyperparameter values and implements actual learning process in `fit` method, which you call while providing input data and labels in form of `X_train` and `y_train` arrays.
	- It will run learning algorithm to learn parameters and store them for future use.
	- In addition to this, estimator also offers other complementary tasks, like:
    	- Feature extraction, which involves transforming input data into numerical features that can be used for ML purposes.
    	- Feature selection, which selects features in your data that most contribute to prediction output of model.
- **Predictors** provide a predict method to take data which needs to be predicted through a NumPy array that we usually refer to as `X_test`.
	- It applies required transformation with respect to parameters that have been learned by `fit` method and provides predicted values or labels.
	- Some unsupervised learning estimators provide a predict method to obtain cluster labels.
    - In addition to predicting, predictor can also implement methods that are in charge of quantifying confidence of prediction, also called model performance.
        - These confidence functions vary from model to model, but their main objective is to determine how far prediction is from reality.
        - This is done by taking an `X_test` with its corresponding `Y_test` and comparing it to predictions made with same `X_test`.
- **Transformer** interfaces implement mechanism to transform given data in form of NumPy array through preprocessing and feature extraction stages.
	- Scaling and normalization methods implement transform method which can be called after learning parameters.
	- Advantage of transformer is that once it has been applied to training dataset, it stores values used for transforming training data.
    	- This can be used to transform test dataset to same distribution.
:::
</details>


<details>
<summary><strong><mark>Ecosystem of Scikit-Learn</mark></strong></summary>

:::{warning}

- Scikit-learn, venerable Python library, stands as a cornerstone in ML landscape.
- Renowned for its simplicity and efficiency, scikit-learn is go-to toolkit for those beginning their journey into realm of ML.
- Library's ecosystem is an embodiment of versatility, encompassing a wide array of ML tasks.
- One of its greatest strengths is its seamless integration with broader python scientific stack.
- Libraries such as numpy and pandas interlock perfectly with scikit-learn.
- Its design principles make it an ideal teaching tool that align with pedagogical aim of demystifying complexities of ML algorithms.
- It is not merely a collection of algorithms, it is a compendium of state-of-art techniques that address a multitude of ML challenges.
:::
</details>


<details>
<summary><strong><mark>Advantages of Scikit-Learn</mark></strong></summary>

:::{hint}

- ease of use
	- scikit-learn is characterized by a clean API, with a small learning curve in comparison to other libraries such as TensorFlow or Keras
	- API is popular for its uniformity and straightforward approach
	- users of scikit-learn do not necessarily need to understand math behind models
- uniformity
	- its uniform API makes it very easy to switch from model to model, as basic syntax required for one model is same for others
- documentation/tutorials
	- library is completely backed up by documentation, which is effortlessly accessible and easy to understand
	- additionally, it also offers step-by-step tutorials that cover all topics required to develop any ML project
- reliability and collaborations
	- as an open source library, scikit-learn benefits from inputs of multiple collaborators who work each day to improve its performance
	- this participation from many experts from different contexts helps to develop not only a more complete library but also a more reliable one
- coverage
	- as you scan list of components that library has, you will discover that it covers most ML tasks, ranging from supervised models such as classification and regression algorithms to unsupervised models such as clustering and dimensionality reduction
	- moreover, due to its many collaborators, new models tend to be added in relatively short amounts of time
:::
</details>


### 3.2 Keras


**Keras** is a high-level NNs API that simplifies the process of building and training DL models.
- Originally an independent library, Keras is now tightly integrated with TensorFlow as its official high-level interface (but also usable standalone), offering an accessible way to experiment with DL without sacrificing performance.
- Keras provides user-friendly abstractions for layers, models, loss functions, and optimizers, allowing users for quick prototyping of NNs for tasks like image classification, text generation, and time series forecasting with minimal code.
- Keras abstracts away much of complexity of TensorFlow while retaining flexibility, making it ideal for beginners and those who need fast experimentation.


### 3.3 TensorFlow


Developed by Google, **TensorFlow** is a powerful open-source library primarily for DL but versatile enough for a broad range of ML tasks.
- It provides a flexible ecosystem for building complex models, including NNs for CV, NLP, and time series analysis.
- TensorFlow supports distributed computing across CPUs, GPUs, and TPUs, making it suitable for both research and production at scale.
- Its robust features, such as TensorBoard for visualization, TensorFlow Serving for model deployment, and TensorFlow Lite for mobile inference, make it a comprehensive framework for end-to-end ML development.
- TensorFlow’s high-level Keras API simplifies model building, while its low-level operations provide flexibility for advanced research.
- TensorFlow is well-suited for tasks like image recognition, NLP, and RL, though its complexity can pose a steeper learning curve for beginners compared to alternatives like PyTorch.


### 3.4 PyTorch


PyTorch is developed by Facebook’s AI Research Lab (FAIR), is auser-friendly and open-source DL library that has gained significant popularity in academia and industry.
- Known for its intuitive design and "define-by-run" (eager execution) approach, PyTorch allows developers to build, train, and debug models in a flexible and interactive manner.
- Its strong support for GPU acceleration and extensive ecosystem-ranging from CV (`torchvision`) to NLP (`torchtext`) and audio (`torchaudio`) -- make it an excellent choice for cutting-edge DL research and production.
- Popular in academia and increasingly in industry, PyTorch excels in rapid prototyping and experimentation but is less optimized for production deployment compared to TensorFlow.
- Its active community and support for GPU acceleration make it a favorite for cutting-edge ML and DL research.


### 3.5 XGBoost & LightGBM


**XGBoost** (Extreme Gradient Boosting) and **LightGBM** (Light Gradient Boosting Machine) are high-performance gradient boosting libraries that have become go-to solutions for structured data problems, such as tabular datasets.
- Both libraries implement optimized gradient boosting algorithms that deliver fast training speeds, high accuracy, and scalability to large datasets.
- XGBoost is known for its robustness and versatility, while LightGBM offers further speed and memory efficiency through histogram-based algorithms and leaf-wise growth strategies.
- These libraries have become essential tools for data scientists working with structured data, outperforming traditional models in many real-world scenarios.


### 3.6 Hugging Face Transformers


**Hugging Face Transformers** is a cutting-edge library that provides access to state-of-the-art pre-trained models for NLP tasks and CV, including text classification, translation, summarization, and question answering.
- Library’s pre-trained models and tokenizers simplify NLP workflows by enabling rapid experimentation with large language models, and in addition, this library supports both TensorFlow and PyTorch backends, integrating with datasets via Hugging Face’s datasets library, and has a vibrant community contributing to its continuous development.


### 3.7 FastAI


**FastAI** is a high-level DL library built on PyTorch, designed to make AI accessible to a wider audience by simplifying complex tasks.
- It provides high-level abstractions and best practices out-of-the-box, allowing users to train powerful models with minimal code and optimal defaults.
- FastAI is particularly well-known for its transfer learning capabilities, enabling quick adaptation of pre-trained models for tasks like image classification and text generation.
- With its focus on practical usage, education, and strong community support, FastAI is ideal for beginners and practitioners who want to quickly deploy models without deep theoretical expertise.


### 3.8 JAX


**JAX**, developed by Google, combines NumPy-like syntax with automatic differentiation and GPU/TPU acceleration, making it ideal for high-performance ML research.
- It enables composable function transformations (gradients, JIT compilation) and scales efficiently across hardware.
- While not as high-level as TensorFlow or PyTorch, JAX is favored for cutting-edge numerical computing, physics simulations, and advanced NN research where speed and flexibility are crucial.


### 3.9 Summary of ML libraries


These libraries cater to different needs:
- Scikit-learn for classical ML, TensorFlow and PyTorch for DL and scalability, Keras for simplicity, XGBoost for high-performance tabular data tasks, and Hugging Face for transformer-based applications.
- Choice of these libraries depends on task, data type, scalability needs, user expertise, and whether focus is research, prototyping, or production deployment.
- A summary of best features and key strengths of these libraries are summarized below.


| Library | Best Feature | Key Strength |
| :-----: | :----------: | :----------: |
| Scikit-Learn | Simple and consistent API <br>for classical machine learning tasks  <br>(classification, regression, clustering) <br>and small/medium datasets | Seamless integration with NumPy/Pandas <br>and extensive documentation for ease-of-use with <br>wide algorithm support |
| PyTorch | Dynamic computation graph (define-by-run) <br>for flexible model building and debugging | Flexible, intuitive framework with strong adoption <br>for academic research in DL tasks |
| TensorFlow | Scalability with GPU/TPU acceleration <br>for complex deep learning models | Excellent ecosystem (Keras, TF Hub, TF-Agents) <br>for production-scale applications |
| Keras | High-level, user-friendly API <br>for rapid prototyping | Simplifies construction of DL models, making it beginner-friendly <br>and efficient with TensorFlow compatibility <br>for quick model development |
| XGBoost & <br>LightGBM | Optimized gradient boosting algorithms | Extremely effective for high-performance <br>supervised learning with tabular/structured data |
| Hugging Face <br>Transformers | Extensive pretrained transformer models <br>for easy fine-tuning | Community-driven ecosystem with user-friendly pipelines <br>for NLP and vision tasks |
| FastAI | Transfer learning made easy <br>for NLP & vision tasks | Fast prototyping with minimal code and strong performance <br>for applied DL |
| JAX | NumPy + autodiff + GPU/TPU acceleration | Cutting-edge numerical computing, works with PyTorch/TensorFlow <br>via interoperability libraries, but offers lower-level control |

