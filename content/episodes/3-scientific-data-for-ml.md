# Scientific Data for Machine Learning


:::{objectives}

- Describe key characteristics of big data -- 4Vs.
- Gain an overview of scientific data with types and forms of data.
- Explain representative data storage formats and their pros and cons.
- Understand data structures for ML/DL.
:::


:::{instructor-note}

- 20 min teaching
- 20 min exercises/discussion
:::


## 1. Big Data


Big Data refers to datasets that are so large, complex, or fast-changing that traditional data processing tools cannot handle them efficiently.
- It encompasses not only sheer **volume** of data but also its **variety**, **velocity**, and **veracity** --- often summarized as **4 Vs** of big data.
	- These datasets can come from numerous sources, including social media, sensor networks, scientific experiments, and transactional systems.
	- Ability to collect and analyze such massive amounts of information allows organizations and researchers to uncover trends, correlations, and insights that would be impossible to detect with smaller datasets.
- Emergence of big data has transformed multiple domains, from business analytics and healthcare to climate science and genomics.
	- Advanced computational methods, distributed storage systems, and parallel processing frameworks such as Hadoop and Spark have become essential for managing and analyzing these vast datasets.
	- Efficient handling of big data enables organizations to make data-driven decisions, optimize operations, and identify opportunities for innovation.


In context of ML, big data provides raw material that fuels learning process.
- Large and diverse datasets allow ML models to capture complex patterns, generalize well to unseen data, and improve predictive performance.
	- Without sufficient and high-quality data, even most sophisticated algorithms cannot perform effectively.
	- From image recognition to NLP, every ML application depends on properly curated datasets for training, validation, and testing.
- Therefore, data is backbone of ML as it serves as foundation for training models to recognize patterns, make predictions, and generate insights.
	- In addition, data determines applicability and scalability of ML solutions across domains, from scientific research to real-world applications.


## 2. Understanding Scientific Data


Scientific data refers to any form of data that is collected, observed, measured, or generated as part of scientific research or experimentation.
- This data is used to support scientific analysis, develop theories, and validate hypotheses.
- It can come from a wide range of sources, including experiments, simulations, observations, or surveys across various scientific fields.
- In general, scientific data can be described ty two terms: **types of data** and **forms of data**.
- They are related but distinct
	- types describe nature of the data
	- forms describe how data is structured and formatted (and stored, which will be discussed below)


### 2.1 Types of scientific data


Types of scientific data refer to what data represents, and it focuses on nature or category of data content.
- At lowest level (Hardware- and system-level view), all data are represented as **bits** and **bytes**.
	- smallest unit of storage in a computer is a **bit**, which holds either a 0 or a 1
	- typically, eight bits are grouped together to form a **byte**
    - `01000001` → `A` → a character in text
    - `01000001` → `65` → an integer
- At a higher, semantic level, data can take forms such as numerical data, text data, and metadata.
    - **numerical data** are commonly represented as integers and floating-point numbers.
        - *Integers* are used to represent exact, discrete quantities, such as counts, indices, labels, or class identifiers, and they support precise arithmetic without rounding error.
        - *Floating-point numbers*, by contrast, represent real-valued quantities with fractional components and are essential for modeling continuous phenomena. Because floating-point representations are finite approximations of real numbers, they introduce rounding errors, making numerical stability an important consideration in scientific and ML computations.
            - In ML/DL, numerical precision plays a critical role in balancing accuracy, memory usage, and computational efficiency.
            - Common floating-point precisions include half-accurate, single precision (32-bit), double precision (64-bit).
            - *Single precision* is most widely used format in ML and DL, as it provides sufficient numerical accuracy while enabling efficient computation on modern hardware such as GPUs.
            - *Half precision* reduces memory consumption and increases training speed, particularly for large NNs, but requires careful handling to avoid numerical underflow or overflow.
            - *Double precision* offers higher numerical accuracy and stability, making it valuable for scientific simulations and sensitive calculations, though it is generally more computationally expensive.
            - In practice, ML and DL systems balance these precision levels to achieve an optimal trade-off between performance and accuracy.
            - Many modern frameworks, like TensorFlow and PyTorch, support mixed precision training, combining half and single precision to optimize performance while maintaining stability.
    - **Text data** represent information in form of characters, words, and symbols and are one of most prevalent data types in modern data analysis.
        - They arise in a wide range of scientific and real-world contexts, including research articles, medical records, social media content, system logs, and biological sequences.
        - Although text data are naturally meaningful to humans, they are inherently unstructured from a computational perspective and must be systematically processed (converted into numerical representations) before they can be used in ML/DL models.
        - Such process usually starts with basic text preprocessing steps such as tokenization, normalization, and building a vocabulary, and then moves on to feature extraction methods like bag-of-words or TF-IDF.
            - More advanced approaches use *embeddings*, where words, subwords, or even characters are represented as dense numerical vectors that capture semantic and syntactic relationships.
            - These representations allow models to understand patterns in language beyond simple keyword matching.
        - Modern DL models, particularly NNs such as RNNs, transformers, and LLMs, learn rich and contextualized representations of text directly from data.
            - These models can capture long-range dependencies, contextual meaning, and linguistic structure, making them effective for tasks such as text classification, information retrieval, machine translation, summarization, and question answering.
            - As a result, text data play a central role in ML and DL, bridging human language and computational intelligence through carefully designed representations and models.
    - **Metadata** are *data about data*, and they provide information that describes, explains, or gives context to primary data, such as units (`°C`), timestamps (`2026-01-03 10:00:00`), identifiers (`Sensor_ID = T-017`), and other descriptive attributes.
        - Metadata do not represent main content itself, but instead tell us what data are, how they were generated, and how they should be interpreted.
        - In ML/DL, metadata play a critical role throughout data and model lifecycle. They help define features and labels, document dataset structure, and record how data are split into training, validation, and testing sets.
        - Contextual information such as timestamps, locations, device types, or user attributes can complement primary data like images, text, or signals and significantly improve model performance.
        - As a result, metadata are not only foundational for data governance and transparency but also an important source of contextual knowledge in modern machine learning and deep learning systems.


### 2.2 Forms of scientific data


Forms of scientific data refer to a way data are organized, represented, and stored, which affects how they can be accessed, analyzed, and processed. Depending on their structure and format, scientific data are typically classified into three main forms: structured data, semi-structured data, and unstructured data. Each form has different characteristics, advantages, and challenges for storage, analysis, and application in scientific research and machine learning.
- **Structured data** are highly organized and follow a predefined schema, usually stored in rows and columns in databases or spreadsheets.
	- Each data element has a fixed data type and format, which makes it easy to search, sort, and analyze using standard statistical methods or ML algorithms.
- **Semi-structured data** have some organizational structure but do not conform strictly to a fixed schema.
	- They contain tags, labels, or metadata that make them partially organized, enabling flexible storage and processing.
	- Examples: JSON or XML files storing experimental logs
	- A Jupyter NB (.ipynb) is primarily a file format that stores code, outputs, text, and metadata in JSON format, so NB itself is essentially semi-structured data.
- **Unstructured data** lack a predefined structure or schema and are not organized in rows or columns.
	- They often consist of raw, complex data that require special processing techniques to extract meaningful information.
	- Examples: Text documents, Images, Audios, Video, Raw sensor signals
	- They are rich in information but require advanced processing to extract meaningful features for ML and DL tasks.


| Form | Characteristics | Examples | Typical Use in ML/DL |
| :--: | :-------------: | :------: | :------------------: |
| Structured | Highly organized, fixed schema | Tables, CSV, spreadsheets | Most classical ML algorithms |
| Semi-structured | Partially organized, flexible schema | JSON, XML, logs | Integrated date |
| Unstructured | No fixed structure | Text, images, audio, video | DL, NLP, CV |


## 3. Data Storage Format


A data storage format defines how data is structured, stored, and represented so it can be efficiently read, processed, and shared between systems or applications.
- There are many types of storage formats used in scientific computing and data analysis.
- **There isn’t one data storage format that works in all cases, so choose a file format that best suits your data**.
- Choice of format affects storage efficiency, speed of access, compatibility with tools, and ease of use for humans.


Below is an overview of common data formats (✅ for *good*, 🟨 for *ok/depends on a case*, and ❌ for *bad*) adapted from Aalto university's [**Python for scientific computing**](https://aaltoscicomp.github.io/python-for-scicomp/work-with-data/#what-is-a-data-format).

| Name | Human <br>readable | Space <br>efficiency | Arbitrary <br>data | Tidy <br>data | Array <br>data | Long term <br>storage/sharing |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| CSV           | ✅ | ❌ | ❌ | ✅ | 🟨 | ✅ |
| Excel         | ❌ | ❌ | ❌ | 🟨 | ❌ | 🟨 |
| JSON          | ✅ | ❌ | 🟨 | ❌ | ❌ | ✅ |
| Pickle        | ❌ | 🟨 | ✅ | 🟨 | 🟨 | ❌ |
| npy           | ❌ | 🟨 | ❌ | ❌ | ✅ | ❌ |
| Feather       | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| HDF5          | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| NetCDF4       | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Parquet       | ❌ | ✅ | 🟨 | ✅ | 🟨 | ✅ |
| Graph formats | 🟨 | 🟨 | ❌ | ❌ | ❌ | ✅ |


## 4. Data Structures for ML/DL


Data structures form backbone of ML tasks, and main data structure for ML/DL is **tensor**.
- A tensor is a multi-dimensional array that provides a unified way to represent and manipulate data of different shapes and complexities.
    - **It generalizes scalars, vectors, and matrices to higher dimensions, serving as fundamental data structure in frameworks like TensorFlow and PyTorch**.
	- *Scalars* (single numbers) are considered 0-dimensional tensors, *vectors* are 1-dimensional tensors, and *matrices* are 2-dimensional tensors.
	- When extended to higher dimensions, tensors can naturally model more complex data types such as color images (3D: height × width × channels), videos (4D: frames × height × width × channels), or batches of training samples.
- Tensors are central to ML/DL because they align closely with mathematics that underpins most algorithms.
	- They serve as format for both input data (such as feature vectors or images) and model parameters (such as weights in a NN).
	- Furthermore, operations on tensors, such as addition, multiplication, reshaping, and convolution, are implemented in highly optimized ways by libraries such as NumPy, PyTorch, and TensorFlow.
	- These frameworks make it possible to process large datasets efficiently by leveraging hardware acceleration on GPUs and TPUs.


:::{figure} ../images/data/varied-data-structures.png
:align: center
:width: 80%

Figure: Varied data structures.
:::


:::{note} Why to use tensors in ML/DL (advantages of Tensor)?

- Generalization of scalars/vectors/matrices: tensors extend these concepts to any number of dimensions, which is essential for handling complex data like images (3D) and videos (4D+).
- Consistency: tensors unify data structures across ML/DL frameworks, simplifying model building, training, and deployment.
- Efficient computation: frameworks like TensorFlow and PyTorch optimize tensor operations for speed (using GPUs/TPUs).
- NN representations: input data (images, text) is converted to tensors.
- Automatic differentiation: tensors support gradient tracking, which is vital for **backpropagation** in NNs.
:::


:::{exercise}

In [**Jupyter Notebook**](../tutorial/3-tensor.ipynb), we provide a tutorial about tensors including
- tensor creation
- tensor's properties (`shape`, `dtype`, `ndim`)
- tensor operations
   - indexing, slicing, transposing
   - element-wise operations: addition, subtraction, *etc.*
   - matrix multiplication(`np.dot`, `torch.matmul`)
   - reshaping, flattening, squeezing, unsqueezing
   - reduction operations: `sum`, `mean`, `max` along axes
   - broadcasting: rules and examples
- tensors in DL frameworks
   - moving tensors between CPUs and GPUs (suppose that you can access to GPU cards)
- converting images (grayscale and color) into tensors
:::

