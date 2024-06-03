# Comparative Study of Machine Learning Algorithms Relevance Classification Task 

## Objective
The goal of this project was to build a machine learning model capable of determining whether documents from a given dataset were relevant to specified topics. This required careful data preparation and preprocessing to ensure that the model was trained on the most relevant and clean data, leading to a more accurate classification outcome.

## Dataset Structure
The dataset consisted of multiple columns containing information about documents and their metadata, such as document identifiers, author names, document bodies, bylines, titles, topic information, and relevance judgments. To streamline the dataset, we identified and removed unnecessary columns that did not directly contribute to our classification task. These columns included 'author', 'byline', 'topic_id', and 'topic_title', as well as an unnamed column used only as a row index.

### Data Preprocessing
To prepare the text data for machine learning, we applied a series of preprocessing steps to standardize the content and remove irrelevant information. Our approach included:

- **Text Cleaning and Tokenization**: All text was converted to lowercase, and then tokenized into individual words. This step helped ensure consistency and allowed for further processing.
- **Stop Words Removal and Lemmatization**: Common English stop words were removed to reduce noise, and words were lemmatized to standardize their base forms. This step helped reduce dimensionality and improved the semantic relevance of the text.
- **Column Combination**: We combined the 'title', 'byline', 'body', 'description', and 'narrative' columns into a single 'combined_text' column, allowing us to work with a unified representation of the document content.

## Handling Missing Data
Given the presence of missing values in the 'combined_text' field, we applied a basic imputation strategy to fill in these gaps. This ensured that no valuable data was lost during preprocessing. The missing values were replaced with an empty string to maintain a consistent data structure.

## Feature Engineering
To convert the text data into a format suitable for machine learning, we employed a Term Frequency-Inverse Document Frequency (TF-IDF) vectorization approach. This technique transformed the text into a numerical representation, allowing us to capture the significance of individual words within the documents. With a TF-IDF vectorizer set to a maximum of 10,000 features, we generated dense arrays for our training and validation datasets.

## Data Splitting
To facilitate model training and validation, we split the dataset into training and validation subsets. Using the 'judgment' column as the target variable, we created an 80/20 models split, ensuring that the training set had a balanced representation of relevant and non-relevant documents. The stratification helped maintain the proportional distribution of the target variable across both sets.

## Machine Learning Model Development


### Standard ML Baseline
Logistic regression is a commonly used algorithm for binary classification tasks due to its simplicity and efficiency. In this project, we utilized GridSearchCV to optimize logistic regression's hyperparameters. Our tuning process focused on regularization strength (`C`), type of regularization (`penalty`), and solver (`solver`). The cross-validation process, with an F1-score-based evaluation, allowed us to balance precision and recall. This model provided a solid baseline for performance, demonstrating that logistic regression can be effective with proper parameter tuning.

### 3-Layer Neural Network 
We designed a 3-layer feed-forward neural network to explore the impact of varying neuron counts in the hidden layers. The model had a dense input layer, one hidden layer, and a dense output layer with a sigmoid activation function for binary classification. We experimented with neuron counts of 5,10,20and 40 in the middle hidden layer, aiming to identify an optimal configuration. Each model included dropout layers to prevent overfitting and was trained with the Adam optimizer and binary cross-entropy loss. The results indicated that a moderate number of neurons 5 to 10 in the hidden layer could achieve stable and accurate results without significant overfitting.

### Deep Neural Networks
We further explored deeper neural network architectures by experimenting with models with 1 to 7 hidden layers. These models had similar configurations, with varying numbers of hidden layers and dropout layers to avoid overfitting. The results showed that deeper networks did not necessarily improve accuracy, emphasizing the importance of balanced architecture rather than simply adding more layers. This finding underscores the need to carefully select the number of hidden layers based on the data and the task at hand.
In addition to varying neuron counts and layer depths, we explored the effects of different activation functions on model performance. We compared "relu," "sigmoid," and "tanh" activation functions, finding that the choice of activation function had a minimal impact on the overall performance in this context. This suggests that other factors, such as regularization and model architecture, play a more significant role in determining the model's accuracy and stability.

### Complex Neural Network
To build a complex neural network for text classification, we first explored the use of Gated Recurrent Units (GRUs). Our approach involved creating a simple architecture with a single GRU layer, followed by a dropout layer for regularization, and a dense output layer with a sigmoid activation function for binary classification. We experimented with various configurations, changing the number of GRU units (16, 32, 64, and 128) to determine the optimal structure. This iterative process allowed us to assess how the complexity of the GRU layer affected the model's validation accuracy over a series of 10 training epochs. Our goal was to find the best-performing GRU model that balanced computational cost and accuracy.

In a subsequent step, we developed a more complex neural network by incorporating a Bidirectional GRU.This model configuration added bidirectional processing to the GRU layer, allowing the network to consider sequence data from both forward and backward directions. The rest of the architecture remained consistent, with a dropout layer for regularization and a dense output layer with a sigmoid activation function for binary classification. By training this model over 10 epochs and tracking validation accuracy, we explored whether the bidirectional approach provided significant benefits over the standard GRU model. The objective was to evaluate whether this increased complexity in the neural network architecture translated into improved performance in text classification tasks.
