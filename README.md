
# Credit Card Sampling Analysis  

## 📌 Project Overview  
This project focuses on addressing the challenge of imbalanced credit card transaction data by applying and evaluating various sampling techniques. By testing multiple machine learning models on these sampled datasets, the goal is to identify the most effective combination of sampling method and model for accurately predicting fraudulent transactions.  

---

## 🧪 Sampling Techniques  
The following sampling techniques were implemented and compared:  
- **Simple Random Sampling**: Randomly selects a subset of the dataset without replacement.  
- **Stratified Sampling**: Ensures proportional representation of each class by dividing the dataset into homogeneous subgroups.  
- **Cluster Sampling**: Divides the dataset into clusters, then selects entire clusters randomly.  
- **Systematic Sampling**: Samples data at fixed intervals, starting from a random point.  
- **Multistage Sampling**: Combines multiple sampling methods for more complex sampling schemes.  

---

## 🤖 Machine Learning Models  
The performance of the following machine learning models was evaluated on the sampled datasets:  
- **Random Forest Classifier**  
- **Gradient Boosting Classifier**  
- **Logistic Regression**  
- **Support Vector Machine (SVM)**  
- **K-Nearest Neighbors (KNN)**  

---

## 📂 Dataset  
The dataset used contains anonymized credit card transaction data, including:  
- **Features**: 'V1' to 'V28' (PCA-transformed features), along with 'Time' and 'Amount'.  
- **Target Variable**: 'Class', where:  
  - `1`: Fraudulent transactions  
  - `0`: Non-fraudulent transactions  

---

## 🔧 Requirements  
This project is implemented in Python. To set up the environment, install the required libraries:  

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```  

---

## 🚀 How to Run  
1. **Clone the repository** or download the project files.  
2. **Prepare the dataset**: Place your dataset in the root directory or update the file path in the script.  
3. **Execute the script**: Run the analysis using the command:  
   ```bash
   python sampling.py
   ```  
4. **View the results**:  
   - The performance metrics are saved in the output file: `results.csv`.  
   - This file provides a pivot table with accuracy scores, allowing a clear comparison of models across sampling techniques.  

---

## 📊 Results and Insights  
The results are saved in a pivot table format in the `results.csv` file:  
- **Rows**: Represent different machine learning models.  
- **Columns**: Represent various sampling techniques.  
- **Cells**: Contain accuracy scores, enabling straightforward comparison and analysis of performance.  

---

## 💡 Key Takeaways  
This project highlights the importance of sampling techniques in handling imbalanced datasets. It also demonstrates how the choice of a machine learning model can influence the accuracy and reliability of predictions, especially in the context of fraud detection.  

---

## 📫 Contact  
Feel free to reach out for questions, suggestions, or collaborations!  
