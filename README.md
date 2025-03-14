# 📊 Data Mining & Classification Analysis  

## 📌 Project Overview  
This project applies **data mining techniques** to analyze ECG signal data and extract meaningful insights. It involves **data preprocessing, feature selection, clustering, and classification** using machine learning algorithms. The primary goal is to improve predictive accuracy and identify patterns within the dataset.  

## 📊 Dataset  
- **Source:** ECG signal dataset (CSV format)  
- **Features:** Multiple physiological attributes related to ECG signals  
- **Preprocessing:** Outlier detection, normalization, and duplicate removal  

## 🔍 Methodology  
### **1️⃣ Data Preprocessing**  
- Converted all data to **float format** for consistency  
- **Removed duplicates & handled missing values**  
- **Outlier detection** using the **IQR method** and replaced them with median values  
- **Normalized features** using **Min-Max Scaling**  

### **2️⃣ Feature Selection & Extraction**  
- **Correlation Analysis**: Removed highly correlated features  
- **SelectKBest (Chi-Square Test)**: Ranked features by importance  
- **Recursive Feature Elimination (RFE)**: Identified key predictive attributes  
- **PCA (Principal Component Analysis)**: Reduced dimensionality while preserving key patterns  

### **3️⃣ Clustering Techniques**  
- **K-Means Clustering**: Used the **Elbow Method** and **Silhouette Score** for optimal k selection  
- **Agglomerative Hierarchical Clustering**: Applied Ward’s method for distance-based grouping  

### **4️⃣ Classification Models**  
- **Logistic Regression**: Evaluated with confusion matrix and accuracy score  
- **K-Nearest Neighbors (KNN)**: Applied different k values for classification performance  
- **Support Vector Machine (SVM)**: Hyperparameter tuned using GridSearchCV  
- **Decision Tree Classifier**: Built using Gini impurity and entropy criteria  

## 📈 Model Evaluation  
- **Confusion Matrix & Classification Report** for each classifier  
- **Accuracy Score** comparison across models  
- **Heatmaps & Visualizations** for better insights  

## 🚀 Results & Conclusion  
- **Feature selection improved model performance** by reducing noise  
- **K-Means and Hierarchical clustering** effectively grouped ECG patterns  
- **SVM and Logistic Regression** performed well on classification tasks  
- **GridSearchCV tuning** enhanced model accuracy  

## ⚙️ Technologies Used  
- **Python**  
- **Pandas, NumPy, Matplotlib, Seaborn**  
- **Scikit-learn** (for feature selection, clustering, and classification)  
- **Keras & TensorFlow** (for deep learning exploration)  

## 📌 How to Run the Project  
1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```  
2. **Install dependencies**  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
   ```  
3. **Run the notebook**  
   ```bash
   jupyter notebook datamining.ipynb
   ```  

## 📝 Future Improvements  
- **Integrate deep learning models** for better ECG signal classification  
- **Apply time-series analysis** for sequential data interpretation  
- **Optimize clustering & classification models** for real-world deployment  

📌 **Contributors:**  
- **Arkala Chandhra Shekar Jyothi Vinay**  
- **Peram Divya**  
