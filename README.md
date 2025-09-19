# 🌲🔥 AI-Driven Forest Fire Prediction  

An interactive **Streamlit web app** that predicts the **risk of forest fires** using **machine learning**.  
The app is powered by a **Random Forest Classifier**, trained on the **UCI Forest Fires Dataset**.  

---

## 🚀 Features  
- 🔹 Train and evaluate a Random Forest model  
- 🔹 Predict **High/Low fire risk** based on:
  - 🌡️ Temperature (°C)  
  - 💧 Humidity (%)  
  - 💨 Wind Speed (km/h)  
- 🔹 Display **prediction probability**  
- 🔹 Show **model accuracy**  
- 🔹 Visualize **feature importance** (temperature, humidity, wind)  
- 🔹 Explore the **raw dataset** in the app  

---

## 📂 Project Structure  

forest-fire-prediction/
│── forest_fire_app.py # Main Streamlit app
│── forestfires.csv # Dataset (UCI Forest Fires)
│── fire_risk_model.pkl # Saved trained model (auto-generated)
│── requirements.txt # Dependencies
│── README.md # Project documentation


---

## 📊 Dataset  

- Dataset: [UCI Forest Fires Dataset](https://archive.ics.uci.edu/ml/datasets/forest+fires)  
- Columns used for prediction:  
  - `temp` → Temperature (°C)  
  - `RH` → Relative Humidity (%)  
  - `wind` → Wind speed (km/h)  
- Target variable:  
  - `area` → Burned area (ha)  
  - Converted into binary: **1 = High Risk (area > 0)**, **0 = Low Risk (area = 0)**  

---

## ⚙️ Installation  

Clone the repository:  

```bash
git clone https://github.com/your-username/forest-fire-prediction.git
cd forest-fire-prediction

Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

Install dependencies:
pip install -r requirements.txt

▶️ Usage

Run the Streamlit app:
streamlit run app.py
The app will open in your browser at: http://localhost:8501/


