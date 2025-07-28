# KRISHI-JNAN

# KRISHI-JÑĀNA 🌾

**Krishi-Jñāna** (कृषि-ज्ञान) is a smart agricultural web platform designed to empower farmers by recommending the most suitable crops based on their local conditions **and** predicting expected **market prices at harvest time**. This two-stage system blends soil science and data-driven forecasting to improve farm decisions and increase profitability.


---

## 🔍 Key Features

### 🌱 Stage 1: Crop Recommendation
- Input: District, Soil Type, and Sowing Month
- Recommends the most **agronomically suitable crops** for that region and time

### 📈 Stage 2: Price Forecasting
- Predicts **future market prices** of the recommended crops at harvest time
- Uses **LSTM deep learning model** trained on historical modal prices

---

## 🛠️ Technologies Used

| Tool / Library     | Purpose                          |
|--------------------|----------------------------------|
| `Streamlit`        | Web UI framework                 |
| `Pandas`, `NumPy`  | Data processing                  |
| `Scikit-learn`     | Encoding, feature transformation |
| `TensorFlow/Keras` | LSTM model for price prediction  |
| `Matplotlib/Seaborn`| Data visualization              |
| `Excel`            | Raw agricultural and market data |

---

## 💻 How It Works

1. **User inputs**:
   - District
   - Soil Type
   - Sowing Month

2. System:
   - Recommends the best-suited crops for the given conditions
   - Maps to a **base district** for market price analysis
   - Predicts **harvest-time price** using pre-trained LSTM model

3. Output:
   - A table of crops with **estimated modal prices**
   - Helps farmers pick **the most profitable crop**

---

## 🖥️ Live Demo (Streamlit Cloud)

[![Open in Streamlit](https://krishi-jnan-ny9cqkayjob9v7xjlb2nks.streamlit.app)



---

##  Why This Project?

Agriculture is the backbone of India, but most farmers lack access to timely data to guide their sowing decisions. **Krishi-Jñāna** is an initiative to bridge that gap by offering a free, intelligent decision-support tool to make agriculture more data-driven and profitable.

---

##  Author

**Shravan Kumar Gogi**

---


