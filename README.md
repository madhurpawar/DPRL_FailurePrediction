# ⚙️ System Reliability Optimization using Dynamic Programming & Reinforcement Learning  

## 🚀 Project Overview  
This project models a **deteriorating system's failure probability** using **Dynamic Programming (DP) and Reinforcement Learning (RL)** to optimize **replacement and preventive maintenance decisions**. By leveraging **stationary distribution analysis, Monte Carlo simulation, Poisson's equation, and Bellman’s optimality equation**, we compute the **long-run average cost** and determine the **best maintenance policy**.

---

## 🔍 Key Features  
✔️ **Failure Probability Tracking** – Models system deterioration over **91 states**  
✔️ **Stationary Distribution Calculation** – Determines steady-state probabilities  
✔️ **Monte Carlo Simulation** – Validates failure patterns over **100,000 iterations**  
✔️ **Poisson Equation Solution** – Computes long-run expected costs  
✔️ **Policy & Value Iteration** – Finds the **optimal preventive maintenance strategy**  

---

## 🏗 Methods Used  
- **Markov Chains** for stationary distribution analysis  
- **Monte Carlo Simulation** for long-run system behavior verification  
- **Poisson’s Equation** to calculate expected limiting cost  
- **Bellman’s Equation** to optimize preventive replacement decisions  
- **Policy Iteration & Value Iteration** for strategic cost reduction  

---

## 📊 Results & Insights  
- **Long-run average cost (stationary distribution):** `0.1461`  
- **Monte Carlo simulation verified cost:** `0.1439`  
- **Poisson equation solution:** `0.1461`  
- **Optimal maintenance policy reduces cost to:** `0.1449`  

These results consistently demonstrate the effectiveness of the model in reducing maintenance costs.

---

## 📷 Visualizations  
🔹 **Failure Probability Evolution** – Visualizes system degradation over time  
🔹 **Monte Carlo Distribution** – Compares theoretical vs simulated stationary behavior  
🔹 **Optimal Replacement Policy** – Highlights preventive vs reactive strategies  