﻿# Qure - Cancer Detection App with Classical and Quantum ML

## Team Members
- Alaa Bou Nassif
- Bruno Fares
- Rita Hochar
- Anthony Nasry Massaad
- Nizar Shehayeb  

*April 2025*

---

## Table of Contents
1. [Introduction](#1-introduction)  
   1.1 [Overview](#11-overview)  
   1.2 [Objectives](#12-objectives)  
2. [Scientific Background](#2-scientific-background)  
   2.1 [Bioelectrical Impedance Analysis (BIA)](#21-bioelectrical-impedance-analysis-bia)  
   2.2 [BIA in Cancer Detection](#22-bia-in-cancer-detection)  
   2.3 [Measuring and Interpreting Impedance Data](#23-measuring-and-interpreting-impedance-data)  
   2.4 [Application of BIA in Cancer Diagnosis](#24-application-of-bia-in-cancer-diagnosis)  
3. [Algorithms for Diagnosis](#3-algorithms-for-diagnosis)  
   3.1 [Dataset Selection](#31-dataset-selection)  
   3.2 [Classical Machine Learning](#32-classical-machine-learning)  
   3.3 [Quantum Machine Learning](#33-quantum-machine-learning)  
   3.4 [Comparison: Classical ML vs. Quantum ML](#34-comparison-classical-ml-vs-quantum-ml)  
4. [Conclusion](#4-conclusion)  
   4.1 [Marketing Perspective](#41-marketing-perspective)  
   4.2 [Future Directions](#42-future-directions)  
   4.3 [Project Summary](#43-project-summary)  

---

## 1. Introduction

### 1.1 Overview
This project aims to develop a non-invasive system for early cancer detection by analyzing the body's electrical properties using **Quantum Machine Learning (QML)**. The system processes bio-electrical impedance data and compares it to healthy profiles, providing a smart, accessible health monitoring tool aligned with the UN's **Sustainable Development Goal 3: Good Health and Well-Being**.

### 1.2 Objectives
- Use **bioelectrical impedance** to measure resistance and detect abnormal tissue properties.  
- Provide an **early warning system** that recommends medical follow-up if needed.  
- Offer a **low-cost, non-invasive solution** to improve early diagnosis, especially in underserved communities.  

---

## 2. Scientific Background

### 2.1 Bioelectrical Impedance Analysis (BIA)
BIA measures the **resistance (R)** and **reactance (Xc)** of the body to a small electrical current, reflecting body composition (water, fat, muscle, etc.). Different tissues resist electrical currents differently (e.g., fat = poor conductor, muscle = good conductor). By measuring resistance and reactance, body composition can be inferred.  

### 2.2 BIA in Cancer Detection
Cancerous tissues exhibit altered properties:  
- Higher hydration levels.  
- Different cellular structure.  
- Lower resistance compared to healthy tissues.  

Tumors show **distinct impedance values** due to their conductivity, making BIA a potential non-invasive detection tool.  

### 2.3 Measuring and Interpreting Impedance Data
Total impedance (**Z**) is calculated as:  
Z = √(R² + X_c²)
Where:
- **R** = Resistance  
- **X<sub>c</sub>** = Reactance  

Cancerous growth alters impedance profiles, detectable via specialized electrodes.  

### 2.4 Application of BIA in Cancer Diagnosis  
BIA offers **quick, affordable, and non-invasive** tissue assessment, enabling proactive monitoring and early diagnosis.  

---

## 3. Algorithms for Diagnosis

### 3.1 Dataset Selection  
- Dataset: **Electric bioimpedance sensing for head and neck squamous cell carcinoma**.  
- **2,015 entries** from 43 patients.  
- Each entry includes:  
  - 10 reactance measurements (10–100 kHz).  
  - 10 phase angle measurements (10–100 kHz).  

### 3.2 Classical Machine Learning  
- Model: **Random Forest Regressor** (100 decision trees).  
- Metrics:  
  - Accuracy: R² score.
  - Loss: Root Mean Squared Error (RMSE).  

### 3.3 Quantum Machine Learning  
1. **Preprocessing**: Standardization + PCA (reduced to 4 dimensions).  
2. **Feature Map**: Parameters transformed into qubits (rotated via Z-gate-like operations).  
3. **Ansatz**: Parametrized quantum circuit (rotations + CNOT gates for entanglement).  
4. **Optimizer**: COBYLA (20 iterations).  
5. **Classifier**: Variational Quantum Classifier (VQC) outputs cancer probability.  

### 3.4 Comparison: Classical ML vs. Quantum ML  

| **Aspect**               | **Classical ML**                          | **Quantum ML**                          |
|--------------------------|------------------------------------------|----------------------------------------|
| **Maturity**             | Reliable, well-established               | Emerging, experimental                 |
| **Hardware**             | Classical computers                      | Quantum processors/simulators          |
| **Speed**                | Fast training/inference                  | Slower (complex simulations)           |
| **Data Handling**         | Effective for tabular data               | Potential for hidden correlations      |
| **Deployment**           | Easy integration                         | Challenging (early-stage tech)         |

---

## 4. Conclusion

### 4.1 Marketing Perspective  
**Qure** empowers users with **early, affordable cancer detection** at home, aligning with **SDG 3**. It doesn’t replace doctors but provides proactive health monitoring.  

### 4.2 Future Directions  
- Collaborate with hospitals for **real-world data**.  
- Expand datasets for **improved accuracy**.  
- Integrate with **fitness/health apps** for seamless monitoring.  

### 4.3 Project Summary  
Qure combines **BIA and QML** to revolutionize early cancer detection, offering a **non-invasive, scalable solution** for global health equity.  

---
