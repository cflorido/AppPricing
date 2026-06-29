# Climate Derivatives Pricing – Valuation Model for Temperature Options

This project aimed to evaluate the feasibility of temperature-based climate derivatives as a hedging mechanism for coffee producers in Colombia. The focus was on valuing Heating Degree Days (HDD) and Cooling Degree Days (CDD) options across regions such as Antioquia, Nariño, Norte de Santander, Cauca, and Tolima. The work followed a complete data science and quantitative finance pipeline, from climate data collection and preprocessing to model development and the creation of an interactive pricing tool.

The preprocessing stage included handling missing values in temperature time series using mean imputation and nearest-neighbor interpolation. Daily maximum and minimum temperatures were transformed into average temperature series, ensuring consistency and continuity for modeling. HDD and CDD indices were computed from historical data to quantify deviations from reference thresholds, forming the basis for derivative pricing.

Exploratory analysis revealed clear seasonal patterns in temperature behavior across coffee regions, with both extreme heat and cold posing significant risks to agricultural productivity. Deterministic models, specifically Fourier series with third-degree harmonics, were applied to capture seasonal cycles, while stochastic models, particularly mean reversion, were integrated to account for random fluctuations and long-term variability. This hybrid methodology ensured both robustness and realism in climate projections.

Option pricing was carried out using HDD and CDD indices under risk-neutral valuation frameworks. Confidence intervals for call and put option prices were derived using historical variability, allowing stakeholders to assess uncertainty in premiums. Results demonstrated that temperature options provide viable coverage against climate risk, mitigating both frost damage and heat stress on crops. By incorporating stochastic and deterministic elements, the framework offers more reliable pricing of weather derivatives.

The final solution was complemented by the development of an interactive tool that enables business users and stakeholders to:

- Explore projected temperature scenarios across multiple regions.
- Visualize pricing dynamics for HDD and CDD call/put options over time.
- Compare premiums and confidence intervals under different climate conditions.
- Experiment with model parameters to assess hedging strategies.

This system provides agricultural and financial stakeholders with a structured approach to climate risk management, combining quantitative rigor with intuitive visualization.

---

## Documents
- [Project Report – English](https://github.com/user-attachments/files/22416294/G4_Informe_Proyecto_Consultoria.pdf)

## Presentations
- [Presentación – Español  ](https://github.com/user-attachments/files/22416299/Blue.Modern.Corporate.Presentation.pdf)

## Authors
- Janeth Alexandra Riveros Baquero  
- Alejandra Garzón Carvajal  
- Carol Sofía Florido Castro – 202111430  
- Santiago Flórez  

Course: Financial Engineering – Universidad de los Andes  
City: Bogotá, Colombia  
Year: 2024  

---

## Repository Structure
app/  
data/  
docs/  
scripts/  
run.py  
requirements.txt  

---

## Technologies Used
- Python 3.11  
- NumPy, Pandas, Statsmodels, SciPy  
- Matplotlib, Seaborn  
- Jupyter Notebook  
- Flask (Web App Framework)  

---

## How to Run
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Run the application
```bash
python run.py
```
