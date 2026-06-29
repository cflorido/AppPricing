from flask import Flask, render_template
import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import adfuller
import yfinance as yfin
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import seaborn as sns
import mplfinance as mpf
import math
from scipy.stats import norm, chi2, jarque_bera, shapiro
from scipy.optimize import brentq
from scipy import stats
from flask import Flask, render_template, request, jsonify, session, redirect
from flask import url_for as flask_url_for
from io import BytesIO
import base64


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "change-me-to-a-secure-random-key"

# ---------------------------------------------------------------------------
# i18n — Spanish is the source language; English translations listed below.
# ---------------------------------------------------------------------------
EN_TRANSLATIONS = {
    # ── App shell ──────────────────────────────────────────────────────────
    'Mi Aplicación': 'My Application',
    'Derivados Climáticos · Colombia': 'Climate Derivatives · Colombia',

    # ── Hero ───────────────────────────────────────────────────────────────
    'Derivados climáticos · Regiones cafeteras': 'Climate Derivatives · Coffee Regions',
    'Valuación de opciones': 'Temperature Option Pricing',
    'de temperatura': '',
    'Modelo matemático para calcular el precio de derivados climáticos en los cinco principales departamentos productores de café de Colombia, usando series de Fourier y reversión a la media.':
        "Mathematical model for pricing climate derivatives across Colombia's five main "
        "coffee-producing departments, using Fourier series and mean-reversion stochastic processes.",
    'Regiones analizadas': 'Regions analyzed',
    'años de datos históricos': 'years of historical data',

    # ── Hero subtitle ──────────────────────────────────────────────────────
    'Valuación de opciones call/put sobre índices acumulados de grados-día (AccHDD, AccCDD) en cinco departamentos cafeteros de Colombia. La temperatura subyacente se modela como T(t) = S(t) + X(t), donde S(t) es una componente estacional determinista via serie de Fourier de tercer armónico y X(t) un proceso de Ornstein-Uhlenbeck con parámetros η, μ y σ estimados por MCO sobre primeras diferencias.':
        'Call/put valuation on accumulated degree-day indices (AccHDD, AccCDD) across five Colombian '
        'coffee departments. The underlying temperature is modelled as T(t) = S(t) + X(t), where S(t) '
        'is a deterministic seasonal component via third-order Fourier series and X(t) an '
        'Ornstein-Uhlenbeck process with parameters η, μ and σ estimated by OLS on first differences.',

    # ── Context strip ──────────────────────────────────────────────────────
    'Índices HDD y CDD como subyacente': 'HDD and CDD Indices as Underlyings',
    'Para cada día t se definen HDD(t) = max(0, c − T(t)) y CDD(t) = max(0, T(t) − c), donde c = E[T] es la temperatura media histórica de la región. Los índices acumulados AccHDD(τ₁,τ₂) = Σ HDD(t) y AccCDD(τ₁,τ₂) = Σ CDD(t) sobre el período contractual constituyen el subyacente no transable de las opciones, lo que impide replicación dinámica y exige valuación bajo medida martingala equivalente.':
        'For each day t: HDD(t) = max(0, c − T(t)) and CDD(t) = max(0, T(t) − c), where c = E[T] is '
        'the regional historical mean. The accumulated indices AccHDD(τ₁,τ₂) = Σ HDD(t) and '
        'AccCDD(τ₁,τ₂) = Σ CDD(t) over the contract period are non-tradable underlyings, precluding '
        'dynamic replication and requiring valuation under an equivalent martingale measure.',
    'Proceso de Ornstein-Uhlenbeck + Fourier': 'Ornstein-Uhlenbeck Process + Fourier Series',
    'La componente estacional S(t) se ajusta via serie de Fourier de orden 3: S(t) = a₀/2 + Σₙ[aₙcos(2πnt/366) + bₙsen(2πnt/366)], con 7 constantes estimadas por curve_fit. La perturbación estocástica X(t) sigue dX = η(μ − X)dt + σdW, discretizada como X(t+1) = μ(1−e^(−η)) + e^(−η)X(t) + ε, donde η = −ln(1+b) y μ = −a/b se obtienen de la regresión MCO de ΔT sobre T(t−1).':
        'The seasonal component S(t) is fitted via order-3 Fourier series: S(t) = a₀/2 + Σₙ[aₙcos(2πnt/366) + bₙsin(2πnt/366)], '
        'with 7 constants estimated by curve_fit. The stochastic perturbation X(t) follows '
        'dX = η(μ − X)dt + σdW, discretised as X(t+1) = μ(1−e^(−η)) + e^(−η)X(t) + ε, '
        'where η = −ln(1+b) and μ = −a/b are derived from OLS regression of ΔT on T(t−1).',
    'Fórmula de valuación libre de arbitraje': 'Arbitrage-Free Pricing Formula',
    'Bajo medida Q riesgo-neutral los precios de las opciones son: C_HDD = μ·e^(−rτ)·max(0, AccHDD − K) y P_HDD = μ·e^(−rτ)·max(0, K − AccHDD), donde μ = USD 25 es el precio por punto de índice (CME), r = 5.2149% la tasa libre de riesgo, K el strike en grados-día y τ = (τ₂−τ₁)/365 el plazo en años. Los intervalos de confianza al 95% se construyen como X̄ ± z_(α/2)·σ/√n sobre las n trayectorias simuladas del proceso OU.':
        'Under risk-neutral measure Q the option prices are: C_HDD = μ·e^(−rτ)·max(0, AccHDD − K) and '
        'P_HDD = μ·e^(−rτ)·max(0, K − AccHDD), where μ = USD 25 is the index point price (CME), '
        'r = 5.2149% the risk-free rate, K the strike in degree-days and τ = (τ₂−τ₁)/365 the term '
        'in years. 95% confidence intervals are built as X̄ ± z_(α/2)·σ/√n over the n simulated OU paths.',

    # ── Methodology section ────────────────────────────────────────────────
    'Fundamento matemático': 'Mathematical Foundation',
    'Arquitectura del modelo': 'Model Architecture',
    'Temperatura promedio diaria': 'Daily Average Temperature',
    'La serie de entrada se construye como el promedio aritmético de la temperatura máxima y mínima registradas por IDEAM en cada estación aeroportuaria:':
        'The input series is constructed as the arithmetic mean of the maximum and minimum temperatures recorded by IDEAM at each airport station:',
    'Las observaciones faltantes se imputan mediante interpolación por vecino más cercano sobre la serie re-muestreada a frecuencia diaria.':
        'Missing observations are imputed via nearest-neighbour interpolation on the series resampled to daily frequency.',
    'Componente determinista — Serie de Fourier': 'Deterministic Component — Fourier Series',
    'La estacionalidad anual se captura con una serie de Fourier de orden N = 3, ajustada por mínimos cuadrados no lineales (scipy.curve_fit) sobre los 8 766 datos históricos:':
        'Annual seasonality is captured with an order N = 3 Fourier series, fitted by nonlinear least squares (scipy.curve_fit) over 8 766 historical observations:',
    'El ajuste entrega 7 constantes {a₀, a₁, b₁, a₂, b₂, a₃, b₃} por región, capturando el ciclo intra-anual de temperatura con un RMSE típico inferior a 1.2 °C.':
        'The fit yields 7 constants {a₀, a₁, b₁, a₂, b₂, a₃, b₃} per region, capturing the intra-annual temperature cycle with a typical RMSE below 1.2 °C.',
    'Componente estocástica — Proceso de Ornstein-Uhlenbeck': 'Stochastic Component — Ornstein-Uhlenbeck Process',
    'El residuo X(t) = T(t) − S(t) sigue un proceso de reversión a la media en tiempo continuo:':
        'The residual X(t) = T(t) − S(t) follows a continuous-time mean-reversion process:',
    'Su discretización de Euler-Maruyama con paso dt = 1/365 da el esquema de simulación:':
        'Its Euler-Maruyama discretisation with step dt = 1/365 gives the simulation scheme:',
    'Los parámetros se estiman regresando ΔT = a + bT(t−1) + ε por MCO, de donde η = −ln(1+b) y μ = −a/b. Se generan n = 5 trayectorias y se toma la media como proyección estocástica.':
        'Parameters are estimated by regressing ΔT = a + bT(t−1) + ε via OLS, yielding η = −ln(1+b) and μ = −a/b. n = 5 trajectories are generated and the mean is taken as the stochastic projection.',
    'Índices de grados-día acumulados': 'Accumulated Degree-Day Indices',
    'La proyección final T̂(t) = [S(t) + X̄(t)] / 2 alimenta el cálculo diario de grados-día, usando como umbral c = E[T̂]:':
        'The final projection T̂(t) = [S(t) + X̄(t)] / 2 feeds the daily degree-day calculation, using threshold c = E[T̂]:',
    'Los acumulados sobre el período contractual [τ₁, τ₂] constituyen el subyacente no transable de las opciones:':
        'The accumulations over the contract period [τ₁, τ₂] constitute the non-tradable option underlyings:',
    'Al ser AccHDD/AccCDD subyacentes no transables, no existe portafolio replicante. El precio se obtiene como esperanza bajo la medida martingala equivalente Q, descontada a la tasa libre de riesgo r:':
        'Since AccHDD/AccCDD are non-tradable underlyings, no replicating portfolio exists. The price is obtained as an expectation under the equivalent martingale measure Q, discounted at the risk-free rate r:',
    'Donde μ = USD 25/punto (CME Group), r = 5.2149%, K es el strike en grados-día y τ = (τ₂−τ₁)/365. Los intervalos de confianza al 95% se derivan como X̄ ± z₀.₀₂₅ · σ/√n sobre las trayectorias OU simuladas.':
        'Where μ = USD 25/point (CME Group), r = 5.2149%, K is the strike in degree-days and τ = (τ₂−τ₁)/365. 95% confidence intervals are derived as X̄ ± z₀.₀₂₅ · σ/√n over the simulated OU trajectories.',
    'Parámetros del modelo': 'Model Parameters',
    'Valores fijos utilizados en la implementación:': 'Fixed values used in the implementation:',
    'Precio por punto de índice': 'Index point price',
    'Tasa libre de riesgo': 'Risk-free rate',
    'Orden de la serie de Fourier': 'Fourier series order',
    'Trayectorias OU simuladas': 'Simulated OU trajectories',
    'Paso de discretización': 'Discretisation step',
    'Nivel de confianza': 'Confidence level',
    'Umbral de grados-día': 'Degree-day threshold',

    # ── Tool panel ─────────────────────────────────────────────────────────
    'Parámetros del modelo': 'Model Parameters',
    'Departamento': 'Department',
    'Mes de análisis': 'Analysis Month',
    'Calcular intervalos': 'Compute Intervals',
    'Strike Price': 'Strike Price',
    'Graficar opciones': 'Plot Options',
    'Precios de opciones': 'Option Prices',
    'Intervalos de confianza · IC 95%': 'Confidence Intervals · CI 95%',

    # ── Months ─────────────────────────────────────────────────────────────
    'Enero': 'January',
    'Febrero': 'February',
    'Marzo': 'March',
    'Abril': 'April',
    'Mayo': 'May',
    'Junio': 'June',
    'Julio': 'July',
    'Agosto': 'August',
    'Septiembre': 'September',
    'Octubre': 'October',
    'Noviembre': 'November',
    'Diciembre': 'December',

    # ── Footer / metadata ──────────────────────────────────────────────────
    'Datos históricos: IDEAM · Precio índice: CME Group (≈ USD 25) · Tasa libre de riesgo: 5.21% · Período: 2000–2024':
        'Historical data: IDEAM · Index price: CME Group (≈ USD 25) · Risk-free rate: 5.21% · Period: 2000–2024',

    # ── Matplotlib chart labels ────────────────────────────────────────────
    'Pricing Derivados Climáticos': 'Climate Derivatives Pricing',
    'Temperatura Promedio por Región': 'Average Temperature by Region',
    'Simulacion de Temperatura para {var}': 'Temperature Simulation for {var}',
    'Proyección de Temperaturas para el Próximo Año (366 días)': 'Temperature Projection for the Next Year (366 days)',
    'Proyeccion Final (Fourier + Mean Reversion)': 'Final Projection (Fourier + Mean Reversion)',
    'Proyeccion Fourier': 'Fourier Projection',
    'Proyeccion Mean Reversioin': 'Mean-Reversion Projection',
    'Ajuste de Temperaturas en {region} usando Transformada de Fourier y Modelos Estocásticos':
        'Temperature Fit in {region} using Fourier Transform and Stochastic Models',
    'Heating Degree Days (HDD) y Cooling Degree Days (CDD) en {region}':
        'Heating Degree Days (HDD) and Cooling Degree Days (CDD) in {region}',
    'Fecha': 'Date',
    'Grados-Día': 'Degree-Days',
    'Precio de Opciones Climáticas en {ciudad}': 'Climate Option Prices in {ciudad}',
    'Mes': 'Month',
    'Precio': 'Price',

    # ── Result labels ──────────────────────────────────────────────────────
    'Call Option': 'Call Option',
    'Put option': 'Put Option',
    'Call HDD: ': 'Call HDD: ',
    'Put HDD: ': 'Put HDD: ',
    'Call CDD: ': 'Call CDD: ',
    'Put CDD: ': 'Put CDD: ',
}


def get_locale():
    return session.get('lang', 'es')


def _translate(text):
    lang = get_locale()
    if lang == 'en':
        return EN_TRANSLATIONS.get(text, text)
    return text


# Expose translation function inside Jinja templates as `_`
app.jinja_env.globals.update(_=_translate)


@app.context_processor
def inject_language():
    return dict(current_language=get_locale())


# Convenience alias for use in Python code
_ = _translate


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/set_language/<lang>')
def set_language(lang):
    if lang in ('es', 'en'):
        session['lang'] = lang
    return redirect(flask_url_for('index'))


@app.route('/calcular', methods=['POST'])
def calcular():
    data = request.json
    ciudad = data['ciudad']
    valor_numerico = float(data['valorNumerico'])

    fig = app_function(valor_numerico, ciudad)

    folder_path = app.static_folder
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, 'grafica.png')
    fig.savefig(file_path)

    imagen_url = flask_url_for('static', filename='grafica.png')
    return jsonify({'graficaUrl': imagen_url})


@app.route('/calcular_intervalos', methods=['POST'])
def calcular_intervalos():
    data = request.json
    ciudad = data['ciudad']
    mes = int(data['mes'])

    Call_Hdd_str, Put_HDD_str, Call_CDD_str, Put_CDD_str = intervalos(ciudad, mes)

    return jsonify({
        'Call_HDD': Call_Hdd_str,
        'Put_HDD': Put_HDD_str,
        'Call_CDD': Call_CDD_str,
        'Put_CDD': Put_CDD_str,
    })


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def app_function(strike, ciudad):
    filename = os.path.join(DATA_DIR, 'datos.csv')
    df = pd.read_csv(filename)

    regions = ['Anti', 'Narino', 'NortSan', 'Tolima', 'Cauca']

    for region in regions:
        max_col  = f"Max_{region}"
        min_col  = f"Min_{region}"
        tprom_col = f"TPROM_{region}"
        df[tprom_col] = (df[max_col] + df[min_col]) / 2

    def fill_missing_data(df, regions):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.set_index('Fecha')
        df = df.resample('D').mean()
        for region in regions:
            tprom_col = f"TPROM_{region}"
            if tprom_col in df.columns:
                df[tprom_col].interpolate(method='nearest', inplace=True)
        df = df.reset_index()
        return df

    df = fill_missing_data(df, regions)

    plt.figure(figsize=(10, 5))
    for column in [f'TPROM_{r}' for r in regions]:
        plt.plot(df.index, df[column], label=column)
    plt.legend()
    plt.title(_('Temperatura Promedio por Región'))
    plt.xlabel(_('Fecha'))
    plt.ylabel('Temperatura (°C)')
    plt.xlim(df.index.min(), pd.Timestamp('2006-7-13'))

    # ── Mean-reversion (Ornstein-Uhlenbeck) paths ──────────────────────────
    temp             = [f'TPROM_{r}' for r in regions]
    df_Temp_f_columns = [f'TPROM_{r}MR' for r in regions]
    df_Temp_f = pd.DataFrame(index=np.arange(366), columns=df_Temp_f_columns)

    def simulate_path(m, sigma, eta, dt, T, n, S0, sigma_e):
        timesteps = int(T / dt)
        paths = np.zeros((n, timesteps + 1))
        paths[:, 0] = S0
        for i in range(n):
            for j in range(1, timesteps + 1):
                paths[i, j] = (m * (1 - np.exp(-eta))
                               + np.exp(-eta) * paths[i, j - 1]
                               + np.random.normal(loc=0, scale=sigma_e))
        return paths

    for idx, col in enumerate(temp):
        y = df[col] - df[col].shift(1)
        X = sm.add_constant(df[col].shift(1))
        model  = sm.OLS(y, X, missing='drop')
        result = model.fit()

        a       = result.params[0]
        b       = result.params[1]
        sigma_e = np.std(result.resid)
        sigma   = np.std(df[col])
        m       = -a / b
        eta     = -np.log(1 + b)

        N  = 365
        T  = 1
        dt = T / N
        n  = 5
        S0 = df[col].values[-1]

        paths = simulate_path(m, sigma, eta, dt, T, n, S0, sigma_e)

        timesteps = int(T / dt)
        t = np.linspace(0, T, timesteps + 1)
        plt.figure(figsize=(10, 6))
        for j in range(n):
            plt.plot(t, paths[j])
        plt.xlabel('Time (years)')
        plt.ylabel('T [°C]')
        plt.title(_('Simulacion de Temperatura para {var}').format(var=col))

        Tp = np.mean(paths, axis=0)
        df_Temp_f[df_Temp_f_columns[idx]] = Tp

    # ── Fourier series (deterministic seasonal component) ──────────────────
    def fourier_series(t, *a):
        t   = np.asarray(t, dtype=float)
        ret = a[0] * np.ones_like(t)
        for i in range(1, len(a) // 2 + 1):
            ret += (a[2*i-1] * np.sin(2 * np.pi * i * t / 366)
                    + a[2*i]   * np.cos(2 * np.pi * i * t / 366))
        return ret

    def deterministic_seasonality(df, regions):
        results = {}
        for region in regions:
            tprom_col = f"TPROM_{region}"
            if tprom_col in df.columns:
                df['day_of_year'] = df.index.dayofyear
                temp_df = df.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=[tprom_col, 'day_of_year'])
                popt, _ = curve_fit(fourier_series, temp_df['day_of_year'],
                                    temp_df[tprom_col], [1] * 7)
                df[f"seasonal_fit_{region}"] = fourier_series(df['day_of_year'], *popt)
                results[region] = popt
        df.drop(columns=['day_of_year'], inplace=True)
        return df, results

    def project_temperatures(df, regions, results):
        future_days = np.arange(1, 367)
        df_future   = pd.DataFrame(index=future_days)
        for region in regions:
            df_future[f"projection_{region}"] = fourier_series(
                future_days, *results[region])
        return df_future

    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df.set_index('Fecha', inplace=True)

    df, results   = deterministic_seasonality(df, regions)
    df_future     = project_temperatures(df, regions, results)

    plt.figure(figsize=(14, 7))
    for region in regions:
        plt.plot(df_future.index, df_future[f"projection_{region}"],
                 label=f'Proyección {region}')
    plt.xlabel('Día del Año')
    plt.ylabel('Temperatura [°C]')
    plt.title(_('Proyección de Temperaturas para el Próximo Año (366 días)'))
    plt.legend()

    # ── Combined projection (Fourier + OU mean) ────────────────────────────
    df_TF = pd.DataFrame(index=np.arange(366), columns=regions)
    for region in regions:
        df_TF[region] = (df_Temp_f[f"TPROM_{region}MR"]
                         + df_future[f"projection_{region}"]) / 2

    for region in regions:
        plt.figure(figsize=(10, 5))
        plt.plot(df_TF.index,     df_TF[region],
                 label=_('Proyeccion Final (Fourier + Mean Reversion)'))
        plt.plot(df_future.index, df_future[f"projection_{region}"],
                 label=_('Proyeccion Fourier'))
        plt.plot(df_Temp_f.index, df_Temp_f[f"TPROM_{region}MR"],
                 label=_('Proyeccion Mean Reversioin'))
        plt.legend()
        plt.title(_('Ajuste de Temperaturas en {region} usando Transformada de Fourier y Modelos Estocásticos').format(region=region))
        plt.xlabel('Fecha')
        plt.ylabel('Temperatura (°C)')

    # ── HDD / CDD calculation ──────────────────────────────────────────────
    HDD_values = {r: [] for r in regions}
    CDD_values = {r: [] for r in regions}

    for region in regions:
        c = np.mean(df_TF[region])
        for _, row in df_TF.iterrows():
            HDD_values[region].append(np.maximum(c - row[region], 0))
            CDD_values[region].append(np.maximum(row[region] - c, 0))

    for region in regions:
        df_TF[f'HDD_{region}'] = HDD_values[region]
        df_TF[f'CDD_{region}'] = CDD_values[region]

    for region in regions:
        plt.figure(figsize=(10, 5))
        plt.plot(df_TF.index, df_TF[region],
                 label=f'HDD ({region})', marker='o', linestyle='-')
        plt.title(_('Heating Degree Days (HDD) y Cooling Degree Days (CDD) en {region}').format(region=region))
        plt.xlabel(_('Fecha'))
        plt.ylabel(_('Grados-Día'))
        plt.legend()
        plt.grid(True)

    date_range    = pd.date_range(start='2024-01-01', end='2024-12-31')
    df_TF.index   = date_range

    # ── Option pricing ─────────────────────────────────────────────────────
    def option_pricing(df, strike_price, r, g, start_date, end_date):
        option_prices = {}
        time_period   = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365
        df_period     = df[start_date:end_date]

        for region in regions:
            hdd_sum  = df_period[f'HDD_{region}'].sum()
            cdd_sum  = df_period[f'CDD_{region}'].sum()
            discount = g * np.exp(-r * time_period)

            option_prices[region] = {
                'call_hdd': discount * max(0, hdd_sum - strike_price),
                'put_hdd':  discount * max(0, strike_price - hdd_sum),
                'call_cdd': discount * max(0, cdd_sum - strike_price),
                'put_cdd':  discount * max(0, strike_price - cdd_sum),
            }
        return option_prices

    strike_price     = strike
    risk_free_rate   = 0.0521487
    index_point_price = 25

    regions = [ciudad]   # scope to selected region for the output chart

    year    = 2024
    results = []
    dates   = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M')

    for date in dates:
        start_date = date.strftime('%Y-%m-01')
        end_date   = (date + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        option_prices = option_pricing(df_TF, strike_price, risk_free_rate,
                                       index_point_price, start_date, end_date)
        for region in regions:
            results.append({
                'Year':     year,
                'Month':    date.strftime('%B'),
                'Region':   region,
                'Call HDD': option_prices[region]['call_hdd'],
                'Put HDD':  option_prices[region]['put_hdd'],
                'Call CDD': option_prices[region]['call_cdd'],
                'Put CDD':  option_prices[region]['put_cdd'],
            })

    df_pricing        = pd.DataFrame(results)
    df_pricing_region = df_pricing[df_pricing['Region'] == ciudad]

    df_pricing_region.plot(x='Month',
                           y=['Call HDD', 'Put HDD', 'Call CDD', 'Put CDD'],
                           kind='line')
    plt.title(_('Precio de Opciones Climáticas en {ciudad}').format(ciudad=ciudad))
    plt.xlabel(_('Mes'))
    plt.ylabel(_('Precio'))

    fig = plt.gcf()
    plt.close('all')
    return fig


def intervalos(lugar, mes):
    filename = os.path.join(DATA_DIR, 'intervalos.csv')
    df       = pd.read_csv(filename)

    def _get(variable):
        mask   = ((df['Region'] == lugar)
                  & (df['Month'] == mes)
                  & (df['Variable'] == variable))
        values = df.loc[mask, 'Confidence Interval'].values
        return str(values[0]) if len(values) > 0 else 'No disponible'

    return _get('Call_HDD'), _get('Put_HDD'), _get('Call_CDD'), _get('Put_CDD')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)