import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Função Black-Scholes com cálculo dos Greeks
def black_scholes_greeks(S, K, T, r, sigma, tipo_opcao='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if tipo_opcao == 'call':
        preco = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        preco = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    # Greeks comuns a calls e puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(d2 if tipo_opcao == 'call' else -d2))
    
    return preco, delta, gamma, theta, vega, rho

# Configuração da interface
st.set_page_config(layout="wide")
st.title("📊 Compreendendo os **Greeks** no Modelo Black-Scholes")
st.markdown("Explore como os parâmetros afetam os **Greeks** (Delta, Gamma, Theta, Vega, Rho) e o preço das opções.")

# Barra lateral com parâmetros
with st.sidebar:
    st.header("⚙️ Parâmetros")
    S = st.slider("Preço Atual do Ativo (S)", 50.0, 150.0, 100.0)
    K = st.slider("Preço de Exercício (K)", 50.0, 150.0, 105.0)
    T = st.slider("Tempo até o Vencimento (anos)", 0.1, 5.0, 1.0)
    r = st.slider("Taxa de Juro Sem Risco (r)", 0.0, 0.2, 0.05)
    sigma = st.slider("Volatilidade (σ)", 0.1, 1.0, 0.2)
    tipo_opcao = st.radio("Tipo de Opção", ["call", "put"])

# Calcular preço e Greeks
preco, delta, gamma, theta, vega, rho = black_scholes_greeks(S, K, T, r, sigma, tipo_opcao)

# Mostrar resultados em colunas
col1, col2 = st.columns([1, 3])
with col1:
    st.success(f"### Preço da Opção: **€{preco:.2f}**")
    
    # Tabela de Greeks
    st.markdown("### Sensibilidades (Greeks)")
    st.markdown(f"""
    - **Delta (Δ):** `{delta:.3f}`  
      *Mudança no preço da opção por €1 no ativo.*
    - **Gamma (Γ):** `{gamma:.3f}`  
      *Mudança no Delta por €1 no ativo.*
    - **Theta (Θ):** `{theta:.3f}/dia`  
      *Erosão diária do valor devido ao tempo.*
    - **Vega (ν):** `{vega:.3f}`  
      *Impacto de um aumento de 1% na volatilidade.*
    - **Rho (ρ):** `{rho:.3f}`  
      *Impacto de um aumento de 1% nas taxas de juro.*
    """)

with col2:
    # Selecionar Greek para visualizar
    grego_selecionado = st.selectbox(
        "Selecione um Greek para visualizar:",
        ["Delta", "Gamma", "Theta", "Vega", "Rho"],
        index=0
    )
    
    # Gerar gráfico do Greek selecionado
    fig, ax = plt.subplots(figsize=(10, 5))
    S_range = np.linspace(50, 150, 100)
    
    # Calcular valores do Greek para cada S no intervalo
    valores_grego = []
    for s in S_range:
        _, d, g, t, v, r_val = black_scholes_greeks(s, K, T, r, sigma, tipo_opcao)
        if grego_selecionado == "Delta":
            valores_grego.append(d)
        elif grego_selecionado == "Gamma":
            valores_grego.append(g)
        elif grego_selecionado == "Theta":
            valores_grego.append(t / 365)  # Theta diário
        elif grego_selecionado == "Vega":
            valores_grego.append(v)
        else:
            valores_grego.append(r_val)
    
    ax.plot(S_range, valores_grego, color='darkorange', linewidth=2)
    ax.axvline(S, color='red', linestyle='--', label='Preço Atual (S)')
    ax.set_title(f"{grego_selecionado} vs Preço do Ativo", fontweight='bold')
    ax.set_xlabel("Preço do Ativo (S)")
    ax.set_ylabel(f"{grego_selecionado}")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Explicação dinâmica sobre o impacto dos parâmetros
    explicacoes = {
        "Delta": "Delta mede a sensibilidade do preço da opção a mudanças no preço do ativo subjacente. Um Delta maior indica uma ligação mais forte ao preço do ativo.",
        "Gamma": "Gamma reflete a taxa de variação do Delta. Valores altos de Gamma indicam maior sensibilidade do Delta a mudanças no preço do ativo.",
        "Theta": "Theta mede a perda de valor da opção devido à passagem do tempo. Quanto mais próximo do vencimento, maior o impacto do Theta.",
        "Vega": "Vega mede a sensibilidade do preço da opção à volatilidade do ativo. Quanto maior o Vega, maior o impacto das mudanças na volatilidade.",
        "Rho": "Rho mede a sensibilidade do preço da opção às mudanças nas taxas de juros. Afeta mais as opções de longo prazo."
    }
    st.info(f"📘 **{grego_selecionado}:** {explicacoes[grego_selecionado]}")
