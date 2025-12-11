import streamlit as st
import pandas as pd
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# Import data loader
from data_loader import load_texas_property_data, process_property_data, create_property_knowledge_base

# Page config
st.set_page_config(
    page_title="LoneStar AI - Property Due Diligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for LoneStar AI branding
st.markdown("""
<style>
    /* Main color scheme: White background, Black text, Teal + Orange accents */

    /* Headers */
    h1, h2, h3 {
        color: #000000;
    }

    /* Primary buttons - Teal */
    .stButton>button {
        background-color: #008B8B;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color: #006666;
        color: white;
    }

    /* Sidebar - Orange border */
    [data-testid="stSidebar"] {
        background-color: #F5F5F5;
        border-right: 3px solid #BF5700;
    }

    /* Info boxes - Teal */
    .stAlert {
        background-color: #E0F2F1;
        border-left: 4px solid #008B8B;
    }

    /* Text inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-color: #008B8B;
    }

    /* Links - Orange */
    a {
        color: #BF5700;
    }

    a:hover {
        color: #8B4000;
    }

    /* Metrics - Orange */
    [data-testid="stMetricValue"] {
        color: #BF5700;
    }

    /* Divider - Orange */
    hr {
        border-color: #BF5700;
    }

    /* Success message */
    .success-box {
        background-color: #E0F2F1;
        border-left: 4px solid #008B8B;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* Warning message */
    .warning-box {
        background-color: #FFF3CD;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* Danger message */
    .danger-box {
        background-color: #F8D7DA;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'texas_data' not in st.session_state:
    st.session_state.texas_data = None
if 'texas_data_loaded' not in st.session_state:
    st.session_state.texas_data_loaded = False
if 'current_api_key' not in st.session_state:
    st.session_state.current_api_key = None

# Load Texas market data
@st.cache_data
def load_market_data():
    """Load and process Texas commercial real estate data"""
    try:
        datasets = load_texas_property_data(use_sample_if_missing=True)
        combined_df = process_property_data(datasets)
        return combined_df, len(datasets)
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        return None, 0

# Configuration
CONFIG = {
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'vector_db_path': './chroma_db',
    'top_k': 5
}

# Sample data (included from the notebook)
DOMAIN_KNOWLEDGE = """
REAL ESTATE VALUATION PRINCIPLES:

Cap Rate Analysis:
- Cap Rate = Net Operating Income (NOI) / Property Value
- Austin multifamily market: typically 4.5% - 6.5%
- Lower cap rates = higher property values, lower returns
- Higher cap rates = lower property values, potentially higher risk

Common Red Flags:
1. Structural Issues:
   - Foundation cracks > 1/4 inch
   - Active mold growth
   - Roof leaks or major damage
   - HVAC systems older than 15 years

2. Financial Red Flags:
   - Vacancy rates above 15%
   - Delinquencies over 30 days
   - Operating expense ratios above 50%
   - Underreported maintenance costs

3. Legal Red Flags:
   - Unpermitted additions/renovations
   - Zoning violations
   - Non-standard lease terms

4. Operational Red Flags:
   - High tenant turnover
   - Month-to-month leases
   - Poor property management

Renovation Cost Estimates (Texas market, 2024):
- HVAC replacement: $5,000 - $8,000 per unit
- Roof replacement: $8,000 - $15,000
- Water heater: $1,000 - $1,500 per unit
- Foundation repair: $2,000 - $10,000
- Electrical panel upgrade: $2,500 - $5,000
"""

def format_docs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def initialize_embeddings():
    """Initialize the embedding model."""
    return HuggingFaceEmbeddings(model_name=CONFIG['embedding_model'])

def initialize_llm(api_key):
    """Initialize the Groq LLM."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=api_key
    )

def create_vectorstore(documents, embeddings):
    """Create vector store from documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG['chunk_size'],
        chunk_overlap=CONFIG['chunk_overlap']
    )

    split_docs = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CONFIG['vector_db_path']
    )

    return vectorstore

def create_qa_chain(vectorstore, llm):
    """Create the RAG QA chain."""
    prompt_template = """
You are an expert real estate analyst for LoneStar AI. Use the context below to answer the question.
Be specific and cite numbers from the documents when available.

Context:
{context}

Question: {question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": CONFIG['top_k']})

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain

# Header
st.markdown("<h1 style='color: #000000;'>LoneStar AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #BF5700;'>AI-Powered Property Due Diligence for Texas</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    # Configuration section with orange tint
    st.markdown("""
    <div style='background: linear-gradient(to bottom, #FFF5E6 0%, #FFE8CC 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h3 style='margin-top: 0; color: #000000;'>Configuration</h3>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get your free API key at https://console.groq.com"
    )

    st.markdown("---")

    # About section with orange tint
    st.markdown("""
    <div style='background: linear-gradient(to bottom, #FFF5E6 0%, #FFE8CC 100%); padding: 1rem; border-radius: 10px;'>
        <h3 style='margin-top: 0; color: #000000;'>About LoneStar AI</h3>
        <p style='color: #333333;'>LoneStar AI analyzes Texas commercial real estate properties in <strong>2 minutes</strong> using:</p>
        <ul style='color: #333333;'>
            <li>1,193 Texas property sales (2018-2025)</li>
            <li>AI-powered RAG analysis</li>
            <li>Real estate domain expertise</li>
        </ul>
        <p style='color: #333333;'><strong>95% faster, 90% cheaper</strong> than traditional due diligence.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Speed", "2 min", delta="vs 2-6 weeks")
    with col2:
        st.metric("Cost", "$5", delta="vs $2K+")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Property Analysis", "Market Intelligence", "Pricing", "Document Upload", "About"])

with tab1:
    st.markdown("### Property Analysis")

    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar to continue.")
    else:
        # Load Texas market data
        if not st.session_state.texas_data_loaded:
            with st.spinner("Loading Texas market data..."):
                st.session_state.texas_data, num_datasets = load_market_data()
                st.session_state.texas_data_loaded = True
                if st.session_state.texas_data is not None:
                    st.success(f"Loaded {len(st.session_state.texas_data)} Texas properties from {num_datasets} datasets")

        # Check if API key has changed
        if st.session_state.current_api_key != api_key:
            # Reset vectorstore and qa_chain if API key changed
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None
            st.session_state.current_api_key = api_key

        # Initialize system
        if st.session_state.vectorstore is None:
            with st.spinner("Initializing LoneStar AI system with market intelligence..."):
                try:
                    embeddings = initialize_embeddings()
                    llm = initialize_llm(api_key)

                    # Create knowledge base with Texas market data
                    documents = [
                        Document(
                            page_content=DOMAIN_KNOWLEDGE,
                            metadata={'source': 'knowledge_base', 'type': 'domain'}
                        )
                    ]

                    # Add Texas market data to knowledge base
                    if st.session_state.texas_data is not None:
                        texas_kb = create_property_knowledge_base(st.session_state.texas_data, max_properties=200)
                        documents.append(
                            Document(
                                page_content=texas_kb,
                                metadata={'source': 'texas_market_data', 'type': 'market_data', 'records': '200'}
                            )
                        )

                    st.session_state.vectorstore = create_vectorstore(documents, embeddings)
                    st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore, llm)

                    st.success("LoneStar AI initialized with Texas market intelligence!")
                except Exception as e:
                    st.error(f"Error initializing system: {str(e)}")
                    # Provide helpful message for invalid API key
                    if "invalid_api_key" in str(e).lower() or "401" in str(e):
                        st.info("💡 Please make sure you're using a valid Groq API key. Get one free at https://console.groq.com")

        # Property details input
        st.markdown("#### Property Details")
        col1, col2 = st.columns(2)

        with col1:
            property_address = st.text_input("Property Address", "1234 Oak Street, Austin, TX 78701")
            property_type = st.selectbox(
                "Property Type",
                ["Commercial Land", "Multifamily", "Retail", "Office", "Industrial", "Mixed-Use"]
            )
            asking_price = st.number_input("Asking Price ($)", min_value=0, value=425000, step=1000)

        with col2:
            property_size = st.number_input("Size (acres)", min_value=0.0, value=0.5, step=0.1)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=1985)
            noi = st.number_input("Net Operating Income (NOI) ($)", min_value=0, value=42000, step=1000)

        # Inspection report input
        st.markdown("#### Inspection Report / Property Notes")
        inspection_text = st.text_area(
            "Enter inspection findings, issues, or property notes:",
            height=200,
            placeholder="Example: HVAC system 18 years old, roof needs replacement in 2-3 years, foundation has minor cracks..."
        )

        # Analysis button
        if st.button("Analyze Property", type="primary"):
            if inspection_text and st.session_state.qa_chain:
                with st.spinner("Analyzing property..."):
                    try:
                        # Calculate cap rate
                        cap_rate = (noi / asking_price * 100) if asking_price > 0 else 0

                        # Create property context
                        property_context = f"""
Property Address: {property_address}
Property Type: {property_type}
Asking Price: ${asking_price:,}
Size: {property_size} acres
Year Built: {year_built}
Net Operating Income: ${noi:,}
Cap Rate: {cap_rate:.2f}%

Inspection Report:
{inspection_text}
"""

                        # Run queries
                        st.markdown("### Analysis Results")

                        # Critical Issues
                        st.markdown("#### Critical Issues")
                        issues_query = f"What are the critical issues found in this property? {property_context}"
                        issues_result = st.session_state.qa_chain.invoke(issues_query)
                        st.markdown(f'<div class="danger-box">{issues_result}</div>', unsafe_allow_html=True)

                        # Valuation Assessment
                        st.markdown("#### Valuation Assessment")
                        valuation_query = f"Based on the Texas market data and this property's cap rate of {cap_rate:.2f}%, is the asking price of ${asking_price:,} fair? {property_context}"
                        valuation_result = st.session_state.qa_chain.invoke(valuation_query)
                        st.markdown(f'<div class="warning-box">{valuation_result}</div>', unsafe_allow_html=True)

                        # Investment Recommendation
                        st.markdown("#### Investment Recommendation")
                        recommendation_query = f"Should I buy, pass, or negotiate on this property? What's a fair offer price? {property_context}"
                        recommendation_result = st.session_state.qa_chain.invoke(recommendation_query)
                        st.markdown(f'<div class="success-box">{recommendation_result}</div>', unsafe_allow_html=True)

                        # Display metrics
                        st.markdown("---")
                        st.markdown("#### Property Metrics")
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                        with metric_col1:
                            st.metric("Cap Rate", f"{cap_rate:.2f}%")
                        with metric_col2:
                            st.metric("Price/Acre", f"${asking_price/property_size:,.0f}" if property_size > 0 else "N/A")
                        with metric_col3:
                            st.metric("NOI", f"${noi:,}")
                        with metric_col4:
                            market_cap = 5.5  # Average Texas market cap rate
                            st.metric("vs Market", f"{cap_rate - market_cap:+.2f}%")

                        # Visual Charts
                        st.markdown("---")
                        st.markdown("#### Visual Analysis")

                        chart_col1, chart_col2 = st.columns(2)

                        with chart_col1:
                            # Chart 1: Cap Rate Comparison
                            st.markdown("##### Cap Rate Comparison")
                            cap_comparison_data = pd.DataFrame({
                                'Category': ['Your Property', 'Texas Market Low', 'Texas Market Avg', 'Texas Market High'],
                                'Cap Rate (%)': [cap_rate, 4.5, 5.5, 6.5]
                            })
                            fig_cap = px.bar(
                                cap_comparison_data,
                                x='Category',
                                y='Cap Rate (%)',
                                title="Your Property vs Texas Market Cap Rates",
                                color='Cap Rate (%)',
                                color_continuous_scale=['#008B8B', '#20B2AA', '#40E0D0'],
                                text='Cap Rate (%)'
                            )
                            fig_cap.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                            fig_cap.update_layout(showlegend=False, xaxis_tickangle=-45)
                            st.plotly_chart(fig_cap, use_container_width=True)

                        with chart_col2:
                            # Chart 2: 5-Year Financial Projection
                            st.markdown("##### 5-Year NOI Projection")
                            years = list(range(2025, 2030))
                            projected_noi = [noi * (1.03 ** i) for i in range(5)]  # 3% annual growth
                            projection_data = pd.DataFrame({
                                'Year': years,
                                'Projected NOI ($)': projected_noi
                            })
                            fig_proj = px.line(
                                projection_data,
                                x='Year',
                                y='Projected NOI ($)',
                                title="5-Year NOI Projection (3% Annual Growth)",
                                markers=True,
                                line_shape='spline'
                            )
                            fig_proj.update_traces(line_color='#008B8B', marker=dict(size=10, color='#008B8B'))
                            fig_proj.update_layout(yaxis_tickformat='$,.0f')
                            st.plotly_chart(fig_proj, use_container_width=True)

                        # Market Comparable Properties
                        if st.session_state.texas_data is not None and property_size > 0:
                            st.markdown("---")
                            st.markdown("#### Market Comparables")

                            # Find similar properties
                            df = st.session_state.texas_data
                            if 'sale_price' in df.columns and 'land_acres' in df.columns:
                                # Filter properties with similar size (+/- 50%)
                                min_size = property_size * 0.5
                                max_size = property_size * 1.5

                                comparable = df[
                                    (df['sale_price'].notna()) &
                                    (df['land_acres'].notna()) &
                                    (df['land_acres'] >= min_size) &
                                    (df['land_acres'] <= max_size)
                                ].copy()

                                if len(comparable) > 0:
                                    comparable['price_per_acre'] = comparable['sale_price'] / comparable['land_acres']
                                    comparable = comparable.sort_values('sale_price', ascending=False).head(5)

                                    st.markdown(f"Found {len(comparable)} similar properties ({min_size:.1f} - {max_size:.1f} acres)")

                                    display_cols = ['sale_price', 'land_acres', 'price_per_acre']
                                    if 'document_date' in comparable.columns:
                                        display_cols.insert(0, 'document_date')
                                    if 'site_class' in comparable.columns:
                                        display_cols.insert(1, 'site_class')

                                    comparable_display = comparable[display_cols].copy()
                                    comparable_display.columns = [col.replace('_', ' ').title() for col in comparable_display.columns]

                                    st.dataframe(comparable_display, use_container_width=True)
                                else:
                                    st.info("No comparable properties found in the dataset for this size range.")

                        st.session_state.analysis_complete = True

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            else:
                st.warning("Please enter inspection report/property notes and ensure the system is initialized.")

with tab2:
    st.markdown("### Texas Market Intelligence")
    st.markdown("Real-time insights from 1,193 Texas commercial property sales (2018-2025)")

    if st.session_state.texas_data is not None and len(st.session_state.texas_data) > 0:
        df = st.session_state.texas_data

        # Market Overview
        st.markdown("#### Market Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Properties", f"{len(df):,}")
        with col2:
            # Count properties with sale_price data
            with_prices = df[df['sale_price'].notna()]
            avg_price = with_prices['sale_price'].mean() if len(with_prices) > 0 else 0
            st.metric("Avg Sale Price", f"${avg_price:,.0f}")
        with col3:
            unique_types = df['site_class'].nunique() if 'site_class' in df.columns else 0
            st.metric("Property Types", f"{unique_types}")
        with col4:
            date_range = "2018-2025"
            if 'document_date' in df.columns:
                dates = pd.to_datetime(df['document_date'], errors='coerce').dropna()
                if len(dates) > 0:
                    date_range = f"{dates.min().year}-{dates.max().year}"
            st.metric("Date Range", date_range)

        st.markdown("---")

        # Property Type Distribution
        st.markdown("#### Property Type Distribution")
        if 'site_class' in df.columns:
            type_counts = df['site_class'].value_counts().head(10)
            fig = px.bar(
                x=type_counts.index,
                y=type_counts.values,
                labels={'x': 'Property Type', 'y': 'Count'},
                title="Top 10 Property Types in Texas Dataset",
                color_discrete_sequence=['#008B8B']
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Price Trends Over Time
        st.markdown("#### Sale Price Trends")
        if 'document_date' in df.columns and 'sale_price' in df.columns:
            df_prices = df[df['sale_price'].notna()].copy()
            df_prices['year'] = pd.to_datetime(df_prices['document_date'], errors='coerce').dt.year

            if len(df_prices) > 0:
                yearly_avg = df_prices.groupby('year')['sale_price'].agg(['mean', 'median', 'count']).reset_index()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yearly_avg['year'],
                    y=yearly_avg['mean'],
                    mode='lines+markers',
                    name='Average Price',
                    line=dict(color='#20B2AA', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=yearly_avg['year'],
                    y=yearly_avg['median'],
                    mode='lines+markers',
                    name='Median Price',
                    line=dict(color='#40E0D0', width=2, dash='dash')
                ))

                fig.update_layout(
                    title="Texas Commercial Land Sale Prices Over Time",
                    xaxis_title="Year",
                    yaxis_title="Sale Price ($)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show transaction volume
                fig2 = px.bar(
                    yearly_avg,
                    x='year',
                    y='count',
                    title="Number of Transactions by Year",
                    labels={'count': 'Number of Sales', 'year': 'Year'},
                    color_discrete_sequence=['#008B8B']
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # Price Distribution
        st.markdown("#### Price Distribution")
        if 'sale_price' in df.columns:
            prices = df[df['sale_price'].notna()]['sale_price']
            if len(prices) > 0:
                fig = px.histogram(
                    prices,
                    nbins=50,
                    title="Distribution of Sale Prices",
                    labels={'value': 'Sale Price ($)', 'count': 'Frequency'},
                    color_discrete_sequence=['#008B8B']
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show percentiles
                st.markdown("##### Price Percentiles")
                percentile_col1, percentile_col2, percentile_col3, percentile_col4, percentile_col5 = st.columns(5)

                with percentile_col1:
                    st.metric("25th %ile", f"${prices.quantile(0.25):,.0f}")
                with percentile_col2:
                    st.metric("50th %ile", f"${prices.quantile(0.50):,.0f}")
                with percentile_col3:
                    st.metric("75th %ile", f"${prices.quantile(0.75):,.0f}")
                with percentile_col4:
                    st.metric("90th %ile", f"${prices.quantile(0.90):,.0f}")
                with percentile_col5:
                    st.metric("Max", f"${prices.max():,.0f}")

        st.markdown("---")

        # Raw Data Sample
        st.markdown("#### Sample Data")
        st.markdown("Preview of Texas commercial property sales data:")
        display_cols = [col for col in ['document_date', 'site_class', 'neighborhood', 'sale_price', 'land_acres', 'address'] if col in df.columns]
        if display_cols:
            st.dataframe(df[display_cols].head(100), use_container_width=True)

    else:
        st.info("Texas market data is loading or unavailable. Please check the data folder.")

with tab3:
    st.markdown("### Pricing Model")
    st.markdown("Flexible hybrid pricing: Fixed base + pay-as-you-grow")

    # Pricing Overview
    st.markdown("""
    <div style='background-color: #FFF5E6; padding: 2rem; border-radius: 10px; border-left: 4px solid #BF5700; margin-bottom: 2rem;'>
        <h2 style='color: #BF5700; margin-top: 0;'>Combined Pricing Structure</h2>
        <h3 style='color: #000000;'>$1,000/month Base + Usage Overage</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1rem;'>
            <div>
                <h4 style='color: #BF5700;'>Fixed Base Fee: $1,000/month</h4>
                <ul style='color: #333333;'>
                    <li><strong>Includes 100,000 tokens/month</strong></li>
                    <li>Access to 1,193 Texas market comparables</li>
                    <li>Real-time market intelligence</li>
                    <li>All platform features</li>
                    <li>Priority support</li>
                </ul>
            </div>
            <div>
                <h4 style='color: #BF5700;'>Usage Overage: $0.0005/token</h4>
                <ul style='color: #333333;'>
                    <li><strong>Only charged beyond 100K tokens</strong></li>
                    <li>Pay only for what you actually use</li>
                    <li>No penalty for low usage months</li>
                    <li>Scales with your business needs</li>
                    <li>Transparent, predictable pricing</li>
                </ul>
            </div>
        </div>
        <p style='color: #666; font-style: italic; margin-top: 1rem; margin-bottom: 0;'>
            <strong>Example:</strong> Use 50K tokens? Pay $1,000. Use 150K tokens? Pay $1,025 ($1,000 base + $25 overage).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Pricing Comparison Chart
    st.markdown("#### Monthly Cost by Usage Level")

    # Create data for comparison
    analyses = [25, 50, 100, 200, 500, 1000]
    avg_tokens_per_analysis = 2000  # Average tokens per property analysis
    base_fee = 1000
    included_tokens = 100000
    overage_rate = 0.0005

    # Calculate hybrid pricing
    total_costs = []
    base_fees = []
    overage_fees = []

    for num_analyses in analyses:
        total_tokens = num_analyses * avg_tokens_per_analysis

        if total_tokens <= included_tokens:
            # Within base plan
            total_cost = base_fee
            overage = 0
        else:
            # Base + overage
            overage_tokens = total_tokens - included_tokens
            overage = overage_tokens * overage_rate
            total_cost = base_fee + overage

        total_costs.append(total_cost)
        base_fees.append(base_fee)
        overage_fees.append(overage)

    pricing_data = pd.DataFrame({
        'Properties Analyzed': analyses,
        'Total Cost': total_costs,
        'Base Fee': base_fees,
        'Overage Fee': overage_fees
    })

    # Create stacked bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Base Fee ($1,000)',
        x=pricing_data['Properties Analyzed'],
        y=pricing_data['Base Fee'],
        marker_color='#20B2AA',
        text=pricing_data['Base Fee'],
        texttemplate='$%{text:,.0f}',
        textposition='inside'
    ))
    fig.add_trace(go.Bar(
        name='Usage Overage',
        x=pricing_data['Properties Analyzed'],
        y=pricing_data['Overage Fee'],
        marker_color='#40E0D0',
        text=pricing_data['Overage Fee'],
        texttemplate='$%{text:,.0f}',
        textposition='outside'
    ))

    fig.update_layout(
        title='Hybrid Pricing: Base Fee + Usage Overage',
        xaxis_title='Number of Properties Analyzed per Month',
        yaxis_title='Monthly Cost ($)',
        barmode='stack',
        yaxis_tickformat='$,.0f',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show pricing breakdown table
    st.markdown("##### Pricing Breakdown by Usage")
    pricing_table = pd.DataFrame({
        'Properties/Month': analyses,
        'Total Tokens': [a * avg_tokens_per_analysis for a in analyses],
        'Base Fee': ['$1,000'] * len(analyses),
        'Overage Fee': [f'${o:,.2f}' for o in overage_fees],
        'Total Monthly Cost': [f'${c:,.2f}' for c in total_costs]
    })
    st.dataframe(pricing_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Token Usage Calculator
    st.markdown("#### Token Usage Calculator")
    st.markdown("Estimate your monthly costs based on usage")

    calc_col1, calc_col2 = st.columns(2)

    with calc_col1:
        monthly_analyses = st.number_input(
            "Properties analyzed per month",
            min_value=1,
            max_value=1000,
            value=100,
            step=10
        )

        complexity = st.select_slider(
            "Analysis complexity",
            options=["Simple", "Standard", "Detailed", "Comprehensive"],
            value="Standard"
        )

    with calc_col2:
        # Token estimates by complexity
        token_estimates = {
            "Simple": 1000,
            "Standard": 2000,
            "Detailed": 4000,
            "Comprehensive": 8000
        }

        estimated_tokens = monthly_analyses * token_estimates[complexity]

        # Calculate hybrid pricing
        base_fee = 1000
        included_tokens = 100000

        if estimated_tokens <= included_tokens:
            total_cost = base_fee
            overage_cost = 0
            within_base = True
        else:
            overage_tokens = estimated_tokens - included_tokens
            overage_cost = overage_tokens * 0.0005
            total_cost = base_fee + overage_cost
            within_base = False

        st.markdown("##### Estimated Monthly Cost")
        st.metric("Total Tokens", f"{estimated_tokens:,}")
        st.metric("Base Fee", f"$1,000.00")
        st.metric("Overage Fee", f"${overage_cost:,.2f}")
        st.metric("Total Cost", f"${total_cost:,.2f}", delta=f"+${total_cost - base_fee:,.2f}" if not within_base else "Included")

        if within_base:
            remaining = included_tokens - estimated_tokens
            st.success(f"Within base plan! {remaining:,} tokens remaining this month.")
        else:
            st.info(f"Using {estimated_tokens - included_tokens:,} overage tokens at $0.0005 each.")

    st.markdown("---")

    # Value Proposition
    st.markdown("#### Why Hybrid Pricing Works Best")

    value_col1, value_col2 = st.columns(2)

    with value_col1:
        st.markdown("""
        **Benefits of Hybrid Model**
        - Predictable base cost: $1,000/month
        - 100K tokens included (50 properties)
        - No penalty for low usage months
        - Pay-as-you-grow for high volume
        - Transparent overage charges
        - Scale without surprises
        """)

    with value_col2:
        st.markdown("""
        **vs Traditional Due Diligence**
        - Traditional: $2,000-$10,000 per property
        - LoneStar AI: $1-$20 per property
        - 90% cost reduction at scale
        - 95% faster (2 min vs 2-6 weeks)
        - Unlimited re-analysis within plan
        - Texas market data included
        """)

    st.markdown("---")

    # ROI Example
    st.markdown("#### Real-World ROI Example")

    roi_col1, roi_col2, roi_col3 = st.columns(3)

    with roi_col1:
        st.markdown("""
        **Scenario: Small Investor**
        - Analyzes 30 properties/month
        - Uses ~60K tokens
        - **Cost: $1,000/month**
        - **Per property: $33**
        - Traditional cost: $60K-$300K
        """)

    with roi_col2:
        st.markdown("""
        **Scenario: Medium Investor**
        - Analyzes 100 properties/month
        - Uses ~200K tokens
        - **Cost: $1,050/month**
        - **Per property: $10.50**
        - Traditional cost: $200K-$1M
        """)

    with roi_col3:
        st.markdown("""
        **Scenario: Large Fund**
        - Analyzes 500 properties/month
        - Uses ~1M tokens
        - **Cost: $1,450/month**
        - **Per property: $2.90**
        - Traditional cost: $1M-$5M
        """)

with tab4:
    st.markdown("### Document Upload")
    st.info("Upload additional property documents, financial statements, or market data to enhance analysis.")

    uploaded_file = st.file_uploader(
        "Upload Document (PDF, TXT, CSV, XLSX)",
        type=["pdf", "txt", "csv", "xlsx"]
    )

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.info("Document processing feature coming soon. For now, paste inspection reports directly in the Property Analysis tab.")

with tab5:
    st.markdown("### About LoneStar AI")

    st.markdown("""
    ## 2-Minute Analysis, Texas-Sized Value

    LoneStar AI is an AI-powered property due diligence assistant designed specifically for Texas commercial real estate investors.

    ### What We Do

    - Analyze properties in **2 minutes** using 1,193 Texas commercial land sales (2018-2025)
    - Upload inspection reports, financials, leases and get instant valuation, risk score, and buy/pass/negotiate decision
    - Powered by **Llama 3.1 8B** (Groq), **ChromaDB**, and **LangChain RAG**

    ### The Problem We Solve

    Traditional due diligence in Texas commercial real estate takes **2-6 weeks** and costs **$2,000-$10,000** per property. In fast-moving markets like Austin (where properties sell in 7 days average), you lose deals while waiting for analysis.

    ### Our Solution

    - **95% faster**: 2 minutes vs 2-6 weeks
    - **90% cheaper**: $1,000/month vs $2,000+ per property
    - **Data-driven**: Backed by real Texas market data
    - **AI-powered**: Full RAG pipeline with market context

    ### Tech Stack

    - **LLM**: Llama 3.1 8B (Groq API)
    - **Vector DB**: ChromaDB
    - **Framework**: LangChain RAG
    - **Data**: 1,193 Texas properties (2018-2025)
    - **Frontend**: Streamlit

    ### Market Focus

    Starting with **Texas commercial land** because:

    - **Growth Market**: TX commercial land sales up 34% since 2020
    - **Tech-Savvy Investors**: Austin/Dallas early PropTech adopters
    - **Data Advantage**: 1,193 cleaned TX property records

    ---

    ### Team

    Built by **Albert Opher**, **Josh Johnson**, **Shawn Gutierrez**, and **Matt Inamine**

    ---

    **OIDD 2550 - Lab 5: LLM Pitch Project**

    Fall 2025
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>LoneStar AI - AI-Powered Property Due Diligence for Texas | "
    "Powered by Groq, ChromaDB, and LangChain</p>",
    unsafe_allow_html=True
)
