# callaway_platform

# ⛳ Callaway Sales Analytics Platform

<img width="1403" alt="image" src="https://github.com/user-attachments/assets/c06391ca-b1d0-49bd-8de5-88aafded179a" />

An intelligent sales analytics platform for Callaway Golf equipment, powered by Snowflake Cortex Analyst and built with Streamlit. This application enables natural language querying of sales data across products, regions, channels, and customer segments.

## 🚀 Features

### Core Capabilities
- **Natural Language Querying**: Ask questions in plain English about your sales data
- **AI-Powered Insights**: Automatic generation of business insights and explanations using Cortex Complete
- **Interactive Visualizations**: Dynamic chart creation with multiple visualization types
- **Smart Chart Configuration**: Auto-detection of optimal chart settings based on data structure
- **Multi-Model Support**: Choose from 30+ LLM models including Claude, GPT-4, Llama, and more

### User Interface
- **Dark/Light Theme Toggle**: Customizable interface themes for better visibility
- **Question Browser**: Pre-built questions organized by category
- **Follow-up Suggestions**: Contextual suggestions based on your queries
- **Export Capabilities**: Download data as CSV or charts as PNG

## 📊 Data Model

### Database Structure
The platform uses two main tables in Snowflake:

#### PRODUCTS Table
- **Product Catalog**: Complete inventory of Callaway products
- **Categories**: Clubs (Drivers, Irons, Wedges, Putters), Balls, Bags, Apparel
- **Brands**: Callaway, Odyssey, TravisMathew
- **Pricing**: MSRP and cost information for margin calculations

#### SALES_DATA Table
- **Transaction Records**: Individual sales with date, location, and customer info
- **Geographic Data**: State and region-level sales tracking
- **Channel Performance**: Retail, Online, and Pro Shop sales
- **Customer Segmentation**: Consumer, Corporate, and Pro customers

### Semantic Model (YAML)
The `callaway_sales_performance.yaml` file defines:
- Table relationships and joins
- Dimension and fact definitions
- Pre-verified analytical queries
- Business-friendly column descriptions

## 🛠️ Technical Architecture

### Components
1. **Streamlit App** (`app.py`)
   - Interactive web interface
   - Session state management
   - Real-time chart rendering

2. **Snowflake Integration**
   - Cortex Analyst for natural language processing
   - Snowpark for data access
   - Cortex Complete for AI insights

3. **Visualization Engine**
   - Plotly for interactive charts
   - Smart chart type detection
   - Theme-aware styling

### Models Available
- OpenAI: GPT-4.1, O4-mini
- Anthropic: Claude 4 Opus/Sonnet, Claude 3.5/3.7 Sonnet
- Meta: Llama 2/3/3.1/3.2/3.3/4 variants
- Mistral: Large, Large2, 7B, Mixtral
- Snowflake: Arctic, Llama implementations
- Others: DeepSeek R1, Gemma, Jamba, Reka

1. **Ask questions naturally**
   - "What were our total sales last month?"
   - "Show me top 10 products by revenue"
   - "Compare California vs Texas sales performance"
   - "Which golf balls have the highest profit margins?"

2. **Explore pre-built questions**
   - Click category buttons to browse questions
   - Use suggested follow-ups for deeper analysis

3. **Customize visualizations**
   - Select chart type (bar, line, area, scatter, pie)
   - Configure axes and groupings
   - Use auto-configure for optimal settings
