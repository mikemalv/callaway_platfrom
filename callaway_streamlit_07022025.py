import _snowflake
import json
import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from snowflake.snowpark.context import get_active_session

# Configure page and sidebar
st.set_page_config(
    page_title="Callaway Sales Analytics",
    page_icon="⛳",
    layout="wide"
)

# Initialize theme in session state
if 'chart_theme' not in st.session_state:
    st.session_state.chart_theme = 'dark'

# Initialize unique ID counter to prevent duplicate key errors
if 'unique_id_counter' not in st.session_state:
    st.session_state.unique_id_counter = 0

# Reset counter if it gets too large
if st.session_state.unique_id_counter > 10000:
    st.session_state.unique_id_counter = 0

# Debug mode (set to True to see session state)
debug = False

# Custom CSS (enhanced with chart styling and theme support)
st.markdown(f"""
    <style>
    /* Apply theme class to root */
    .stApp {{
        data-theme: "{st.session_state.chart_theme}";
    }}
    
    /* Base styles */
    [data-testid="stSidebar"] {{
        background-color: rgb(5, 145, 254);
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stDateInput label {{
        color: white !important;
    }}
    
    /* Button styling for both modes */
    .stButton > button {{
        background-color: #003087 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 4px !important;
    }}
    .stButton > button:hover {{
        background-color: #004087 !important;
        color: white !important;
        border: none !important;
    }}
    
    /* Theme-specific styles based on session state */
    {'''
    /* Dark mode styles */
    .stApp,
    .stApp p,
    .stApp span,
    .stApp div,
    .stApp label,
    .stApp .stMarkdown,
    .element-container,
    [data-testid="stMarkdownContainer"] {{
        color: rgba(255, 255, 255, 0.9) !important;
    }}
    
    .stApp {{
        background-color: #0E1117 !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: #262730 !important;
        color: white !important;
    }}
    
    /* Headers in dark mode */
    h1, h2, h3, h4, h5, h6 {{
        color: white !important;
    }}
    
    /* Chat input in dark mode */
    .stChatInput textarea {{
        color: white !important;
        background-color: #262730 !important;
    }}
    
    /* Code blocks in dark mode */
    .stCodeBlock {{
        background-color: #1e1e1e !important;
    }}
    ''' if st.session_state.chart_theme == 'dark' else '''
    /* Light mode styles */
    .stApp,
    .stApp p,
    .stApp span,
    .stApp div,
    .stApp label,
    .stApp .stMarkdown,
    .element-container,
    [data-testid="stMarkdownContainer"] {{
        color: rgba(0, 0, 0, 0.9) !important;
    }}
    
    .stApp {{
        background-color: white !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: white !important;
        color: black !important;
    }}
    
    /* Headers in light mode */
    h1, h2, h3, h4, h5, h6 {{
        color: black !important;
    }}
    
    /* Chat input in light mode */
    .stChatInput textarea {{
        color: black !important;
        background-color: #f5f5f5 !important;
    }}
    
    /* Code blocks in light mode */
    .stCodeBlock {{
        background-color: #f5f5f5 !important;
    }}
    '''}
    
    /* Tab styling for better visibility */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 4px;
        padding: 8px 16px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: #0591fe !important;
        color: white !important;
    }}
    </style>
""", unsafe_allow_html=True)

def refresh_snowflake_connection():
    """Refresh Snowflake connection and clear cached data"""
    try:
        # Reset counter
        st.session_state.unique_id_counter = 0
        
        # Clear session state except preserved keys
        preserved_keys = ['chart_theme', 'unique_id_counter', 'all_questions', 'current_category']
        for key in list(st.session_state.keys()):
            if key not in preserved_keys:
                del st.session_state[key]
        
        # Get fresh Snowflake session
        session = get_active_session()
        if session:
            session.clear_queries()  # Clear any cached queries
        
        # Reinitialize messages
        st.session_state.messages = []
        st.session_state.suggestions = []
        st.session_state.active_suggestion = None
        
        st.success("✅ Data connection refreshed successfully!")
        time.sleep(1)  # Give user time to see success message
        st.rerun()  # Rerun the app to reflect changes
        
    except Exception as e:
        st.error(f"❌ Error refreshing data: {str(e)}")

# Sidebar
with st.sidebar:
    # Move logo to top of sidebar
    col1, col2, col3 = st.columns([1, 2, 1])  # For center alignment
    with col2:
        st.image("https://swingfit.net/wp-content/uploads/2022/09/Callaway-Golf-Logo-596x343-1.png", width=150)
    st.divider()
    
    st.title("Select Option")
    selected_model = st.sidebar.selectbox(
    "Select Your Model:",
        (
            'claude-4-opus',
            'claude-4-sonnet',
            'claude-3-7-sonnet',
            'claude-3-5-sonnet',
            'deepseek-r1',
            'gemma-7b',
            'jamba-1.5-mini',
            'jamba-1.5-large',
            'jamba-instruct',
            'llama2-70b-chat',
            'llama3-8b',
            'llama3-70b',
            'llama3.1-8b',
            'llama3.1-70b',
            'llama3.1-405b',
            'llama3.2-1b',
            'llama3.2-3b',
            'llama3.3-70b',
            'llama4-maverick',
            'llama4-scout',
            'mistral-large',
            'mistral-large2',
            'mistral-7b',
            'mixtral-8x7b',
            'openai-gpt-4.1',
            'openai-o4-mini',
            'reka-core',
            'reka-flash',
            'snowflake-arctic',
            'snowflake-llama-3.1-405b',
            'snowflake-llama-3.3-70b'
        ),
        index=24,  # Set default to openai-gpt-4.1
        key="model_name"
    )
    
    # Chart Theme Toggle
    st.divider()
    st.subheader("Chart Settings")
    theme_col1, theme_col2 = st.columns(2)
    with theme_col1:
        if st.button("🌙 Dark", key="dark_theme", use_container_width=True):
            st.session_state.chart_theme = 'dark'
    with theme_col2:
        if st.button("☀️ Light", key="light_theme", use_container_width=True):
            st.session_state.chart_theme = 'light'
    
    st.caption(f"Current theme: {st.session_state.chart_theme.title()}")
    
    # Add Start Over and Refresh buttons
    st.divider()
    if st.button("Start Over"):
        try:
            # Reset the unique ID counter
            st.session_state.unique_id_counter = 0
            
            # Clear all state except essentials
            preserved_keys = ['chart_theme', 'unique_id_counter', 'all_questions', 'current_category']
            for key in list(st.session_state.keys()):
                if key not in preserved_keys:
                    del st.session_state[key]
            
            # Reset to Popular category
            st.session_state.current_category = 'Popular'
            
            st.success("✅ Session cleared successfully!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing session state: {str(e)}")
    
    if st.button("Refresh Data", key="refresh_button"):
        refresh_snowflake_connection()

# Constants
DATABASE    = "CALLAWAY_PLATFORM"
SCHEMA      = "PUBLIC"
STAGE       = "CALLAWAY_STAGE"
FILE        = "callaway_sales_performance.yaml"

# Main content area
st.divider()

# Apply theme-based styling for the title
is_dark_mode = st.session_state.get('chart_theme', 'dark') == 'dark'
title_color = "#FFFFFF" if is_dark_mode else "#000000"
theme_indicator = "🌙 Dark Mode" if is_dark_mode else "☀️ Light Mode"

st.markdown(f"""
<div style="margin-bottom: 20px;">
    <h2 style="color: {title_color}; margin-bottom: 5px;">⛳ Callaway Sales Analytics</h2>
    <p style="color: {title_color}; font-size: 0.9em; opacity: 0.8;">Chart Theme: {theme_indicator} (Toggle in sidebar)</p>
</div>
""", unsafe_allow_html=True)

st.divider()

def send_message(prompt: str) -> dict:
    """Calls the REST API and returns the response."""
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
    }
    resp = _snowflake.send_snow_api_request(
        "POST",
        f"/api/v2/cortex/analyst/message",
        {},
        {},
        request_body,
        {},
        30000,
    )
    if resp["status"] < 400:
        return json.loads(resp["content"])
    else:
        raise Exception(
            f"Failed request with status {resp['status']}: {resp}"
        )

def generate_explanation(question: str, sql_query: str, df: pd.DataFrame, model: str) -> str:
    """Generate an explanation of the query results using Cortex Complete"""
    try:
        session = get_active_session()
        
        # Prepare data summary for the prompt
        data_summary = f"""
        Row count: {len(df)}
        Columns: {', '.join(df.columns.tolist())}
        """
        
        # Add sample data if not too large
        if len(df) > 0:
            sample_rows = min(5, len(df))
            data_summary += f"\nFirst {sample_rows} rows:\n{df.head(sample_rows).to_string()}"
            
            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                data_summary += f"\n\nNumeric column statistics:\n{df[numeric_cols].describe().to_string()}"
        
        # Create the prompt for explanation
        explanation_prompt = f"""
        You are a business analyst explaining data insights to executives. 
        
        User Question: {question}
        
        SQL Query executed:
        {sql_query}
        
        Data Results Summary:
        {data_summary}
        
        Please provide a clear, concise business explanation that includes:
        1. A direct answer to the user's question
        2. Key insights from the data (mention specific numbers)
        3. Any notable patterns or trends
        4. Business implications or recommendations
        
        Keep the explanation under 200 words and use bullet points for clarity.
        Focus on what matters to business decision-makers.
        """
        
        # Call Cortex Complete for explanation
        complete_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            '{explanation_prompt.replace("'", "''")}'
        ) as explanation
        """
        
        result = session.sql(complete_query).collect()
        if result and len(result) > 0:
            return result[0]['EXPLANATION']
        else:
            return "Unable to generate explanation."
            
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def get_chart_config_from_df(df: pd.DataFrame, chart_type: str = 'auto'):
    """Generate a smart chart configuration based on the dataframe structure"""
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    # Try to identify date columns by content
    for col in text_cols:
        try:
            pd.to_datetime(df[col])
            date_cols.append(col)
            text_cols.remove(col)
        except:
            pass
    
    config = {
        'chart_type': chart_type,
        'x_axis': None,
        'y_axis': None,
        'color': None,
        'size': None,
        'values': None,
        'names': None
    }
    
    # Smart column selection based on chart type and data
    if chart_type == 'auto':
        # Auto-detect best chart type
        if len(df) == 1 and len(numeric_cols) > 1:
            chart_type = 'bar'
            # Transpose for single row
            config['transpose'] = True
        elif len(numeric_cols) >= 2 and len(text_cols) >= 1:
            chart_type = 'bar'
        elif len(numeric_cols) >= 1 and len(text_cols) >= 1:
            chart_type = 'bar'
        else:
            chart_type = 'table'
    
    # Configure based on chart type
    if chart_type in ['bar', 'line', 'area']:
        # Prioritize meaningful columns
        if date_cols:
            config['x_axis'] = date_cols[0]
        elif text_cols:
            # Look for category-like columns
            category_keywords = ['name', 'product', 'category', 'state', 'region', 'brand', 'type', 'channel']
            category_cols = [col for col in text_cols if any(keyword in col.lower() for keyword in category_keywords)]
            config['x_axis'] = category_cols[0] if category_cols else text_cols[0]
        
        if numeric_cols:
            # Prioritize financial metrics
            financial_keywords = ['revenue', 'sales', 'amount', 'total', 'profit', 'margin', 'cost', 'price']
            financial_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in financial_keywords)]
            config['y_axis'] = financial_cols[0] if financial_cols else numeric_cols[0]
    
    elif chart_type == 'scatter':
        if len(numeric_cols) >= 2:
            config['x_axis'] = numeric_cols[0]
            config['y_axis'] = numeric_cols[1]
            if text_cols:
                config['color'] = text_cols[0]
    
    elif chart_type == 'pie':
        if numeric_cols and text_cols:
            config['values'] = numeric_cols[0]
            config['names'] = text_cols[0]
    
    config['chart_type'] = chart_type
    return config

def create_chart_from_config(df: pd.DataFrame, config: dict):
    """Create a Plotly chart based on configuration"""
    
    # Theme settings
    is_dark_mode = st.session_state.get('chart_theme', 'dark') == 'dark'
    
    if is_dark_mode:
        template = 'plotly_dark'
        bg_color = 'rgba(17, 17, 17, 0.9)'
        text_color = 'rgba(255, 255, 255, 0.9)'
        grid_color = 'rgba(255, 255, 255, 0.1)'
    else:
        template = 'plotly_white'
        bg_color = 'rgba(255, 255, 255, 0.9)'
        text_color = 'rgba(0, 0, 0, 0.9)'
        grid_color = 'rgba(0, 0, 0, 0.1)'
    
    # Color palette
    colors = ['#0591fe', '#003087', '#66B2FF', '#0066CC', '#99CCFF']
    
    try:
        # Handle transposition for single row data
        if config.get('transpose') and len(df) == 1:
            # Keep first column as identifier, melt the rest
            id_col = df.columns[0]
            value_cols = [col for col in df.columns[1:] if pd.api.types.is_numeric_dtype(df[col])]
            if value_cols:
                df_melted = pd.melt(df, id_vars=[id_col], value_vars=value_cols, var_name='Metric', value_name='Value')
                df = df_melted
                config['x_axis'] = 'Metric'
                config['y_axis'] = 'Value'
        
        chart_type = config.get('chart_type', 'bar')
        
        if chart_type == 'bar':
            fig = px.bar(df, 
                        x=config.get('x_axis'), 
                        y=config.get('y_axis'),
                        color=config.get('color'),
                        title=config.get('title', 'Bar Chart'),
                        template=template,
                        color_discrete_sequence=colors)
        
        elif chart_type == 'line':
            fig = px.line(df, 
                         x=config.get('x_axis'), 
                         y=config.get('y_axis'),
                         color=config.get('color'),
                         title=config.get('title', 'Line Chart'),
                         template=template,
                         color_discrete_sequence=colors,
                         markers=True)
        
        elif chart_type == 'area':
            fig = px.area(df, 
                         x=config.get('x_axis'), 
                         y=config.get('y_axis'),
                         color=config.get('color'),
                         title=config.get('title', 'Area Chart'),
                         template=template,
                         color_discrete_sequence=colors)
        
        elif chart_type == 'scatter':
            fig = px.scatter(df, 
                            x=config.get('x_axis'), 
                            y=config.get('y_axis'),
                            color=config.get('color'),
                            size=config.get('size'),
                            title=config.get('title', 'Scatter Plot'),
                            template=template,
                            color_discrete_sequence=colors)
        
        elif chart_type == 'pie':
            fig = px.pie(df, 
                        values=config.get('values'), 
                        names=config.get('names'),
                        title=config.get('title', 'Pie Chart'),
                        template=template,
                        color_discrete_sequence=colors)
            fig.update_traces(textposition='inside', textinfo='percent+label')
        
        else:
            return None
        
        # Update layout
        fig.update_layout(
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(gridcolor=grid_color),
            yaxis=dict(gridcolor=grid_color),
            hoverlabel=dict(
                bgcolor='rgba(50, 50, 50, 0.95)' if is_dark_mode else 'rgba(250, 250, 250, 0.95)',
                font_size=12,
                font_color='white' if is_dark_mode else 'black'
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def chart_picker(df: pd.DataFrame, message_index: int):
    """Interactive chart configuration component"""
    
    # Chart types with icons
    chart_types = {
        'bar': '📊 Bar Chart',
        'line': '📈 Line Chart',
        'area': '📉 Area Chart',
        'scatter': '⚡ Scatter Plot',
        'pie': '🥧 Pie Chart'
    }
    
    # Initialize chart config in session state
    config_key = f'chart_config_{message_index}'
    if config_key not in st.session_state:
        st.session_state[config_key] = get_chart_config_from_df(df, 'bar')
    
    # Create columns for chart configuration
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Chart type selector
        selected_type = st.selectbox(
            "Chart Type",
            options=list(chart_types.keys()),
            format_func=lambda x: chart_types[x],
            key=f"chart_type_{message_index}",
            index=list(chart_types.keys()).index(st.session_state[config_key]['chart_type'])
        )
        st.session_state[config_key]['chart_type'] = selected_type
    
    with col2:
        # Title input
        title = st.text_input(
            "Chart Title",
            value=st.session_state[config_key].get('title', 'Data Visualization'),
            key=f"chart_title_{message_index}"
        )
        st.session_state[config_key]['title'] = title
    
    with col3:
        # Auto-configure button
        if st.button("🔄 Auto Configure", key=f"auto_config_{message_index}"):
            st.session_state[config_key] = get_chart_config_from_df(df, selected_type)
            st.rerun()
    
    # Show appropriate axis selectors based on chart type
    if selected_type in ['bar', 'line', 'area', 'scatter']:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox(
                "X-Axis",
                options=[None] + list(df.columns),
                index=0 if st.session_state[config_key]['x_axis'] is None else list(df.columns).index(st.session_state[config_key]['x_axis']) + 1 if st.session_state[config_key]['x_axis'] in df.columns else 0,
                key=f"x_axis_{message_index}"
            )
            st.session_state[config_key]['x_axis'] = x_axis
        
        with col2:
            # Filter numeric columns for y-axis
            numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
            y_axis = st.selectbox(
                "Y-Axis",
                options=[None] + numeric_cols,
                index=0 if st.session_state[config_key]['y_axis'] is None else numeric_cols.index(st.session_state[config_key]['y_axis']) + 1 if st.session_state[config_key]['y_axis'] in numeric_cols else 0,
                key=f"y_axis_{message_index}"
            )
            st.session_state[config_key]['y_axis'] = y_axis
        
        with col3:
            # Color grouping (optional)
            color = st.selectbox(
                "Color By (optional)",
                options=[None] + list(df.columns),
                index=0,
                key=f"color_{message_index}"
            )
            st.session_state[config_key]['color'] = color
    
    elif selected_type == 'pie':
        col1, col2 = st.columns(2)
        
        with col1:
            # Values column (numeric)
            numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
            values = st.selectbox(
                "Values",
                options=[None] + numeric_cols,
                index=0 if st.session_state[config_key]['values'] is None else numeric_cols.index(st.session_state[config_key]['values']) + 1 if st.session_state[config_key]['values'] in numeric_cols else 0,
                key=f"values_{message_index}"
            )
            st.session_state[config_key]['values'] = values
        
        with col2:
            # Names column (text)
            names = st.selectbox(
                "Labels",
                options=[None] + list(df.columns),
                index=0 if st.session_state[config_key]['names'] is None else list(df.columns).index(st.session_state[config_key]['names']) + 1 if st.session_state[config_key]['names'] in df.columns else 0,
                key=f"names_{message_index}"
            )
            st.session_state[config_key]['names'] = names
    
    # Create and display the chart
    if st.session_state[config_key].get('x_axis') or st.session_state[config_key].get('values'):
        fig = create_chart_from_config(df, st.session_state[config_key])
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"display_chart_{message_index}")
            
            # Export button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.info("💡 Hover over the chart and click the camera icon to download as PNG")
        else:
            st.warning("Please select appropriate columns for the chart")
    else:
        st.info("👆 Configure the chart settings above to visualize your data")
    
    return st.session_state[config_key]

def display_content(content: list, message_index: int = None, user_question: str = None) -> None:
    """Displays a content item for a message with enhanced Plotly charts."""
    # Ensure message_index is unique
    if message_index is None:
        message_index = len(st.session_state.messages)
    
    # Ensure unique_id_counter exists
    if 'unique_id_counter' not in st.session_state:
        st.session_state.unique_id_counter = 0
    
    # Create a stable unique ID based on message index and content hash
    content_str = str(content)[:100]  # Use first 100 chars of content for hash
    content_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
    unique_msg_id = f"msg_{message_index}_{content_hash}"
    
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            # Add some spacing for better appearance
            st.markdown("<br>", unsafe_allow_html=True)
            # Display suggestions
            for suggestion_index, suggestion in enumerate(item["suggestions"]):
                # Increment counter for each button to ensure uniqueness
                st.session_state.unique_id_counter += 1
                button_key = f"sugg_{st.session_state.unique_id_counter}"
                if st.button(
                    f"💡 {suggestion}", 
                    key=button_key,
                    use_container_width=True,
                    help="Click to ask this question"
                ):
                    st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            # First run the SQL to get results
            df = pd.DataFrame()  # Initialize empty dataframe
            
            with st.spinner("Running SQL..."):
                try:
                    session = get_active_session()
                    df = session.sql(item["statement"]).to_pandas()
                    
                    # Debug: Show data info
                    if debug:
                        with st.expander("🔍 Query Debug Info"):
                            st.info(f"Query returned {len(df)} rows and {len(df.columns)} columns")
                            st.write("Columns:", df.columns.tolist())
                            st.write("Data types:", df.dtypes.to_dict())
                            if len(df) > 0:
                                st.write("Sample data:")
                                st.write(df.head())
                    
                except Exception as e:
                    st.error(f"Error executing SQL: {str(e)}")
                    if debug:
                        st.code(item["statement"])
            
            # Generate and display the explanation/summary FIRST
            if user_question and len(df) > 0:
                with st.spinner("Generating insights..."):
                    explanation = generate_explanation(
                        user_question, 
                        item["statement"], 
                        df, 
                        st.session_state.model_name
                    )
                    
                    # Display the explanation
                    if explanation and explanation.strip():
                        st.markdown("### 💡 Summary & Insights")
                        st.markdown(explanation)
                        st.markdown("---")
            
            # Then show the SQL query (collapsed by default)
            with st.expander("SQL Query", expanded=False):
                st.code(item["statement"], language="sql")
            
            # Finally show the data results with enhanced chart picker
            if len(df.index) > 0:
                # Create tabs for data and visualization
                data_tab, viz_tab = st.tabs(["📊 Data", "📈 Visualization"])
                
                with data_tab:
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button for CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"callaway_data_{unique_msg_id}.csv",
                        mime="text/csv",
                        key=f"download_{unique_msg_id}"
                    )
                
                with viz_tab:
                    # Use the chart picker component
                    chart_config = chart_picker(df, message_index)
                    
            else:
                st.warning("No data returned from query")

def add_follow_up_suggestions(content, original_question):
    """Add contextual follow-up questions based on the query"""
    
    follow_up_questions = {
        "revenue": [
            "Break this down by month",
            "Show me the trend over the last year",
            "Which products contribute most to revenue?",
            "Compare to previous period"
        ],
        "product": [
            "Show me sales trends for this product",
            "What's the profit margin?",
            "Which regions buy this most?",
            "Compare to similar products"
        ],
        "state": [
            "Drill down to city level",
            "Show me growth rate by state",
            "Which products sell best here?",
            "Compare neighboring states"
        ],
        "channel": [
            "Show conversion rates by channel",
            "What's the average basket size?",
            "Display channel growth trends",
            "Which products perform best in each channel?"
        ],
        "driver": [
            "Compare different driver models",
            "Show me driver sales by price point",
            "Which regions prefer which drivers?",
            "Display driver market share"
        ],
        "margin": [
            "Which products have improving margins?",
            "Show margin trends over time",
            "Compare margins across categories",
            "Impact of discounts on margins"
        ]
    }
    
    # Determine which follow-ups to show based on keywords in original question
    suggested_follow_ups = []
    question_lower = original_question.lower()
    
    for keyword, questions in follow_up_questions.items():
        if keyword in question_lower:
            suggested_follow_ups.extend(questions[:2])  # Add first 2 questions
    
    # If no specific matches, add generic follow-ups
    if not suggested_follow_ups:
        suggested_follow_ups = [
            "Show me more details",
            "Compare to different time period",
            "Break down by category",
            "Display trend analysis"
        ]
    
    # Add follow-up suggestions to the response
    content.append({
        "type": "text",
        "text": "\n### 💭 You might also want to ask:"
    })
    content.append({
        "type": "suggestions",
        "suggestions": suggested_follow_ups[:4]  # Limit to 4 suggestions
    })
    
    return content

def process_message(prompt: str) -> None:
    """Processes a message and adds the response to the chat."""
    st.session_state.messages.append(
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🏌️"):
        with st.spinner("Generating response..."):
            response = send_message(prompt=prompt)
            content = response["message"]["content"]
            
            # Add follow-up suggestions
            content = add_follow_up_suggestions(content, prompt)
            
            display_content(content=content, user_question=prompt)
    st.session_state.messages.append({"role": "assistant", "content": content})

def get_category_icon(category):
    """Return an emoji icon for each category"""
    icons = {
        "Popular": "🔥",
        "Product Analysis": "🏌️",
        "Geographic Insights": "🗺️",
        "Financial Metrics": "💰",
        "Customer Insights": "👥",
        "Seasonal Trends": "📅",
        "Channel Performance": "🛒",
        "Competitive Analysis": "🏆"
    }
    return icons.get(category, "📊")

def initialize_chat():
    """Initialize chat with welcome message and relevant suggestions"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.suggestions = []
        st.session_state.active_suggestion = None
        st.session_state.current_category = "Popular"  # Track current category
        
        # Comprehensive question sets organized by category
        st.session_state.all_questions = {
            "Popular": [
                "What were our total sales revenue last month?",
                "Show me top 10 products by revenue",
                "Which states generated the most sales?",
                "Compare online vs retail channel performance"
            ],
            "Product Analysis": [
                "What are the top selling drivers by revenue?",
                "Show me sales trends for putters over the last 6 months",
                "Which golf ball SKUs have the highest profit margins?",
                "Compare performance of different iron sets"
            ],
            "Geographic Insights": [
                "Compare California vs Texas vs Florida sales",
                "Which regions show the highest growth rate?",
                "Show me sales heat map by state",
                "What are the top performing zip codes?"
            ],
            "Financial Metrics": [
                "What's our year-over-year revenue growth?",
                "Show me profit margins by product category",
                "What's the average order value by channel?",
                "Display monthly revenue trends for 2024"
            ],
            "Customer Insights": [
                "Compare consumer vs corporate customer sales",
                "What's the customer lifetime value by segment?",
                "Show me purchase frequency by customer type",
                "Which customer segments are most profitable?"
            ],
            "Seasonal Trends": [
                "Show me monthly sales patterns for the year",
                "Which products sell best in spring vs summer?",
                "How do holidays impact our sales?",
                "Compare Q4 performance across years"
            ],
            "Channel Performance": [
                "Compare online vs retail vs pro shop sales",
                "Which channel has the highest conversion rate?",
                "Show me channel profitability analysis",
                "What's the growth rate by sales channel?"
            ],
            "Competitive Analysis": [
                "How does Callaway brand compare to other brands?",
                "Show market share by product category",
                "What's our pricing position vs competitors?",
                "Display brand performance trends"
            ]
        }
        
        # Add initial welcome message
        welcome_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "👋 Welcome to Callaway Sales Analytics! I can help analyze your golf equipment sales data across products, regions, channels, and more. Choose a category below or type your own question:"
                }
            ]
        }
        
        st.session_state.messages.append(welcome_message)

# Initialize chat if needed
if "messages" not in st.session_state:
    initialize_chat()

if len(st.session_state.messages) == 0:
    initialize_chat()

# Ensure all_questions exists even if not initialized
if 'all_questions' not in st.session_state:
    st.session_state.all_questions = {
        "Popular": [
            "What were our total sales revenue last month?",
            "Show me top 10 products by revenue",
            "Which states generated the most sales?",
            "Compare online vs retail channel performance"
        ]
    }
    st.session_state.current_category = "Popular"

# Display chat history
for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"], avatar="🏌️" if message["role"] == "assistant" else None):
        # Extract the original question if this is showing historical messages
        user_question = None
        if message_index > 0 and st.session_state.messages[message_index-1]["role"] == "user":
            user_question = st.session_state.messages[message_index-1]["content"][0]["text"]
        display_content(content=message["content"], message_index=message_index, user_question=user_question)

# Display category selector and questions outside of chat history
if 'all_questions' in st.session_state and len(st.session_state.messages) > 0:
    # Add a container for the question browser
    with st.container():
        st.markdown("---")
        st.markdown("## 💡 Quick Question Browser")
        st.markdown("**Select a category below to see relevant questions you can ask:**")
        
        # Category buttons in a clean grid
        st.markdown("")  # Add spacing
        cols = st.columns(4)
        categories = list(st.session_state.all_questions.keys())
        
        for idx, category in enumerate(categories):
            col_idx = idx % 4
            with cols[col_idx]:
                # Use stable keys based on category name
                button_key = f"cat_btn_{category.replace(' ', '_')}"
                # Highlight current category with emoji
                is_selected = st.session_state.get('current_category', 'Popular') == category
                button_label = f"{get_category_icon(category)} {category}"
                if is_selected:
                    button_label = f"▶️ {button_label}"
                
                if st.button(
                    button_label,
                    key=button_key,
                    use_container_width=True,
                    help=f"View {category} questions"
                ):
                    st.session_state.current_category = category
                    st.rerun()
        
        # Display questions for current category with better formatting
        st.markdown("")  # Add spacing
        current_cat = st.session_state.get('current_category', 'Popular')
        st.markdown(f"### {get_category_icon(current_cat)} {current_cat} Questions")
        st.markdown(f"*Click any question below to ask it:*")
        
        # Create columns for questions to make better use of space
        questions = st.session_state.all_questions[current_cat]
        
        for idx, question in enumerate(questions):
            # Use hash of question for stable key
            question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
            button_key = f"q_{current_cat.replace(' ', '_')}_{idx}_{question_hash}"
            
            if st.button(
                f"💡 {question}",
                key=button_key,
                use_container_width=True,
                help="Click to ask this question"
            ):
                st.session_state.active_suggestion = question
        
        st.markdown("---")  # Separator before chat input

# Handle user input
if user_input := st.chat_input("Ask about Callaway sales, products, regions, or performance metrics..."):
    process_message(prompt=user_input)

# Handle suggestion clicks
if 'active_suggestion' in st.session_state and st.session_state.active_suggestion:
    suggestion = st.session_state.active_suggestion
    st.session_state.active_suggestion = None  # Clear immediately
    process_message(prompt=suggestion)