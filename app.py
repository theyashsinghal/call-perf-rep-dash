import streamlit as st
import pandas as pd
from io import BytesIO

def calculate_report(df, avg_col=None):
    # 1. Standardize text columns
    df['outcome'] = df['outcome'].str.strip().str.lower()
    
    # 2. Check for required columns
    # Note: We still check for 'recording_url' to maintain file compatibility, 
    # even though logic now prioritizes 'contacted'.
    required_cols = ['bot', 'mobile_number', 'outcome', 'contacted', 'date', 'recording_url']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"Missing columns: {', '.join(missing_cols)}")
        return None, None

    # 3. Data Cleaning
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Ensure 'contacted' is numeric (0 or 1)
    df['contacted'] = pd.to_numeric(df['contacted'], errors='coerce').fillna(0).astype(int)
    
    df['bot'] = df['bot'].fillna('Blank_Bot_Name')
    df['recording_url'] = df['recording_url'].fillna('').astype(str).str.strip()

    all_bots = sorted(df['bot'].unique())

    # --- LOGIC UPDATE: ALL ATTEMPTS (No Deduplication) ---
    
    # Define flags directly on the main dataframe (per attempt)
    df['connected_flag'] = df['contacted'] == 1
    df['not_connected_flag'] = df['contacted'] == 0
    
    follow_up_exclude = {"assign to live agent", "converted", "lost"}
    
    df['converted_flag'] = df['outcome'] == 'converted'
    df['lost_flag'] = df['outcome'] == 'lost'
    df['assigned_to_agent_flag'] = df['outcome'] == 'assign to live agent'
    df['follow_up_flag'] = ~df['outcome'].isin(follow_up_exclude)

    # Prepare Rows for the Summary Table
    unique_leads_row = {'Metric': 'Unique Numbers (Ref)'} # Just for reference
    total_attempts_row = {'Metric': 'Total Attempts'}
    avg_attempts_row = {'Metric': 'Avg Attempts (N/A)'} # Not applicable in "All Attempts" view
    
    connected_row = {'Metric': 'Connected'}
    connectivity_perc_row = {'Metric': 'Connectivity %'}
    not_connected_row = {'Metric': 'Not Connected'}
    follow_up_row = {'Metric': 'Follow Up'}
    assigned_agent_row = {'Metric': 'Assigned to human agent'}
    lost_row = {'Metric': 'Lost'}
    converted_row = {'Metric': 'Converted'}

    # Create sheets based on the flags (containing all matching rows)
    sheets_by_category = {}
    sheets_by_category["connected"] = df[df['connected_flag']]
    sheets_by_category["not_connected"] = df[df['not_connected_flag']]
    sheets_by_category["converted"] = df[df['converted_flag']]
    sheets_by_category["lost"] = df[df['lost_flag']]
    sheets_by_category["assigned_to_human_agent"] = df[df['assigned_to_agent_flag']]
    sheets_by_category["follow_up"] = df[df['follow_up_flag']]

    # Detailed summary sheet (All rows sorted)
    lead_summary_cols = [
        'bot', 'mobile_number', 'date', 'outcome',
        'connected_flag', 'not_connected_flag', 'converted_flag', 'lost_flag',
        'assigned_to_agent_flag', 'follow_up_flag'
    ]
    sheets_by_category["lead_summary"] = df[lead_summary_cols].sort_values(
        ['bot', 'date'], ascending=[True, False]
    )

    # Calculate Metrics per Bot
    for bot_name in all_bots:
        bot_df_all = df[df['bot'] == bot_name]
        
        # Base count is now the total number of rows (attempts)
        total_attempts = len(bot_df_all)
        unique_leads_count = bot_df_all['mobile_number'].nunique() # Kept for reference
        
        # Summing flags (Count of attempts that met criteria)
        connected_count = bot_df_all['connected_flag'].sum()
        not_connected_count = bot_df_all['not_connected_flag'].sum()
        follow_up_count = bot_df_all['follow_up_flag'].sum()
        assigned_to_agent = bot_df_all['assigned_to_agent_flag'].sum()
        lost = bot_df_all['lost_flag'].sum()
        converted = bot_df_all['converted_flag'].sum()

        # Calculation: (Connected Attempts / Total Attempts)
        connectivity_perc = round((connected_count / total_attempts), 2) if total_attempts > 0 else 0.0

        # Populate rows
        unique_leads_row[bot_name] = unique_leads_count
        total_attempts_row[bot_name] = total_attempts
        avg_attempts_row[bot_name] = 1.0 # Conceptually 1 per row
        
        connected_row[bot_name] = connected_count
        connectivity_perc_row[bot_name] = connectivity_perc
        not_connected_row[bot_name] = not_connected_count
        follow_up_row[bot_name] = follow_up_count
        assigned_agent_row[bot_name] = assigned_to_agent
        lost_row[bot_name] = lost
        converted_row[bot_name] = converted

    report_data = [
        unique_leads_row,
        total_attempts_row,
        avg_attempts_row,
        connected_row,
        connectivity_perc_row,
        not_connected_row,
        follow_up_row,
        assigned_agent_row,
        lost_row,
        converted_row
    ]
    
    # Optional Average Column Logic
    if avg_col is not None and avg_col in df.columns:
        avg_row = {'Metric': f"Avg {avg_col}"}
        for bot_name in all_bots:
            bot_df = df[df['bot'] == bot_name]
            avg_val = round(bot_df[avg_col].mean(), 2) if not bot_df.empty else 0.0
            avg_row[bot_name] = avg_val
        report_data.append(avg_row)

    report_df = pd.DataFrame(report_data)
    report_df = report_df.set_index('Metric')

    return report_df, sheets_by_category


def style_summary_df(df):
    def style_data_rows(s):
        style = 'background-color: #C0C0C0; color: #000000; border: 1px solid #000000;'
        if s.name == 'Converted':
            style = 'background-color: #fff700; color: #000000; border: 1px solid #000000;'
        if s.name.startswith("Avg "):
            style = 'background-color: #e6f2ff; color: #004080; border: 1px solid #004080; font-style: italic;'
        return [style] * len(s)

    def style_index_cells(label):
        style = 'background-color: #C0C0C0; color: #000000; font-weight: bold; border: 1px solid #000000;'
        if label == 'Converted':
            style = 'background-color: #fff700; color: #000000; font-weight: bold; border: 1px solid #000000;'
        if label.startswith("Avg "):
            style = 'background-color: #e6f2ff; color: #004080; font-weight: bold; border: 1px solid #004080; font-style: italic;'
        return style

    styler = df.style
    styler = styler.apply(style_data_rows, axis=1, subset=pd.IndexSlice[:, df.columns])
    styler = styler.map_index(style_index_cells, axis=0)
    styler = styler.set_table_styles([
        {'selector': 'th.col_heading', 'props': [
            ('background-color', '#fff700'),
            ('color', '#000000'),
            ('font-weight', 'bold'),
            ('border', '1px solid #000000')
        ]},
        {'selector': 'th.index_name', 'props': [
            ('background-color', '#C0C0C0'),
            ('color', '#000000'),
            ('font-weight', 'bold'),
            ('border', '1px solid #000000')
        ]}
    ], overwrite=True)
    styler = styler.format('{:.2f}', subset=pd.IndexSlice[['Avg Attempts (N/A)', 'Connectivity %'], :])
    return styler


def style_generic_df(df):
    styler = df.style
    styler = styler.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#fff700'),
            ('color', '#000000'),
            ('font-weight', 'bold')
        ]}
    ], overwrite=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        styler = styler.format('{:.2f}', subset=numeric_cols)
    return styler


# --- Main Application ---
st.title("Call Performance Report Generator")

uploaded_files = st.file_uploader(
    "Upload Call Log Files (CSV or XLSX)", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True
)

if uploaded_files:
    dataframes = []
    
    # Read and aggregate all files
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, dtype={'mobile_number': str})
            else:
                df = pd.read_excel(uploaded_file, dtype={'mobile_number': str})
            dataframes.append(df)
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
            
    if dataframes:
        raw_df = pd.concat(dataframes, ignore_index=True)

        st.markdown("### Raw Data Preview (Combined)")
        st.dataframe(raw_df.head(), use_container_width=True)

        numeric_cols = list(raw_df.select_dtypes(include='number').columns)

        avg_enabled = st.checkbox("Show average of a numeric column in summary", key="avg_toggle")
        avg_col = None
        if avg_enabled and numeric_cols:
            avg_col = st.selectbox("Select numeric column for average", numeric_cols, key="avg_col_select")

        # Generate Report
        report_df, sheets_by_category = calculate_report(raw_df, avg_col if avg_enabled else None)

        if report_df is not None:
            st.markdown("### Summary Report")
            styled_summary = style_summary_df(report_df)
            st.dataframe(styled_summary, use_container_width=True)

            # --- Checkbox for Extended Report ---
            extended_report = st.checkbox("Enable Extended Report (Include detailed data sheets)", value=False)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Always write Summary
                styled_summary.to_excel(writer, sheet_name="Summary", index=True)
                
                # Conditionally write detailed sheets
                if extended_report:
                    for category_name, df_cat in sheets_by_category.items():
                        styled_cat = style_generic_df(df_cat)
                        styled_cat.to_excel(writer, sheet_name=category_name, index=False)

            output.seek(0)

            st.download_button(
                label="📥 Download Styled Excel Report",
                data=output,
                file_name="Styled_Call_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error("No valid data found in the uploaded files.")
else:
    st.info("Please upload CSV or XLSX files to start the report generation.")
