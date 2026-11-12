from pathlib import Path
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Machine Learning Laboratory",
    page_icon="🤖",
    layout="wide",
)

experiments = {
    1: {
        "title": "Import, Load and View a Dataset",
        "date": "02-09-2026",
        "group": "Exp 1 to 4",
        "file": "exp1_read_view_load.py",
        "objective": "Import, load and view the dataset.",
    },
    2: {
        "title": "Summary and Statistics of Dataset",
        "date": "02-09-2026",
        "group": "Exp 1 to 4",
        "file": "exp2_summary.py",
        "objective": "Display the dataset summary and statistical information.",
    },
    3: {
        "title": "Linear Regression",
        "date": "02-09-2026",
        "group": "Exp 1 to 4",
        "file": "exp3_linear_regression.py",
        "objective": "Implement linear regression for prediction.",
    },
    4: {
        "title": "Bayesian Logistic Regression and SVM",
        "date": "02-09-2026",
        "group": "Exp 1 to 4",
        "file": "exp4_classification.py",
        "objective": "Implement Bayesian logistic regression and SVM for classification.",
    },
}


def list_csv_datasets() -> dict[str, str]:
    """Return all CSV files in the project directory in a clean, readable format."""
    dataset_map = {}
    for csv_file in sorted(ROOT.glob("*.csv")):
        label = csv_file.stem.replace("_", " ").title()
        dataset_map[label] = csv_file.name
    return dataset_map


st.sidebar.title("🤖 ML Laboratory")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📚 Experiments", "📊 Dataset Analysis"],
)

if page == "🏠 Dashboard":
    st.title("🤖 Machine Learning Laboratory")
    st.write("Interactive dashboard for the practical machine learning experiments.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Experiments", len(experiments))
    with col2:
        st.metric("Available Scripts", len(experiments))
    with col3:
        st.metric("CSV Datasets", len(list_csv_datasets()))

    st.divider()
    st.subheader("📋 List of Practice Experiments")

    table_rows = [
        {
            "S.No": number,
            "Experiment": exp["title"],
            "Experiment Group": exp["group"],
            "Date": exp["date"],
        }
        for number, exp in experiments.items()
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.info("Use the sidebar to view the code, run an experiment, or inspect the available dataset files.")

elif page == "📚 Experiments":
    st.title("📚 Machine Learning Experiments")

    selected_number = st.selectbox(
        "Select Experiment",
        list(experiments.keys()),
        format_func=lambda x: f"Experiment {x} - {experiments[x]['title']}",
    )
    exp = experiments[selected_number]
    st.divider()
    st.header(f"Experiment {selected_number}: {exp['title']}")
    st.info(f"**Objective:** {exp['objective']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Experiment", selected_number)
    with col2:
        st.metric("Date", exp["date"])
    with col3:
        st.metric("Group", exp["group"])

    st.divider()
    st.subheader("💻 Python Code")

    code_file = ROOT / exp["file"]
    if code_file.exists():
        code = code_file.read_text(encoding="utf-8")
        st.code(code, language="python")

        st.subheader("▶️ Execute Experiment")
        if st.button(f"Run Experiment {selected_number}", type="primary"):
            try:
                result = subprocess.run(
                    [sys.executable, str(code_file)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                st.subheader("📤 Output")
                if result.stdout:
                    st.code(result.stdout, language="text")
                if result.stderr:
                    st.error("Error / Warning")
                    st.code(result.stderr, language="text")

                if result.returncode == 0:
                    st.success("Experiment executed successfully.")
                else:
                    st.error("Experiment execution failed.")

            except subprocess.TimeoutExpired:
                st.error("The experiment took too long to execute.")
            except Exception as error:
                st.error(f"Execution error: {error}")
    else:
        st.warning(f"Code file '{exp['file']}' has not been created yet.")
        st.info("Create the corresponding Python file in the same folder as app.py.")

elif page == "📊 Dataset Analysis":
    st.title("📊 Dataset Analysis")

    available_datasets = {}
    for name, filename in list_csv_datasets().items():
        try:
            available_datasets[name] = pd.read_csv(ROOT / filename)
        except Exception as error:
            st.error(f"Error loading {filename}: {error}")

    if not available_datasets:
        st.error("No CSV files found in the project folder.")
        st.stop()

    selected_dataset = st.selectbox("Select Dataset", list(available_datasets.keys()))
    data = available_datasets[selected_dataset]
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", data.shape[0])
    with col2:
        st.metric("Columns", data.shape[1])
    with col3:
        st.metric("Missing Values", int(data.isnull().sum().sum()))
    with col4:
        st.metric("Duplicate Rows", int(data.duplicated().sum()))

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "📊 Statistics", "🔍 Information", "❌ Missing Values"])

    with tab1:
        st.subheader("First 5 Rows")
        st.dataframe(data.head(), use_container_width=True)
        st.subheader("Last 5 Rows")
        st.dataframe(data.tail(), use_container_width=True)

    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(data.describe(include="all").transpose(), use_container_width=True)

    with tab3:
        information = pd.DataFrame(
            {
                "Column": data.columns,
                "Data Type": data.dtypes.astype(str),
                "Non-Null Values": data.notnull().sum().values,
                "Missing Values": data.isnull().sum().values,
                "Unique Values": data.nunique().values,
            }
        )
        st.dataframe(information, use_container_width=True)

    with tab4:
        missing = data.isnull().sum()
        missing_df = pd.DataFrame({"Column": missing.index, "Missing Values": missing.values})
        st.dataframe(missing_df, use_container_width=True)
        if missing.sum() == 0:
            st.success("✅ No missing values found.")
        else:
            st.warning(f"Total missing values: {missing.sum()}")

st.sidebar.divider()
st.sidebar.caption("Machine Learning Laboratory\nPython + Pandas + Scikit-learn + Streamlit")