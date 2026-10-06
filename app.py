import os
import streamlit as st

from agent import ask_agent


st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare AI Assistant")

st.write(
    "Ask questions about healthcare data and hospital policies."
)


# -------------------------------
# QUESTION INPUT
# -------------------------------

question = st.text_input(
    "Ask your healthcare data question:",
    placeholder="Example: Show total revenue by medical condition"
)


# -------------------------------
# ASK AGENT
# -------------------------------

if st.button("Ask"):

    if not question:

        st.warning(
            "Please enter a question."
        )

    else:

        # Remove old chart so it is not
        # displayed for a new unrelated question
        chart_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "latest_chart.png"
        )

        if os.path.exists(chart_path):
            os.remove(chart_path)

        with st.spinner(
            "Analyzing your question..."
        ):

            try:

                answer = ask_agent(
                    question
                )

                st.subheader("Answer")

                st.write(answer)

                # Display chart if agent created one
                if os.path.exists(chart_path):

                    st.subheader(
                        "Visualization"
                    )

                    st.image(
                        chart_path
                    )

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )