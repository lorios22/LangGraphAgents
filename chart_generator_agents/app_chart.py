import streamlit as st
import matplotlib
matplotlib.use('TkAgg')  # Set non-interactive backend before importing pyplot

import io
import base64
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from chart_generator_agent import research_chain

def main():
    st.title("Article Classifier Agents Interface")

    # User input
    user_input = st.text_input("Enter your prompt:")

    if st.button("Run Workflow"):
        with st.spinner("Running workflow..."):
            # Store messages to display in the interface
            output_messages = []

            for s in research_chain.stream(user_input, {"recursion_limit": 100}):
                if "__end__" not in s:
                    output_messages = []
                    output_messages.append(s)
                    print(s)
    
                if output_messages:
                    for output_message in output_messages:
                        classifiers = output_message.get('ChartGenerator', {})
                        for valor in classifiers.values():
                            for idx, elemento in enumerate(valor, start=1):
                                if idx == 3 and hasattr(elemento, 'content'):
                                        st.write(plt.figure(elemento.content))
        
if __name__ == "__main__":
    main()