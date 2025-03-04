import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Streamlit app configuration
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon="🧮")
st.title("Text to Math Problem Solver Using Google Gemma 2")

# Sidebar input for API key
groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Initialize Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find various information on the mentioned topics."
)

# Initialize Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for solving mathematical expressions. Only input mathematical expressions should be provided."
)

# Define a reasoning prompt
reasoning_prompt = """
You are an agent specialized in solving mathematical and reasoning questions. 
Logically arrive at the solution and provide a detailed step-by-step explanation.
Question: {question}
Answer:
"""

reasoning_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=reasoning_prompt
)

# Chain for reasoning-based questions
reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# Initialize the agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Chat session management
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a math assistant. Ask me any math or reasoning question!"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input for question
question = st.text_area("Enter your question:")

if st.button("Find My Answer"):
    if question.strip():
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            # Callback for Streamlit UI updates
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            # Run the agent
            response = assistant_agent.run(question, callbacks=[st_cb])

            # Store and display response
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please enter a question.")
