import streamlit as st
from openai import OpenAI
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from typing_extensions import Annotated, TypedDict

client = OpenAI()

class GraphOutput(TypedDict):
    plot_code: Annotated[str, "The Syntactically valid Python code snippet for generating the plot and saving it as 'plot.png'."]
    plot_type: Annotated[str, "The type of plot (e.g., 'bar', 'line', 'scatter', etc.)."]

# Initialize the SQL database connection
from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Initialize the LLM
from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Pull the SQL query prompt template from LangChain hub
from langchain import hub
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
# (Optional) Print the prompt template for debugging:
#query_prompt_template.messages[0].pretty_print()

# Define the functions for each step
tools = [{
    "type": "function",
    "function": {
        "name": "generate_graph",
        "description": "Generate a graph using Seaborn library from the results (dataframe) and the user question.",
        "parameters": {
        "type": "object",
        "properties": {
            "results": {
                "type": "number",
                "description": "The Result from the SQL query to make a graph from."
            },
            "question": {
                "type": "string",
                "description": "The user question to make a graph from."
            }
        },
        "required": ["results", "question"],
        "additionalProperties": False
    },
        "strict": True
    }
}]

def relevance_check(question):
    """Check if the question is relevant to the Chinook database schema."""
    schema_info = db.get_table_info()  # This should return details of the Chinook schema.
    prompt = (
        "You are a helpful assistant specialized in the Chinook database, which is a sample database for a digital media store. "
        "Based on the following Chinook database schema, determine if the user question is relevant to the Chinook database. "
        "Answer only with 'yes' or 'no'.say yes for everything for now unless it is completely unrelated u feel.\n\n"
        f"Question: {question}\n"
        f"Database (digital-media store) Chinook Schema: {schema_info}\n"
    )
    response = llm.invoke(prompt,temperature=0.2)
    return response.content.strip().lower() == "yes"

def write_query(state):
    """Generate a SQL query from the user question."""
    class QueryOutput(TypedDict):
        query: Annotated[str, ..., "Syntactically valid SQL query."]

    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 25,
        "table_info": db.get_table_info(),
        "input": state["question"],
    })
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt, temperature=0.2)
    return {"query": result["query"]}

def execute_query(state):
    """Execute the generated SQL query against the database."""
    from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_graph(state):
    """
    Generates a graph from the SQL result by calling an LLM to produce a Python code snippet for data visualization.
    The LLM returns a JSON with two keys: "plot_code" (the code to generate and save the plot as 'plot.png')
    and "plot_type" (the type of plot generated). This function executes the code and returns only the plot type.
    """
    try:
        # If state["result"] is a string, parse it into a Python object.
        if isinstance(state["result"], str):
            result_data = ast.literal_eval(state["result"])
        else:
            result_data = state["result"]
            
        # Convert the parsed result into a DataFrame with explicit column names.
        df = pd.DataFrame(result_data, columns=["X", "Y"])
        # Convert the SQL result into a DataFrame.
        # df = pd.DataFrame(state["result"])
    except Exception as e:
        st.error(f"Error converting result to DataFrame: {e}")
        st.write("Type:", type(state["result"]))
        st.write("Content:", state["result"])
        return {"plot_type": "Error"}
    
    # Serialize the DataFrame to JSON.
    df_json = df.to_json(orient="records")
    
    # Build the prompt for the LLM.
    graph_prompt = (
    f"User Question: {state['question']}\n"
    f"SQL Query : {state['query']}\n"
    f"Result (in JSON): {df_json}\n\n"
    "Based on the above data and the user question, generate a minimal, clean Python code snippet using the Seaborn/Matplotlib library to create the most appropriate plot for this data. "
    "The code should use the 'veridis' color palette if possible and assume that a pandas DataFrame named 'df' is already defined containing the data. "
    "It must only include the plotting commands (no import statements or comments) and save the plot to a file named 'plot.png'. "
    "Return your answer as a JSON object with exactly two keys: "
    "  - plot_code: the complete Syntactically valid Python code snippet (as a string) to generate and save the plot, "
    "  - plot_type: a string indicating the type of plot (e.g., 'bar', 'line', 'scatter', or any other type). "
    "Do not include any additional text."
    )
    # graph_prompt = (
    #     f"User Question: {state['question']}\n"
    #     f"Data (in JSON): {df_json}\n\n"
    #     "Based on the above data and question, generate valid Python code using Seaborn/Matplotlib data visualization library"
    #     "that creates an appropriate plot from the data. The code should assume that the data is available in a pandas "
    #     "DataFrame named 'df' and must save the generated plot to a file called 'plot.png'.\n"
    #     "You should only output the code snippet and the type of plot generated, in a JSON format.No need for code comments "
    #     "Code should be only of the plotting part and also saving the plot in a file called 'plot.png', not the import statements or anything else.\n"
    #     # "Do not include any comments or explanations in the code.\n"
    #     "Return your answer in a JSON object with the following keys:\n"
    #     "  - plot_code: The complete Python code snippet (as a string) to generate and save the plot.\n"
    #     "  - plot_type: A string indicating the type of plot (e.g., 'bar', 'line', 'scatter', etc.)."
    #     "Do not include any additional text.\n\n"
    # )
    
    # messages = [
    #     {"role": "system", "content": "You are an assistant that writes Python code for data visualization using seaborn data visualization library."},
    #     {"role": "user", "content": graph_prompt}
    # ]
    
    # Call the LLM.
    structured_llm = llm.with_structured_output(GraphOutput)
    result = structured_llm.invoke(graph_prompt, temperature=0.2)
    # completion = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=messages
    # )   

    # output_text = completion.choices[0].message.content
    
    # # Remove any markdown formatting.
    # if output_text.startswith("```"):
    #     output_text = output_text.strip("```").strip()
    
    try:
        # Parse the JSON output.
        plot_code = result["plot_code"]
        plot_type = result["plot_type"]
        st.subheader("Generated Plot Code")
        st.code(plot_code, language="python")
        # st.write("Plot code:", plot_code)
        st.write("Plot type:", plot_type)
    except Exception as e:
        st.error(f"Error parsing JSON output from LLM: {e}")
        return {"plot_type": "Error"}
    
    # Prepare a local namespace with necessary libraries.
    local_vars = {"df": df, "plt": plt, "sns": __import__("seaborn"), "pd": pd}
    
    try:
        # Execute the generated code snippet, which should save the plot as "plot.png".
        exec(plot_code, local_vars)
    except Exception as e:
        st.error(f"Error executing generated plot code: {e}")
        return {"plot_type": "Error"}
    
    # Return only the type of plot generated.
    return {"plot_type": plot_type}

def generate_answer(state):
    """
    Generate a natural language answer from the SQL result.
    Uses function calling to decide if a graph is needed:
      - If a graph is needed, calls the 'generate_graph' tool with the SQL result and user question.
      - Then, makes a final LLM call to get the answer.
      - If no graph is needed, returns the final answer directly.
    """
    # Construct the initial prompt.
    prompt = (
        "Given the following user question, corresponding SQL query, and SQL result, decide if a graph is needed to better answer the question(only if it is needed, not when we can just list the answer simply). "
        "If a graph is appropriate, output a function call to generate_graph with parameters 'results' (the SQL result) and 'question' (the user question). (use appropriate tool)"
        "If a graph is not needed, simply output the final answer in plain text (standard font/sizing)."
        "Do not output any other unwanted texts, just either the tool call or the final output, no reasoning steps or anything\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}\n"
    )
    
    input_messages = [
        {"role": "developer", "content": "You are an assistant that determines if a graph is needed and generates a final answer accordingly."},
        {"role": "user", "content": prompt}
    ]
    
    # Call the LLM with our graph tool definition.
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=input_messages,
    tools=tools,
    temperature=0.2
    )   
    
    message = completion.choices[0]
    
    if message.finish_reason == "tool_calls":
        # LLM decided a graph is needed.
        # tool_call = completion.choices[0].message.tool_calls[0]
        # args = json.loads(tool_call.function.arguments)
        # Call our generate_graph function with the provided parameters.
        graph_response = generate_graph(state)
        state["type_graph_generated"] = graph_response.get("plot_type")
        
        # Now, generate the final answer including the graph info.
        final_prompt = (
            "A graph was generated for the SQL result. Please now provide a final, clear natural language answer "
            "that summarizes the SQL query, its result, and includes reference to the generated graph.Do not include units if not known\n\n"
            f"Question: {state['question']}\n"
            f"SQL Query: {state['query']}\n"
            f"SQL Result: {state['result']}\n"
            f"Type of Graph generated: {graph_response}\n"
        )

        final_messages = [
            {"role": "developer", "content": "You are an assistant that drafts answers based on SQL query results, user question and references the generated graph."},
            {"role": "user", "content": final_prompt}
        ]

        final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=final_messages,
        temperature=0.2,
        )

        final_answer = final_response.choices[0].message.content

        return {"answer": final_answer}
    
    else:
        # LLM decided that no graph is needed, return the answer directly.
        return {"answer": message.message.content}

# def generate_answer(state):
#     """Generate a natural language answer from the SQL result."""
#     prompt = (
#         "Given the following user question, corresponding SQL query, "
#         "and SQL result, perform the following:"
#         "first determine whether a graph is appropriate for answering the question and use approproate tools if it is deemeed to be needed."
#         "If graph is not neededd then simply answer the user question.\n\n"
#         f'Question: {state["question"]}\n'
#         f'SQL Query: {state["query"]}\n'
#         f'SQL Result: {state["result"]}'
#     )
#     response = llm.invoke(prompt)
#     return {"answer": response.content}

# --- Streamlit UI ---

st.title("NL to SQL AI Agent")
st.write("Ask a question about the database and see the generated SQL query, its result (optionally a graph with code), and an answer.")

# User input for the question
question = st.text_input("Enter your question for the database:")

if st.button("Run Query"):
    if not question:
        st.error("Please enter a question before running the query.")
    else:
        # Create a state dictionary to track progress
        state = {"question": question, "query": "", "result": "", "answer": "", "type_graph_generated" : ""}

        # Check if the question is relevant to the database
        relevance = relevance_check(state["question"])
        if not relevance:
            # st.write(db.get_table_info())
            st.warning("The question may not be relevant to the database. Please try another question.")
        else:
            # Generate the SQL query
            with st.spinner("Generating SQL query..."):
                query_dict = write_query(state)
                state["query"] = query_dict["query"]

            st.subheader("Generated SQL Query")
            st.code(state["query"], language="sql")

            # Execute the SQL query
            with st.spinner("Executing SQL query..."):
                result_dict = execute_query(state)
                state["result"] = result_dict["result"]

            st.subheader("SQL Query Result")
            st.write(state["result"])

            # Generate the final answer
            with st.spinner("Generating answer..."):
                answer_dict = generate_answer(state)
                state["answer"] = answer_dict["answer"]

            st.subheader("Answer")
            st.markdown(state["answer"], unsafe_allow_html=True)
            # st.write(state["answer"])

            # If a graph was generated, display it and offer a download.
            if state.get("type_graph_generated") and state["type_graph_generated"] != "Error":
                st.subheader("Generated Plot")
                st.image("plot.png")
                with open("plot.png", "rb") as file:
                    st.download_button("Download Plot", data=file, file_name="plot.png", mime="image/png")
