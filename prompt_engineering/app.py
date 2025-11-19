import streamlit as st
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Week 1: Prompt Engineering & LLM Fundamentals",
    layout="wide"
)

@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found in .env file")
        return None
    return OpenAI(api_key=api_key)

client = get_openai_client()

st.title("Week 1: Prompt Engineering & LLM Fundamentals")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prompt Design",
    "🌡️ LLM Parameters",
    "🔧 Function Calling",
    "📊 Comparison"
])

with tab1:
    st.header("🎯 Prompt Design")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        prompt_type = st.selectbox(
            "Prompt Type",
            ["Zero-Shot", "Few-Shot", "Role-Based"]
        )
        
        if prompt_type == "Zero-Shot":
            user_question = st.text_area(
                "Question:",
                "Classify the sentiment of: 'I love this product!'",
                height=100
            )
            system_prompt = "You are a helpful AI assistant."
            
        elif prompt_type == "Few-Shot":
            examples = st.text_area(
                "Examples:",
                """Example 1:
Input: "This movie was terrible"
Output: Negative

Example 2:
Input: "Best purchase ever!"
Output: Positive""",
                height=150
            )
            user_question = st.text_input(
                "Question:",
                "Classify: 'The service was okay'"
            )
            system_prompt = "You are a sentiment classifier. Follow the examples provided."
            
        else:  # Role-Based
            role = st.selectbox(
                "Role:",
                ["Expert Programmer", "Creative Writer", "Data Analyst", "Teacher", "Custom"]
            )
            
            if role == "Custom":
                role_description = st.text_input("Role description:", "You are a...")
            else:
                role_map = {
                    "Expert Programmer": "You are an expert programmer with 20 years of experience. Provide detailed, well-commented code solutions.",
                    "Creative Writer": "You are a creative writer specializing in storytelling. Use vivid descriptions and engaging narratives.",
                    "Data Analyst": "You are a data analyst expert. Provide insights with clear explanations and suggest visualizations.",
                    "Teacher": "You are a patient teacher who explains concepts clearly with examples and analogies."
                }
                role_description = role_map[role]
            
            system_prompt = role_description
            user_question = st.text_area(
                "Question:",
                "Explain recursion in programming.",
                height=100
            )
    
    with col2:
        st.subheader("Response")
        
        if st.button("Generate", key="prompt_design"):
            if not client:
                st.error("API key not loaded")
            else:
                with st.spinner("Generating..."):
                    try:
                        messages = [{"role": "system", "content": system_prompt}]
                        
                        if prompt_type == "Few-Shot" and examples:
                            messages.append({"role": "user", "content": examples})
                        
                        messages.append({"role": "user", "content": user_question})
                        
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            temperature=0.7,
                            max_tokens=500
                        )
                        
                        st.write(response.choices[0].message.content)
                        st.caption(f"Tokens: {response.usage.total_tokens}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

with tab2:
    st.header("🌡️ LLM Parameters")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Controls")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=1000,
            value=300,
            step=50
        )
        
        model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
        )
        
        st.divider()
        
        test_prompt = st.text_area(
            "Prompt:",
            "Write a short creative story about a robot learning to paint.",
            height=100
        )
    
    with col2:
        st.subheader("Response")
        
        if st.button("Generate", key="params"):
            if not client:
                st.error("API key not loaded")
            else:
                with st.spinner("Generating..."):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user", "content": test_prompt}
                            ],
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens
                        )
                        
                        st.write(response.choices[0].message.content)
                        
                        usage = response.usage
                        st.caption(f"Tokens - Prompt: {usage.prompt_tokens} | Completion: {usage.completion_tokens} | Total: {usage.total_tokens}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

with tab3:
    st.header("🔧 Function Calling")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Functions")
        
        def get_weather(location: str, unit: str = "celsius") -> dict:
            """Simulated weather function"""
            return {
                "location": location,
                "temperature": 22 if unit == "celsius" else 72,
                "unit": unit,
                "condition": "sunny"
            }
        
        def calculate_mortgage(principal: float, interest_rate: float, years: int) -> dict:
            """Calculate monthly mortgage payment"""
            monthly_rate = interest_rate / 100 / 12
            num_payments = years * 12
            monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                            ((1 + monthly_rate)**num_payments - 1)
            
            return {
                "monthly_payment": round(monthly_payment, 2),
                "total_payment": round(monthly_payment * num_payments, 2),
                "total_interest": round((monthly_payment * num_payments) - principal, 2)
            }
        
        # Function definitions for OpenAI
        functions = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calculate_mortgage",
                "description": "Calculate monthly mortgage payment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "principal": {
                            "type": "number",
                            "description": "Loan amount in dollars"
                        },
                        "interest_rate": {
                            "type": "number",
                            "description": "Annual interest rate (e.g., 5.5 for 5.5%)"
                        },
                        "years": {
                            "type": "integer",
                            "description": "Loan term in years"
                        }
                    },
                    "required": ["principal", "interest_rate", "years"]
                }
            }
        ]
        
        for func in functions:
            with st.expander(f"{func['name']}"):
                st.json(func)
        
        st.divider()
        
        user_query = st.text_area(
            "Query:",
            "What's the weather like in Tokyo? Also, calculate the monthly payment for a $300,000 mortgage at 6.5% interest for 30 years.",
            height=100
        )
    
    with col2:
        st.subheader("Result")
        
        if st.button("Execute", key="function_call"):
            if not client:
                st.error("API key not loaded")
            else:
                with st.spinner("Processing..."):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": user_query}],
                            functions=functions,
                            function_call="auto"
                        )
                        
                        message = response.choices[0].message
                        
                        if message.function_call:
                            function_name = message.function_call.name
                            function_args = json.loads(message.function_call.arguments)
                            
                            st.markdown(f"**Called:** `{function_name}`")
                            st.json(function_args)
                            
                            if function_name == "get_weather":
                                function_response = get_weather(**function_args)
                            elif function_name == "calculate_mortgage":
                                function_response = calculate_mortgage(**function_args)
                            
                            st.markdown("**Result:**")
                            st.json(function_response)
                            
                            second_response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "user", "content": user_query},
                                    message,
                                    {
                                        "role": "function",
                                        "name": function_name,
                                        "content": json.dumps(function_response)
                                    }
                                ]
                            )
                            
                            st.markdown("**Response:**")
                            st.write(second_response.choices[0].message.content)
                        else:
                            st.write(message.content)
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

with tab4:
    st.header("📊 Comparison")
    
    comparison_prompt = st.text_area(
        "Prompt:",
        "Write a tagline for a tech startup that makes AI-powered gardening tools.",
        height=100
    )
    
    if st.button("Generate Comparisons", key="comparison"):
        if not client:
            st.error("API key not loaded")
        else:
            col1, col2, col3 = st.columns(3)
            
            configs = [
                {"name": "Low Temp (0.2)", "temp": 0.2, "col": col1},
                {"name": "Medium Temp (0.7)", "temp": 0.7, "col": col2},
                {"name": "High Temp (1.5)", "temp": 1.5, "col": col3}
            ]
            
            for config in configs:
                with config["col"]:
                    st.subheader(config["name"])
                    with st.spinner("Generating..."):
                        try:
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": comparison_prompt}],
                                temperature=config["temp"],
                                max_tokens=150
                            )
                            st.write(response.choices[0].message.content)
                            st.caption(f"Tokens: {response.usage.total_tokens}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
