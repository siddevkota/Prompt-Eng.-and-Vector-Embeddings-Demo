# Week 1: Prompt Engineering & LLM Fundamentals Demo

An interactive Streamlit application demonstrating key concepts from Week 1 of LLM learning, including prompt engineering techniques, LLM parameter tuning, and function calling with the OpenAI API.

## 🎯 Features

### 1. **Prompt Design Techniques**
- **Zero-Shot Prompting**: Direct questions without examples
- **Few-Shot Prompting**: Learn from provided examples
- **Role-Based Prompting**: Assign specific roles to the AI (programmer, writer, analyst, etc.)

### 2. **LLM Parameters Exploration**
- **Temperature**: Control creativity and randomness (0.0 - 2.0)
- **Top P**: Nucleus sampling for response diversity
- **Max Tokens**: Control response length
- Real-time parameter adjustment with visual feedback

### 3. **Function/Tool Calling**
- Demonstrates how LLMs can call external functions
- Example functions: Weather lookup, Mortgage calculator
- Shows the complete function calling workflow

### 4. **Side-by-Side Comparison**
- Compare responses with different temperature settings
- Visualize how parameters affect output quality and creativity

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Navigate to the project folder:**
   ```bash
   cd week1_prompt_engineering_demo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key:**
   
   **Option A: Using environment variable (recommended)**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your actual API key
   # OPENAI_API_KEY=sk-your-actual-key-here
   ```
   
   **Option B: Enter in the app**
   - You can also enter your API key directly in the Streamlit sidebar when running the app

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to that URL

## 📚 How to Use

### Tab 1: Prompt Design
1. Select a prompt type (Zero-Shot, Few-Shot, or Role-Based)
2. Enter your question or customize the examples
3. Click "Generate Response" to see results
4. Observe how different prompt techniques affect the output

### Tab 2: LLM Parameters
1. Adjust the sliders to change Temperature, Top P, and Max Tokens
2. Enter a creative prompt
3. Generate responses and see how parameters affect creativity and focus
4. Try extreme values (temp=0 vs temp=2) to see dramatic differences

### Tab 3: Function Calling
1. Review the available functions (weather, mortgage calculator)
2. Ask a question that requires function execution
3. Watch the LLM decide which function to call and with what parameters
4. See the function result and final natural language response

### Tab 4: Comparison
1. Enter a prompt to test
2. Generate three responses simultaneously with different temperatures
3. Compare how creativity and consistency vary

## 🎓 Learning Outcomes

After using this demo, you'll understand:

- ✅ How to design effective prompts for different scenarios
- ✅ The impact of temperature and other parameters on LLM behavior
- ✅ When to use deterministic vs creative settings
- ✅ How function calling enables LLMs to use external tools
- ✅ Best practices for interacting with the OpenAI API

## 🔧 Technical Details

- **Framework**: Streamlit
- **API**: OpenAI GPT-3.5-turbo / GPT-4
- **Language**: Python 3.8+
- **Architecture**: Single-page interactive web application

## 💡 Tips

1. **Start with low temperature (0.2-0.3)** for factual tasks
2. **Use higher temperature (0.8-1.2)** for creative writing
3. **Few-shot prompting** works best when examples are clear and consistent
4. **Role-based prompting** helps maintain consistent tone and expertise
5. **Function calling** is ideal for structured data retrieval and calculations

## 📖 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🛠️ Troubleshooting

**Issue: "Please set OPENAI_API_KEY environment variable"**
- Solution: Enter your API key in the sidebar or set it in the .env file

**Issue: "Rate limit exceeded"**
- Solution: Wait a moment between requests or upgrade your OpenAI plan

**Issue: "Module not found"**
- Solution: Make sure you've installed all requirements: `pip install -r requirements.txt`

## 🎨 Customization

Feel free to:
- Add your own function definitions in Tab 3
- Modify the example prompts
- Add more role types in the role-based prompting section
- Experiment with different OpenAI models (gpt-4, gpt-4-turbo, etc.)

## 📝 License

This is an educational demo project. Feel free to use and modify for learning purposes.

---

**Happy Learning! 🚀**
