import streamlit as st
from typing import Generator
from groq import Groq
import os
import logging
import json
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set Streamlit page configuration
st.set_page_config(page_icon="💬", layout="wide", page_title="Groq Goes Brrrrrrrr...")

CHAT_HISTORY_DIR = "chat_history"
CHAT_HISTORY_FILE = os.path.join(CHAT_HISTORY_DIR, "chat_history.json")

def icon(emoji: str):
  """Shows an emoji as a Notion-style page icon."""
  st.write(
      f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
      unsafe_allow_html=True,
  )

def initialize_session_state():
  """Initialize session state variables."""
  if "messages" not in st.session_state:
      st.session_state.messages = []
  if "selected_model" not in st.session_state:
      st.session_state.selected_model = "llama3-70b-8192"  # Default to LLaMA3-70b-8192
  if "system_prompt" not in st.session_state:
      st.session_state.system_prompt = ""
  if "temperature" not in st.session_state:
      st.session_state.temperature = 0.7  # Default temperature
  if "top_p" not in st.session_state:
      st.session_state.top_p = 0.9  # Default top_p
  if "top_k" not in st.session_state:
      st.session_state.top_k = 50  # Default top_k
  if "chat_history" not in st.session_state:
      st.session_state.chat_history = load_chat_history()
  if "round_trip_time" not in st.session_state:
      st.session_state.round_trip_time = None
  if "metrics" not in st.session_state:
      st.session_state.metrics = {}

def get_groq_client() -> Groq:
  """Initialize and return the Groq client."""
  api_key = os.getenv("GROQ_API_KEY")
  if not api_key:
      st.error("API key not found. Please set the GROQ_API_KEY environment variable.")
      st.stop()
  return Groq(api_key=api_key)

def display_chat_messages():
  """Display chat messages from the session state."""
  for message in st.session_state.messages:
      avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
      with st.chat_message(message["role"], avatar=avatar):
          st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
  """Yield chat response content from the Groq API response."""
  for chunk in chat_completion:
      if chunk.choices[0].delta.content:
          yield chunk.choices[0].delta.content

def save_chat_history():
  """Save the current chat history to a file."""
  if not os.path.exists(CHAT_HISTORY_DIR):
      os.makedirs(CHAT_HISTORY_DIR)
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  model_name = st.session_state.selected_model
  filename = os.path.join(CHAT_HISTORY_DIR, f"chat_{timestamp}_{model_name}.json")
  with open(filename, "w") as f:
      json.dump(st.session_state.messages, f)
  st.session_state.chat_history[filename] = f"{timestamp} ({model_name})"
  with open(CHAT_HISTORY_FILE, "w") as f:
      json.dump(st.session_state.chat_history, f)

def load_chat_history():
  """Load chat history from a file."""
  if os.path.exists(CHAT_HISTORY_FILE):
      with open(CHAT_HISTORY_FILE, "r") as f:
          return json.load(f)
  return {}

def load_chat(filename):
  """Load chat messages from a file."""
  with open(filename, "r") as f:
      st.session_state.messages = json.load(f)

def main():
  """Main function to run the Streamlit app."""
  icon(" ")
  st.subheader("Groq Chat", divider="gray", anchor=False)

  client = get_groq_client()
  initialize_session_state()

  # Define model details
  models = {
      "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
      "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
      "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
      "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
      "llama-3.1-70b-versatile": {"name": "Llama 3.1 70B (Preview)", "tokens": 8000, "developer": "Meta"},
      "llama-3.1-8b-instant": {"name": "Llama 3.1 8B (Preview)", "tokens": 8000, "developer": "Meta"},
      "llama3-groq-70b-8192-tool-use-preview": {"name": "Llama 3 Groq 70B Tool Use (Preview)", "tokens": 8192, "developer": "Groq"},
      "llama3-groq-8b-8192-tool-use-preview": {"name": "Llama 3 Groq 8B Tool Use (Preview)", "tokens": 8192, "developer": "Groq"},
  }

  # Add an expander for advanced model settings
  with st.expander("Advanced Settings"):
      model_option = st.selectbox(
          "Choose a model:",
          options=list(models.keys()),
          format_func=lambda x: models[x]["name"],
          index=1  # Default to LLaMA3-70b-8192
      )

      # Detect model change and clear chat history if model has changed
      if st.session_state.selected_model != model_option:
          st.session_state.messages = []
          st.session_state.selected_model = model_option

      # Add a text input for the system prompt
      st.text_input(
          "System Prompt (optional):",
          value=st.session_state.system_prompt,
          on_change=lambda: st.session_state.update({"system_prompt": st.session_state.system_prompt}),
          key="system_prompt",
          help="Enter a system prompt to guide the model's behavior."
      )

      max_tokens_range = models[model_option]["tokens"]

      # Adjust max_tokens slider dynamically based on the selected model
      max_tokens = st.slider(
          "Max Tokens:",
          min_value=512,  # Minimum value to allow some flexibility
          max_value=max_tokens_range,
          # Default value or max allowed if less
          value=min(8192, max_tokens_range),
          step=512,
          help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
      )

      st.session_state.temperature = st.slider(
          "Temperature:",
          min_value=0.0,
          max_value=1.0,
          value=st.session_state.temperature,
          step=0.1,
          help="Controls the randomness of the model's output. Lower values make the output more deterministic."
      )

      st.session_state.top_p = st.slider(
          "Top P:",
          min_value=0.0,
          max_value=1.0,
          value=st.session_state.top_p,
          step=0.1,
          help="Controls the diversity of the model's output. Lower values make the output more focused."
      )

  # Add a button to clear the chat
  if st.button("Clear Chat"):
      st.session_state.messages = []

  # Add a button to save the chat
  if st.button("Save Chat"):
      save_chat_history()

  # Display chat messages from history on app rerun
  display_chat_messages()

  if prompt := st.chat_input("Enter your prompt here..."):
      st.session_state.messages.append({"role": "user", "content": prompt})

      with st.chat_message("user", avatar='👨‍💻'):
          st.markdown(prompt)

      # Prepare messages for the API call
      messages = []
      if st.session_state.system_prompt:
          messages.append({"role": "system", "content": st.session_state.system_prompt})
      messages.extend(st.session_state.messages)

      # Fetch response from Groq API
      try:
          start_time = datetime.now()
          chat_completion = client.chat.completions.create(
              model=model_option,
              messages=messages,
              max_tokens=max_tokens,
              temperature=st.session_state.temperature,
              top_p=st.session_state.top_p,
              stream=True
          )

          # Use the generator function with st.write_stream
          with st.chat_message("assistant", avatar="🤖"):
              chat_responses_generator = generate_chat_responses(chat_completion)
              full_response = st.write_stream(chat_responses_generator)
          end_time = datetime.now()
          rtt = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds

          # Calculate metrics
          input_tokens = len(prompt.split())
          output_tokens = len(full_response.split())
          total_tokens = input_tokens + output_tokens
          inference_time = (end_time - start_time).total_seconds()  # Calculate inference time manually
          input_speed = input_tokens / inference_time
          output_speed = output_tokens / inference_time
          total_speed = total_tokens / inference_time

          # Display metrics in a visually appealing way
          with st.expander("Metrics"):
              st.markdown(
                  f"""
                  <div style="background-color: ##121418; padding: 10px; border-radius: 5px; color: white;">
                      <div style="display: flex; justify-content: space-between; align-items: center;">
                          <div>
                              <strong>Round trip time:</strong> {rtt:.2f} ms
                          </div>
                          <div>
                              <strong>Model:</strong> {model_option}
                          </div>
                      </div>
                      <table style="width: 100%; margin-top: 10px;">
                          <thead>
                              <tr>
                                  <th style="text-align: left;">&nbsp;</th>
                                  <th style="text-align: left;">Input</th>
                                  <th style="text-align: left;">Output</th>
                                  <th style="text-align: left;">Total</th>
                              </tr>
                          </thead>
                          <tbody>
                              <tr>
                                  <td style="text-align: left;"><strong>Speed (T/s)</strong></td>
                                  <td style="text-align: left;">{input_speed:.2f}</td>
                                  <td style="text-align: left;">{output_speed:.2f}</td>
                                  <td style="text-align: left;">{total_speed:.2f}</td>
                              </tr>
                              <tr>
                                  <td style="text-align: left;"><strong>Tokens</strong></td>
                                  <td style="text-align: left;">{input_tokens}</td>
                                  <td style="text-align: left;">{output_tokens}</td>
                                  <td style="text-align: left;">{total_tokens}</td>
                              </tr>
                              <tr>
                                  <td style="text-align: left;"><strong>Inference Time (s)</strong></td>
                                  <td style="text-align: left;">{inference_time:.2f}</td>
                                  <td style="text-align: left;">{inference_time:.2f}</td>
                                  <td style="text-align: left;">{inference_time:.2f}</td>
                              </tr>
                          </tbody>
                      </table>
                  </div>
                  """,
                  unsafe_allow_html=True
              )

      except Exception as e:
          st.error(f"An error occurred: {e}", icon="🚨")
          logging.error(f"Error fetching response: {e}")
          full_response = ""  # Initialize full_response to an empty string in case of error

      # Append the full response to session_state.messages
      if isinstance(full_response, str):
          st.session_state.messages.append(
              {"role": "assistant", "content": full_response})
      else:
          # Handle the case where full_response is not a string
          combined_response = "\n".join(str(item) for item in full_response)
          st.session_state.messages.append(
              {"role": "assistant", "content": combined_response})

  # Sidebar for chat history
  st.sidebar.header("Chat History")
  for filename, timestamp in st.session_state.chat_history.items():
      if st.sidebar.button(timestamp):
          load_chat(filename)
          st.rerun()  # Replace st.experimental_rerun with st.rerun

if __name__ == "__main__":
  main()