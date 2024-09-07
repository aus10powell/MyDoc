# MyDoc

Test out OpenBioLLM Medical LLM question here: https://huggingface.co/spaces/aus10powell/MyDoc

This repository provides tools and scripts for converting PaliGemma model checkpoints to HuggingFace format and integrating with Streamlit for displaying LangChain agent thoughts.

## Features

- **Model Conversion**: Convert PaliGemma model checkpoints from the original repository to HuggingFace format using the script in [`venv/lib/python3.10/site-packages/transformers/models/paligemma/convert_paligemma_weights_to_hf.py`](venv/lib/python3.10/site-packages/transformers/models/paligemma/convert_paligemma_weights_to_hf.py).
- **Streamlit Integration**: Display LangChain agent thoughts in Streamlit using the callback handler in [`venv/lib/python3.10/site-packages/streamlit/external/langchain/streamlit_callback_handler.py`](venv/lib/python3.10/site-packages/streamlit/external/langchain/streamlit_callback_handler.py).

## Usage

### Model Conversion

To convert a PaliGemma model checkpoint, run the following command: