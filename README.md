# Customer Review Summarizer

A powerful, user-friendly application that leverages Natural Language Processing to automatically generate concise summaries of customer reviews using a fine-tuned BART sequence-to-sequence model.

## ✨ Features

- **Single Review Processing:** Generate summaries for individual customer reviews with a clean, interactive interface
- **Bulk Processing:** Upload CSV or Excel files containing multiple reviews for batch processing
- **Quality Metrics:** Calculate and visualize ROUGE scores when reference summaries are provided
- **Modern UI:** Beautiful, responsive design with interactive elements
- **Download Results:** Export processed summaries as CSV files for further analysis

## 📋 Requirements

- Python 3.7+
- Streamlit
- PyTorch
- Transformers (HuggingFace)
- NLTK
- Pandas
- ROUGE-score

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://huggingface.co/spaces/Prabhat9801/Customer_Review_Summarizer
   cd customer-review-summarizer
   ```
1. Clone this model repository:
   ```bash
   git lfs install  # Required for large model files
   git clone https://huggingface.co/Prabhat9801/model

   ```

3. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

Create a `requirements.txt` file with the following contents:

```
streamlit>=1.24.0
torch>=1.12.0
transformers>=4.30.0
nltk>=3.7
pandas>=1.5.0
openpyxl>=3.0.10
rouge-score>=0.1.2
scikit-learn>=1.0.2
protobuf>=3.20.0,<4.0.0

```

## 🏃‍♂️ Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501` in your web browser.

## 💻 Usage

### Single Review Processing

1. Select the "Single Review" tab
2. Enter your product review in the text area
3. Optionally, add a reference summary for comparison
4. Click "Generate Summary"
5. View the generated summary and quality metrics (if reference provided)

### Bulk Processing

1. Select the "Bulk Processing" tab
2. Prepare a CSV or Excel file with a column named 'review' (required)
   - Optionally include a column named 'reference_summary' for quality comparison
3. Upload your file using the file uploader
4. Click "Process Bulk Reviews"
5. View results in the table and download processed data as CSV

## 🧠 Model Information

The application uses a fine-tuned sequence-to-sequence model from HuggingFace:
- Primary model: `Prabhat9801/model` (BART-based fine-tuned for review summarization)
- Fallback model: `facebook/bart-large-cnn` (if the primary model fails to load)

## 📊 Evaluation Metrics

The application uses ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics to evaluate summary quality:

- **ROUGE-1:** Unigram overlap between generated and reference summaries
- **ROUGE-2:** Bigram overlap between generated and reference summaries
- **ROUGE-L:** Longest Common Subsequence between generated and reference summaries

## 🔧 Customization

You can modify the following parameters in the code:
- `MAX_INPUT_LENGTH`: Maximum length of input text (default: 384 tokens)
- `MAX_TARGET_LENGTH`: Maximum length of generated summaries (default: 48 tokens)
- Beam search parameters for summary generation

## 🛠️ Troubleshooting

- **NLTK Download Issues:** The app will create a local `nltk_data` directory to store required NLTK data
- **Model Loading Failures:** The app will automatically fall back to a standard BART model if the custom model fails to load
- **File Format Issues:** The app supports CSV and Excel formats - if Excel reading fails, it attempts to read as CSV

## 📝 License

[MIT](LICENSE)

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the web application framework
- [HuggingFace](https://huggingface.co/) for model hosting and transformers library
- [PyTorch](https://pytorch.org/) for the deep learning framework

---

## 📬 Contact

For questions or support contact [prabhatkumarsictc12@gmail.com](prabhatkumarsictc12@gmail.com).
