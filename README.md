# 📄 Resume Parser Pro

An **AI-powered Resume Information Extraction Tool** built using **Streamlit**, **PyMuPDF**, and **python-docx**.  
It automatically extracts **Name**, **Email**, **Phone**, **Skills**, **Education**, and **Experience** from PDF/DOCX resumes.

---

## 🚀 Features

- 📂 **Upload multiple resumes** (PDF/DOCX)
- 🔍 **Extract:**
  - 👤 Candidate Name
  - 📧 Email
  - 📱 Phone Number
  - 🎯 Skills
  - 🎓 Education Details
  - 💼 Experience Section
- 📊 **Export parsed data** in:
  - JSON format
  - CSV format
- 📈 **Summary statistics** (files processed, names found, avg. skills)
- 🔐 **Works locally** (no data sent to cloud APIs)

---

## 🧠 How It Works

1. **Text Extraction**
   - PDF → Extracted using `PyMuPDF (fitz)`
   - DOCX → Extracted using `python-docx`
2. **Regex-Based Parsing**
   - Email, phone, name identified using regex and text heuristics.
3. **Keyword Matching**
   - Skills extracted from a pre-defined keyword dictionary.
4. **Section Detection**
   - Education and experience parsed based on section headers and pattern rules.
5. **Results Display**
   - Results shown in Streamlit dashboard with export options.

---

## 🧰 Tech Stack

| Component | Library |
|------------|----------|
| Frontend UI | Streamlit |
| PDF Parsing | PyMuPDF (`fitz`) |
| DOCX Parsing | python-docx |
| Data Handling | Pandas |
| Regex Matching | re |
| Export | JSON, CSV |

---

## ⚙️ Installation

1. Clone or download this repository:
   ```bash
   git clone https://github.com/yourusername/resume-parser-pro.git
   cd resume-parser-pro
2.Install dependencies:
  pip install streamlit pymupdf python-docx pandas
3.Run the app:
  streamlit run app.py
