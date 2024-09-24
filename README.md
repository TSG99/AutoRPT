
---
# AutoRPT: Automatic Rapid Prosody Transcription Tool

**AutoRPT** is a tool designed to automatically annotate prosodic features following the Rapid Prosody Transcription (RPT) protocol. It is currently trained on Standard American English (SAE), with future updates planned to include other language varieties.

### About the Project

This project is being developed by a team of undergraduate and graduate students, led by **PI Associate Professor Jonathan Howell** at **Montclair State University**. It is produced in conjunction with research funded by **NSF grant 2316030**, focusing on identifying the prosodic features of **“Three Varieties of English in NJ”**. The tool is designed to streamline the annotation of prosodic events using **Rapid Prosodic Transcription (RPT)**, as outlined by **Cole et al. (2017)**.

### Why We Built This Tool

1. **Limited Corpora for Specific Varieties**: Few corpora (with the exception of **CORAAL**) include **African American English (AAE)** and **Latinae English (LE)**.
2. **Lack of Prosodic Annotations**: Even fewer corpora provide prosodic annotations for these varieties of English.
3. **Incomplete Annotation Schemes**: Current annotation schemes often do not account for the unique prosodic features of AAE and LE.
4. **Challenges in Crowdsourcing**: Annotating prosody through crowdsourcing methods can be difficult and inconsistent.

### Corpus and Training

AutoRPT is currently trained on the **Boston University Radio Corpus**, which serves as the foundation for the tool’s prosodic annotations. As research progresses, the model will be adapted to annotate prosodic features in other varieties of English, including those spoken in New Jersey.

### Prosodic Event Annotation and Detection in Three Varieties of English

AutoRPT is part of ongoing research into the detection of prosodic events across the following varieties:

- **Mainstream American English (MAE)**
- **African American English (AAE)**
- **Latinae English (LE)** (as spoken in New Jersey)

---

## Installation Instructions

To run AutoRPT, you'll need to install several Python libraries. Follow the steps below to set up the tool on your system.

### Prerequisites

1. Ensure that you have Python version 3.7 or higher. You can download the latest version of Python [here](https://www.python.org/downloads/).
2. It is recommended to create a virtual environment to manage the dependencies specific to AutoRPT.

### Step 1: Create a Virtual Environment (Optional but Recommended)

Setting up a virtual environment ensures that package installations for AutoRPT do not interfere with other Python projects on your machine.

#### For Windows:
```bash
python -m venv AutoRPT
AutoRPT\Scripts\activate
```

#### For macOS/Linux:
```bash
python3 -m venv AutoRPT
source AutoRPT/bin/activate
```

### Step 2: Install Dependencies

Once your virtual environment is set up and activated, you can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

This command will install all the necessary Python packages listed in the `requirements.txt` file.

### Required Python Packages

The key dependencies for AutoRPT are:

1. **Praat-ParselMouth**: A Python interface to Praat for conducting phonetic analyses.
2. **TextGrid**: A library used to handle Praat TextGrid objects for annotating speech.
3. **Scikit-learn**: A widely-used library for machine learning tasks such as classification and regression.
4. **Pandas**: A powerful data manipulation and analysis library.
5. **PyTorch**: An open-source deep learning framework, used for building and training machine learning models.

### Step 3: Run AutoRPT

Once the dependencies are installed, you can run AutoRPT with the following command:

```bash
python AutoRPT.py
```

AutoRPT will then start processing and annotating prosodic features based on the input data.

---
