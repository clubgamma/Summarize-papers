# Summarized Paper Project

![Hacktoberfest Badge](https://img.shields.io/badge/Hacktoberfest-2024-blueviolet)
![Open Source](https://img.shields.io/badge/Open%20Source-Contributions%20Welcome-brightgreen)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green)

## Project Overview 🌍

The **Summarized Paper Project** aims to utilize machine learning techniques to condense lengthy academic papers into concise summaries. This project addresses the challenge of information overload in academic research, making it easier for researchers and students to grasp essential concepts quickly.

### Key Features:

- **Text Processing**: Cleaning and pre-processing text data from academic papers.
- **Summarization Models**: Implementing models such as BERT, GPT, and other NLP techniques to generate summaries.
- **User Interface**: A simple web interface for users to input text and receive summarized output.
- **Evaluation**: Measuring summary quality through ROUGE scores and user feedback.

The project aims to facilitate faster comprehension of academic research by providing automated summarization tools.


## Project Structure 📁

```bash
.

├── data/                 # Datasets for academic papers
├── notebooks/ 
│   ├── summarization of a long text.ipynb #jupyter notebook for a long text which can be summarized by all pretrained models
│   ├── summarize-papers.ipynb  # a single notebook for summarization of academic papers 
│   ├── summarization.ipynb     # notebook with main function stored inside src/summarization_utils.py
├── src/                  # Source code for the project
│   ├── summarization_utils.py # Python file which contains all the function
│   ├── data_prosessing.py           
│   ├── model_training.py
├── README.md             # Project documentation
└── CONTRIBUTING.md       # Contribution guidelines
```

## Getting Started 🤗🚀

To contribute to the project, follow these steps:

### Fork the Repository:

Click the fork button at the top right of the repository page.

### Clone Your Fork:

Clone the forked repository to your local machine.

```bash
git clone https://github.com/your-username/summarized-paper-project.git
```

### Navigate to the Project Directory:

```bash
cd summarized-paper-project
```

### Create a New Branch:

```bash
git checkout -b my-new-branch
```

### Make Your Changes:

Add your features or improvements to the project.

### Commit Your Changes:

```bash
git add .
git commit -m "Add relevant message here"
```

### Push to Your Branch:

```bash
git push origin my-new-branch
```

### Create a Pull Request:

Go to your forked repository on GitHub and create a pull request to the main repository.

---

## Contribution Guidelines 📚

We welcome your contributions! Please follow these guidelines:

- **Creativity Allowed**: Submit pull requests even if they break the rules—we may merge them anyway!
- **Preserve Existing Content**: Don’t remove existing content.
- **Code Style**: Your code can be neat, messy, or complex—as long as it works!
- **Add Your Name**: Remember to add your name to the `contributorsList` file.
- **Keep it Small**: Small pull requests help prevent merge conflicts.

---

## Avoiding Conflicts (Syncing Your Fork) 🔄

To keep your fork up to date with the main repository:

### Add Upstream Remote:

```bash
git remote add upstream https://github.com/clubgamma/summarized-paper-project.git
```

### Verify the New Remote:

```bash
git remote -v
```

### Sync Your Fork with Upstream:

```bash
git fetch upstream
git merge upstream/main
```

---

## Add Your Name 🌟

Please add your name to the `CONTRIBUTING.md` file using the following format:

#### Name: [YOUR NAME](GitHub link)
- Place: City, State, Country
- Bio: Who are you?
- GitHub: [GitHub account name](GitHub link)

---

## Pull Request Labels

We have the following PR labels:

- `level 1` - Basic level contributions
- `level 2` - Intermediate level contributions
- `level 3` - Advanced level contributions
- `hacktoberfest-accepted` - Contributions accepted for Hacktoberfest

---

We look forward to your contributions! 🎉
