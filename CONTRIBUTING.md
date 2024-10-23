# Contributing to AQI Prediction Project

We are excited that you're interested in contributing to our AQI Prediction project! To ensure a smooth and collaborative experience, please read the following guidelines before submitting your contributions.

## How to Contribute 🛠️

### 1. Fork the Repository

Start by forking the repository to your GitHub account.

- Click the **Fork** button at the top right of this repository.

### 2. Clone Your Fork

Clone the forked repository to your local machine.

```bash
git clone https://github.com/your-username/hacktoberfest2024.git
```

### 3. Navigate to the Project Directory

Change your current directory to the cloned repository.

```bash
cd hacktoberfest2024
```

### 4. Create a New Branch

To make changes, create a new branch where you will work on your contribution.

```bash
git checkout -b my-new-branch
```

### 5. Make Your Changes

- Implement your feature, fix bugs, or make improvements.
- Ensure your changes do not remove or alter existing features unless necessary.
- Follow the project structure and maintain coding conventions.

### 6. Commit Your Changes

Once you’ve made your changes, stage them for the next commit and then commit with a clear and concise message.

```bash
git add .
git commit -m "Brief description of the changes made"
```

### 7. Push to Your Fork

Push your local changes to your fork on GitHub.

```bash
git push origin my-new-branch
```

### 8. Create a Pull Request (PR)

Once your changes are pushed, go to your fork on GitHub and create a pull request to the main repository.

- Provide a clear title and description of your changes in the pull request.
- Reference any related issue numbers, if applicable.

---

## Code of Conduct 📝

- Be respectful of other contributors.
- Please preserve the existing codebase and do not remove others' work unless required.
- Ensure that your code works and passes any tests that are in place.
- Avoid breaking changes—if unavoidable, document them clearly.
- Stick to the project style guidelines (indentation, naming conventions, etc.).
- Comment your code where necessary to help others understand your logic.

---

## Syncing Your Fork to Avoid Conflicts 🔄

To stay up-to-date with the main repository and avoid merge conflicts, sync your fork with the upstream repository.

### 1. Add Upstream Remote

```bash
git remote add upstream https://github.com/clubgamma/hacktoberfest2024.git
```

### 2. Fetch Upstream Changes

Fetch the latest changes from the upstream repository.

```bash
git fetch upstream
```

### 3. Merge Upstream Changes into Your Branch

```bash
git merge upstream/main
```

### 4. Resolve Conflicts

If you encounter any merge conflicts, resolve them before pushing your changes.

### 5. Push to Your Branch

```bash
git push origin my-new-branch
```

---

## Adding Your Name to the Contributor List ✨

Once your pull request is merged, feel free to add your name to the `contributorsList.md` file in the following format:

```markdown
#### Name: [Smit Gandhi](https://github.com/smitngandhi)
- Place: City:- Ahmedabad, State:- Gujarat, Country:- India
- Bio: I'm Smit Gandhi, a 19-year-old passionate about machine learning and enthusiastic about reading books.
- GitHub: [smitngandhi](https://github.com/smitngandhi)
```

---

## Pull Request Labels 🏷️

We use the following labels to categorize PRs:

- **level 1**: Basic level contributions
- **level 2**: Intermediate level contributions
- **level 3**: Advanced level contributions
- **hacktoberfest-accepted**: PRs accepted for Hacktoberfest

---

## Contact

If you have any questions or need help, feel free to open an issue or reach out to the project maintainers.

We look forward to your contributions! 🚀
```

This `CONTRIBUTING.md` file provides clear and simple steps on how contributors can start contributing, sync their fork, and add their names to the contributors' list. It also includes guidelines on code conduct and how to create effective pull requests.