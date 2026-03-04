---
description: 'Generate clear, structured release notes from commits between two version tags.'
tools: []
---

# Purpose
Create release notes for a new version by summarizing commits between two tags.

# When to use
Use this before publishing a release.

# Required inputs
* Previous and current release tags (example: `v0.8.0`, `v0.8.1`)
* Commit history for that range (example: `git log v0.8.0...v0.8.1`)

# Expected output
* One markdown release note document that follows the template below.

# Steps
1. Collect all commits between the two tags.
2. Group changes by topic: Features, Bug-fixes, Tests, Docs, Build/Compatibility, etc.
3. Write concise bullet points and merge related changes.
4. Add contributors with profile links.
5. Fill the full changelog table with commit percentages.
6. Add the full commit list command/output section.

# Rules
- Only include changes present in the selected commit range.
- Do not include unreleased or planned work.
- Keep the template structure unless the user asks to change it.
- Keep wording clear, short, and user-friendly.
- Follow semantic versioning standards.

# Release Notes Template

## Example

```markdown
# Acherus vX.Y.Z

This release introduces ...: [short summary of highlights].

## 🚀 New Features

* 🧩 **Component/Module**
  * [Feature description including PR-number]
  * [Feature description]

* 🧩 **Component/Module**
  * [Feature description]

## 🏷️ Other Features

* 🔁 **Tests**
  * [Test-related changes]

* 📚 **Documentation and Examples**
  * [Docs-related changes]
  * [Examples-related changes]

* 🐍 **Build and Compatibility**
  * [Build/compatibility changes]

* 🎨 **Code Style**
  * [Style-related changes]
  * [Formatter-related changes]
  * [Linting-related changes]

* 🛡️ **Security**
  * [Security-related changes]

## 🐛 Bug-fixes
* [Bugfix description]
* [Bugfix description]

## 👨‍💻 Contributors
* [**@username**](https://github.com/username) — [Contribution summary] (#PR-number)


## 📝 Full changelog

| **N commits** | 📚 Docs | 🧪 Tests | 🐛 Fixes | 🎨 Style | ✨ Features | Other |
|---------------|---------|----------|----------|----------|-------------|-------|
| % of Commits  | XX%     | XX%      | XX%      | XX%      | XX%         | XX%   |

`git log vX.Y.Z-1...vX.Y.Z --date=short --pretty=format:"* %ad %d %s (%aN)*`

Full changelog: https://github.com/ismatorresgarcia/acherus/compare/vX.Y.Z-1...vX.Y.Z
```
