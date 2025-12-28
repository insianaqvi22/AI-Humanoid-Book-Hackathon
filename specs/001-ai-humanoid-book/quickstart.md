# Quickstart Guide: AI Humanoid Robotics Book

## Overview
This guide will help you get started with the AI Humanoid Robotics Book project. The book is built using Docusaurus, a modern static website generator, and is designed to provide comprehensive educational content about AI-powered humanoid robotics.

## Prerequisites
- Node.js (version 16 or higher)
- Git
- Basic knowledge of Markdown syntax
- Understanding of AI and robotics concepts (for content creation)

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/insianaqvi22/AI-Humanoid-Book-Hackathon.git
cd AI-Humanoid-Book-Hackathon
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Start the Development Server
```bash
npm start
```
This command starts a local development server and opens the book in your browser. Most changes are reflected live without having to restart the server.

### 4. Project Structure
The book content is organized as follows:
- `docs/` - Contains all the book content in Markdown format
- `docs/getting-started/` - Introduction and foundational concepts
- `docs/ai-concepts/` - AI-related topics for humanoid robotics
- `docs/humanoid-design/` - Design principles for humanoid robots
- `docs/implementation/` - Practical implementation examples
- `docs/advanced-topics/` - Advanced concepts and future directions
- `docs/code-examples/` - Code samples in various programming languages

## Adding New Content

### Creating a New Chapter
1. Create a new Markdown file in the appropriate directory under `docs/`
2. Add frontmatter to the file:
```markdown
---
title: Chapter Title
sidebar_position: [position_number]
description: Brief description of the chapter
---
```

### Adding a New Section to an Existing Chapter
1. Add a new heading in the chapter file using Markdown syntax:
```markdown
## Section Title
Content for the section...
```

### Adding Code Examples
Use fenced code blocks with language specification:
```markdown
import numpy as np

def robot_move(direction):
    """Move the robot in the specified direction"""
    print(f"Moving robot {direction}")
```

## Building the Book
To build the static site for production:
```bash
npm run build
```
The built site will be in the `build/` directory and can be deployed to any static hosting service.

## Deployment
The book is configured to deploy to GitHub Pages. After pushing changes to the `main` branch, the site will be automatically updated.

## Next Steps
- Review the [content guidelines](/docs/contributing/content-guidelines.md) for writing standards
- Check the [existing chapters](/docs/intro.md) to understand the content structure
- Look at [code examples](/docs/code-examples/python/intro.md) to understand the implementation approach