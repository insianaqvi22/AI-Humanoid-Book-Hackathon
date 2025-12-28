# Content API Contract: AI Humanoid Robotics Book

## Overview
This document defines the contract between the book content (Markdown files) and the Docusaurus framework that renders the content as a website.

## Content Metadata Requirements

### Required Frontmatter
Every Markdown file in the `docs/` directory must include the following frontmatter:

```yaml
---
title: string (required)           # Title of the document
sidebar_position: integer (required) # Position in sidebar navigation
description: string (required)     # Brief description for SEO
---
```

### Optional Frontmatter
```yaml
---
image: string                     # Social media preview image
keywords: array of strings        # SEO keywords
tags: array of strings            # Content tags for categorization
---

## Content Structure Contract

### Document Structure
Each Markdown document must follow this structure:
1. Frontmatter (YAML header)
2. Main title (H1) - should match the title in frontmatter
3. Content sections (H2, H3, etc.)
4. Code examples (fenced code blocks)
5. Diagrams/images (with alt text)

### Content Requirements
- All content must be in valid Markdown format
- Code examples must specify the programming language
- Images must include alt text for accessibility
- Internal links must use relative paths
- External links should open in new tabs when appropriate

## Navigation Contract

### Sidebar Structure
- Sidebar items are automatically generated from the directory structure
- Sidebar position is determined by the `sidebar_position` frontmatter value
- Nested items follow the directory hierarchy

### Breadcrumb Contract
- Breadcrumbs are automatically generated based on the directory structure
- The hierarchy follows: Home > Category > Subcategory > Document

## Search Contract

### Content Indexing
- All text content is indexed for search
- Headings (H1, H2, H3) are prioritized in search results
- Code blocks are included in search indexing
- Images alt text is included in search indexing

## Build Contract

### Build Process Requirements
- All internal links must be valid relative paths
- All image references must point to existing files
- Markdown syntax must be valid (no parsing errors)
- Frontmatter must be properly formatted YAML