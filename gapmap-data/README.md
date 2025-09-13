# Gap Map Data Export

## Overview
This export contains research gaps and foundational capabilities data from the Gap Map project.

## Files
- `gapmap-data.json`: Complete dataset with all entities
- `fields.json`: Research fields/disciplines
- `gaps.json`: R&D Gaps 
- `capabilities.json`: Foundational Capabilities
- `resources.json`: Related resources and publications
- `metadata.json`: Export information and statistics

## Data Structure
- Fields have: id, name, slug, description
- R&D Gaps have: id, name, slug, description, field, foundationalCapabilities (IDs), tags
- Capabilities have: id, name, slug, description, gaps (IDs), resources (IDs), tags
- Resources have: id, title, url, summary, types

## Contact
For questions or corrections, please contact gapmap@convergentresearch.org.

## Last Updated
2025-08-04
