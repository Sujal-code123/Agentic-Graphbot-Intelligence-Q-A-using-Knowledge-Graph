# Agentic-Graphbot-Intelligence-Q-A-using-Knowledge-Graph
# Intelligent Hospital Assistance using Agentic AI and Knowledge Graph Reasoning

## Overview

This project is an AI-powered hospital analytics and query system that allows users to ask natural language questions about hospital data. The system automatically converts user queries into Cypher queries, retrieves data from a Neo4j Knowledge Graph, and generates structured insights along with visual charts.

The system integrates Large Language Models (LLMs), Knowledge Graphs, and embeddings to enable intelligent reasoning over healthcare data.

---

## Features

- Natural Language Querying of hospital data
- Automatic Cypher query generation using LLM
- Knowledge Graph built using Neo4j
- Semantic search using sentence embeddings
- Visualization of query results (Bar, Line, Pie charts)
- Interactive UI using Gradio

---

## Architecture

User Question  
↓  
LLM generates Cypher Query  
↓  
Neo4j Knowledge Graph  
↓  
Query Result  
↓  
LLM generates explanation  
↓  
Visualization (Charts)

---

## Tech Stack

- Python
- Neo4j Graph Database
- LangChain
- OpenAI / LLM API
- Sentence Transformers
- Pandas
- Matplotlib
- Gradio

---

## Dataset

The system uses hospital related datasets:

- patients.csv
- doctors.csv
- appointments.csv
- treatments.csv
- billing.csv

These datasets are used to construct a healthcare knowledge graph.

---

## Knowledge Graph Structure

Nodes:

- Patient
- Doctor
- Appointment
- Treatment
- Bill

Relationships:

Patient → HAS_APPOINTMENT → Appointment  
Doctor → PERFORMED_BY → Appointment  
Appointment → HAS_TREATMENT → Treatment  
Patient → HAS_BILL → Bill  
Treatment → BILLED_AS → Bill

---
