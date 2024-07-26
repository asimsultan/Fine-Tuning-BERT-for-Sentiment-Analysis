# BERT Sentiment Analysis

This project fine-tunes BERT for sentiment analysis on the IMDB dataset. The goal is to classify movie reviews as positive or negative.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Directory Structure](#directory-structure)
- [Results](#results)
- [Resources](#resources)
- [License](#license)

## Overview

Sentiment analysis is a natural language processing (NLP) task where the goal is to determine the sentiment expressed in a piece of text. In this project, we use the BERT model from Hugging Face's transformers library to perform sentiment analysis on the IMDB dataset, which contains movie reviews labeled as positive or negative.

## Requirements

- Python 3.6+
- transformers
- torch
- datasets

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
