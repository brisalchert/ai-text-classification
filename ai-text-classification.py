#-----------------------------------------------------------------------------------------------------------------------
# AI Text Classification
#
# Python script for training a CNN for classifying AI-generated versus human-generated text.
#-----------------------------------------------------------------------------------------------------------------------

# Import packages
import numpy as np
import pandas as pd

# Load dataset from csv
ai_human_df = pd.read_csv('ai_human.csv')

# Preview data
print(ai_human_df.head())
