#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as mpatches
import warnings

# Use Non-Interactive Backend:
# set the matplotlib backend to a non-interactive one:
mpl.use('Agg')

# Ignore seaborn categorical dtype warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

def exploratory_data_analysis():
    battles = pd.read_csv("data_files/battles.csv")
    character_deaths = pd.read_csv("data_files/character-deaths.csv")
    character_predictions = pd.read_csv("data_files/character-predictions.csv")

    battles.loc[:, "defender_count"] = (4 - battles[["defender_1", "defender_2", "defender_3", "defender_4"]].isnull().sum(axis=1))
    battles.loc[:, "attacker_count"] = (4 - battles[["attacker_1", "attacker_2", "attacker_3", "attacker_4"]].isnull().sum(axis=1))

    battles.info()

    battles.head()

    # Check for Missing Values
    print("Missing Values:\n", battles.isnull().sum())

    # Handle missing values
    battles["attacker_king"].fillna("Unknown", inplace=True)
    battles["defender_king"].fillna("Unknown", inplace=True)
    battles["attacker_outcome"].fillna("Unknown", inplace=True)

    # Examine the Distribution of Battles Across Years
    plt.figure(figsize=(10, 6))
    sns.countplot(x="year", data=battles)
    plt.title("Distribution of Battles Across Years")
    plt.savefig('appendix/distribution_of_battles.png')

    majorevents = battles.groupby('year').sum()[["major_death", "major_capture"]].plot.bar(rot=0)
    _ = majorevents.set(xlabel="Year", ylabel="No. of Death/Capture Events", ylim=(0, 9)), majorevents.legend(["Major Deaths", "Major Captures"])

    # Explore Types of Battles and Their Outcomes
    plt.figure(figsize=(12, 6))
    sns.countplot(x="battle_type", hue="attacker_outcome", data=battles)
    plt.title("Battle Types and Outcomes")
    plt.savefig('appendix/battle_types_and_outcomes.png')

    # Investigate the Involvement of Attackers and Defenders
    attackers = battles[["attacker_1", "attacker_2", "attacker_3", "attacker_4"]].stack().value_counts()
    defenders = battles[["defender_1", "defender_2", "defender_3", "defender_4"]].stack().value_counts()

    # Plotting the top attackers and defenders
    plt.figure(figsize=(14, 6))
    attackers.head(10).plot(kind="bar", color="orange", alpha=0.7, label="Attackers")
    defenders.head(10).plot(kind="bar", color="blue", alpha=0.7, label="Defenders")
    plt.title("Top Attackers and Defenders")
    plt.legend()
    plt.savefig('appendix/top_attackers_and_defenders.png')

    # Analyze the Size of Attacking and Defending Forces
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="attacker_size", y="defender_size", data=battles, hue="attacker_outcome")
    plt.title("Size of Attacking and Defending Forces")
    plt.savefig('appendix/size_forces.png')

    character_deaths.info()

    character_deaths.head()

    # Check for Missing Values
    print("Missing Values:\n", character_deaths[["Death Year", "Book of Death", "Death Chapter"]].isnull().sum())

    # Explore Gender Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Gender", data=character_deaths)
    plt.title("Gender Distribution of Characters")
    plt.savefig('appendix/gender_distribution.png')

    # Analyze Nobility Status
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Nobility", data=character_deaths)
    plt.title("Nobility Status of Characters")
    plt.savefig('appendix/nobility_status_characters.png')

    # Investigate Allegiances and Their Relation to Fates
    plt.figure(figsize=(12, 6))
    sns.countplot(x="Allegiances", hue="Death Year", data=character_deaths)
    plt.xticks(rotation=45, ha="right")
    plt.title("Allegiances and Death Years")
    plt.savefig('appendix/allegiances_and_death_years.png')

    # Examine Death-related Columns
    plt.figure(figsize=(14, 6))
    sns.countplot(x="Death Year", hue="Book of Death", data=character_deaths)
    plt.title("Death Year and Book of Death")
    plt.savefig('appendix/death_year_and_book_of_death.png')