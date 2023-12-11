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
    print("Running Exploratory Data Analysis...")
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
    plt.show()

    # Investigate the Involvement of Attackers and Defenders
    attackers = battles[["attacker_1", "attacker_2", "attacker_3", "attacker_4"]].stack().value_counts()
    defenders = battles[["defender_1", "defender_2", "defender_3", "defender_4"]].stack().value_counts()

    # Plotting the top attackers and defenders
    plt.figure(figsize=(14, 6))
    attackers.head(10).plot(kind="bar", color="orange", alpha=0.7, label="Attackers")
    defenders.head(10).plot(kind="bar", color="blue", alpha=0.7, label="Defenders")
    plt.title("Top Attackers and Defenders")
    plt.legend()
    plt.show()

    # Analyze the Size of Attacking and Defending Forces
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="attacker_size", y="defender_size", data=battles, hue="attacker_outcome")
    plt.title("Size of Attacking and Defending Forces")
    plt.show()

    character_deaths.info()

    character_deaths.head()

    # Check for Missing Values
    print("Missing Values:\n", character_deaths[["Death Year", "Book of Death", "Death Chapter"]].isnull().sum())

    # Explore Gender Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Gender", data=character_deaths)
    plt.title("Gender Distribution of Characters")
    plt.show()

    # Analyze Nobility Status
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Nobility", data=character_deaths)
    plt.title("Nobility Status of Characters")
    plt.show()

    # Investigate Allegiances and Their Relation to Fates
    plt.figure(figsize=(12, 6))
    sns.countplot(x="Allegiances", hue="Death Year", data=character_deaths)
    plt.xticks(rotation=45, ha="right")
    plt.title("Allegiances and Death Years")
    plt.show()

    # Examine Death-related Columns
    plt.figure(figsize=(14, 6))
    sns.countplot(x="Death Year", hue="Book of Death", data=character_deaths)
    plt.title("Death Year and Book of Death")
    plt.show()

    # Check for NANs values
    data_NaN = character_predictions.isna().sum()
    data_NaN[data_NaN > 0]
    len(data_NaN)

    # Check which characters have a negative age and it's value.
    print("Mean age of characters before adjustment - ",character_predictions["age"].mean())
    print(character_predictions["name"][character_predictions["age"] < 0])
    print(character_predictions['age'][character_predictions['age'] < 0])

    character_predictions.loc[1684, "age"] = 24.0  # Doreah is actually around 24
    character_predictions.loc[1868, "age"] = 0.0  # Rhaego was never born

    print("Mean age of characters after adjustment - ",character_predictions["age"].mean())

    character_predictions["age"].fillna(character_predictions["age"].mean(), inplace=True)
    character_predictions["culture"].fillna("", inplace=True)
    character_predictions.fillna(value=-1, inplace=True)

    # plotting violin plots to visualize the distribution for both alive, dead
    f, ax = plt.subplots(2, 2, figsize=(17, 15))
    sns.violinplot(data=character_predictions, x="isPopular", y="isNoble", hue="isAlive", split=True, ax=ax[0, 0])
    ax[0, 0].set_title('Noble and Popular vs Mortality')
    ax[0, 0].set_yticks(range(2))

    sns.violinplot(data=character_predictions, x="isPopular", y="male", hue="isAlive", split=True, ax=ax[0, 1])
    ax[0, 1].set_title('Male and Popular vs Mortality')
    ax[0, 1].set_yticks(range(2))

    sns.violinplot(data=character_predictions, x="isPopular", y="isMarried", hue="isAlive", split=True, ax=ax[1, 0])
    ax[1, 0].set_title('Married and Popular vs Mortality')
    ax[1, 0].set_yticks(range(2))

    sns.violinplot(data=character_predictions, x="isPopular", y="book1", hue="isAlive", split=True, ax=ax[1, 1])
    ax[1, 1].set_title('Book_1 and Popular vs Mortality')
    ax[1, 1].set_yticks(range(2))
    plt.title("Violin plots to visualize the distribution for both classes (alive, dead) in our dataset")
    plt.show()

    set(character_predictions['culture'])

    # Get all of the values for culture in our dataset
    cult = {
        'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
        'Ghiscari': ['ghiscari', 'ghiscaricari', 'ghis'],
        'Asshai': ["asshai'i", 'asshai'],
        'Lysene': ['lysene', 'lyseni'],
        'Andal': ['andal', 'andals'],
        'Braavosi': ['braavosi', 'braavos'],
        'Dornish': ['dornishmen', 'dorne', 'dornish'],
        'Myrish': ['myr', 'myrish', 'myrmen'],
        'Westermen': ['westermen', 'westerman', 'westerlands'],
        'Westerosi': ['westeros', 'westerosi'],
        'Stormlander': ['stormlands', 'stormlander'],
        'Norvoshi': ['norvos', 'norvoshi'],
        'Northmen': ['the north', 'northmen'],
        'Free Folk': ['wildling', 'first men', 'free folk'],
        'Qartheen': ['qartheen', 'qarth'],
        'Reach': ['the reach', 'reach', 'reachmen'],
        'Ironborn': ['ironborn', 'ironmen'],
        'Mereen': ['meereen', 'meereenese'],
        'RiverLands': ['riverlands', 'rivermen'],
        'Vale': ['vale', 'valemen', 'vale mountain clans']
    }

    # Grouping culture names
    def get_cult(value):
        value = value.lower()
        v = [k for (k, v) in cult.items() if value in v]
        return v[0] if len(v) > 0 else value.title()
    character_predictions.loc[:, "culture"] = [get_cult(x) for x in character_predictions["culture"]]