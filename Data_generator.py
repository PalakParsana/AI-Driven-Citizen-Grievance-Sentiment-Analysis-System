"""
src/data_generator.py
---------------------
Generates a realistic synthetic Citizen Grievance dataset.
Mimics real-world 311 service request / government complaint data.

Columns (Basic + Extra):
  complaint_id       - Unique ID
  date               - Submission date
  complaint_text     - Raw citizen complaint text
  department         - Target department label
  sentiment          - Sentiment label
  city               - City name
  priority_level     - 1 (low) to 5 (critical)
  resolution_days    - Days taken to resolve
  is_repeat_complaint- 0/1
  submission_channel - online / phone / app / walk-in

Usage:
    python src/data_generator.py
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

np.random.seed(42)
random.seed(42)

OUTPUT_PATH = Path("data/grievances_dataset.csv")

# ── Departments ───────────────────────────────────────────────────────────────
DEPARTMENTS = [
    "Water", "Electricity", "Roads",
    "Sanitation", "Transport", "Healthcare", "Education"
]

# ── Cities (Indian) ───────────────────────────────────────────────────────────
CITIES = [
    "Bengaluru", "Mumbai", "Delhi", "Hyderabad",
    "Chennai", "Pune", "Ahmedabad", "Kolkata"
]

# ── Submission Channels ───────────────────────────────────────────────────────
CHANNELS = ["online", "phone", "app", "walk-in"]

# ── Complaint Templates per Department ────────────────────────────────────────
COMPLAINTS = {
    "Water": [
        "There is no water supply in our area for the past {days} days. Residents are suffering badly.",
        "The water coming from the tap is extremely dirty and has a foul smell. Children are falling sick.",
        "Water pipe has burst on {street} and water is flooding the road. Immediate repair needed urgently.",
        "Our colony has not received clean drinking water since {days} days. This is unacceptable.",
        "Water pressure is very low in our building. Third floor gets no water at all.",
        "Sewage water is mixing with drinking water supply causing serious health hazards.",
        "The water tank in our area overflows daily wasting thousands of litres. Nobody is fixing it.",
        "We have been submitting water shortage complaints for months but no action has been taken.",
    ],
    "Electricity": [
        "Power cut in our area since {days} days. No electricity at all, we cannot survive in this heat.",
        "Frequent power outages every day. This is affecting our business and daily life terribly.",
        "A transformer on {street} exploded yesterday. Still not repaired. Very dangerous situation.",
        "Electricity bill is extremely high this month despite less usage. Clear overcharging happening.",
        "Street lights are not working for weeks. Area is completely dark and robbery risk is high.",
        "Live wire is hanging dangerously on the road. Children are playing nearby. Very urgent!",
        "No electricity for {days} hours during peak summer. Elderly patients are in critical condition.",
        "Power fluctuation is destroying our appliances. This is causing huge financial loss to us.",
    ],
    "Roads": [
        "Road on {street} has huge potholes for months. Two-wheeler accidents happening daily.",
        "No road repair done after monsoon. Roads are completely broken and damaged beyond repair.",
        "The footpath on main road is encroached by shops. Pedestrians are forced to walk on road.",
        "Road cave-in near school gate is very dangerous for children. Needs immediate attention.",
        "Speed breakers on {street} are unmarked and causing serious accidents at night.",
        "Road construction work started {days} months ago but still incomplete. Causing major traffic.",
        "Entire road is waterlogged after rain due to blocked drains. Vehicles getting stuck daily.",
        "Road markings completely faded. No lane discipline possible causing accidents.",
    ],
    "Sanitation": [
        "Garbage not collected for {days} days in our area. Foul smell everywhere. Health risk high.",
        "Garbage dump near our society is overflowing. Rodents and mosquitoes breeding rapidly.",
        "Public toilet near our area is extremely dirty and not maintained. Spreading disease.",
        "Drainage blocked for weeks causing sewage overflow on streets near school.",
        "Stray dogs attacking residents due to uncollected garbage attracting them near homes.",
        "Open defecation happening near children playground due to no public toilet facility.",
        "Waste collection vehicle has not come for {days} days. Residents dumping on road.",
        "Sewage drain open near hospital is a major health hazard. Requires urgent sealing.",
    ],
    "Transport": [
        "Bus frequency on route {route} reduced drastically. Commuters waiting {days} hours daily.",
        "Bus conductor behaved very rudely with senior citizens and refused to give seat.",
        "Auto rickshaws refusing short distance trips and overcharging passengers in our area.",
        "No bus shelter on main road stop. Commuters standing in rain and hot sun daily.",
        "Bus driver overspeeding dangerously. Narrowly missed accident with school children today.",
        "Metro station lift not working for {days} weeks. Elderly and disabled unable to use.",
        "Illegal parking by trucks blocking entire road near market every morning.",
        "Traffic signal at main junction not working for days causing major accidents daily.",
    ],
    "Healthcare": [
        "Government hospital has no doctors available on most days. Patients turned away.",
        "Medicines not available at government dispensary for {days} weeks. Patients suffering badly.",
        "Ambulance took {days} hours to arrive for emergency call. Patient condition worsened.",
        "Hospital staff behaving very rudely with poor patients. This is completely unacceptable.",
        "No female doctor available at health center. Women patients feel extremely uncomfortable.",
        "Vaccination camp cancelled without notice. Parents came with children from far areas.",
        "Hospital building in very poor condition. Ceiling leaking on patients during rains.",
        "Long waiting queues at government hospital. Patients waiting since morning without help.",
    ],
    "Education": [
        "Government school has no teachers for {days} months. Children sitting idle in class.",
        "School building roof leaking badly during rains. Children sitting in water. Dangerous.",
        "Mid-day meal quality very poor. Children falling sick after eating food in school.",
        "No drinking water available in government school. Children going thirsty all day.",
        "School has not received textbooks yet. Academic year already {days} months in progress.",
        "Teacher absent without any substitute for {days} days. No learning happening for children.",
        "Toilet facility in school completely non-functional. Girl students leaving school early.",
        "School admission process very corrupt. Poor families being asked for illegal donations.",
    ],
}

# ── Sentiment mapping ─────────────────────────────────────────────────────────
SENTIMENT_WEIGHTS = {
    "Water":       ["Negative", "Critical/Urgent", "Negative", "Neutral"],
    "Electricity": ["Negative", "Critical/Urgent", "Negative", "Neutral"],
    "Roads":       ["Negative", "Neutral", "Negative", "Neutral"],
    "Sanitation":  ["Negative", "Negative", "Critical/Urgent", "Neutral"],
    "Transport":   ["Neutral", "Negative", "Neutral", "Positive"],
    "Healthcare":  ["Critical/Urgent", "Negative", "Critical/Urgent", "Negative"],
    "Education":   ["Negative", "Neutral", "Negative", "Critical/Urgent"],
}

STREETS = ["MG Road", "Brigade Road", "Hosur Road", "Outer Ring Road",
           "Anna Salai", "Park Street", "FC Road", "SV Road"]
ROUTES  = ["101", "201", "42A", "15C", "88", "303", "7D"]


def fill_template(text):
    return (text
            .replace("{days}",   str(random.randint(2, 30)))
            .replace("{street}", random.choice(STREETS))
            .replace("{route}",  random.choice(ROUTES)))


def sentiment_to_priority(sentiment):
    return {"Positive": 1, "Neutral": 2, "Negative": 3, "Critical/Urgent": 5}.get(sentiment, 2)


def generate_dataset(n_rows=3000):
    records = []
    start   = pd.Timestamp("2022-01-01")
    end     = pd.Timestamp("2023-12-31")

    for i in range(n_rows):
        dept       = random.choice(DEPARTMENTS)
        text_tmpl  = random.choice(COMPLAINTS[dept])
        text       = fill_template(text_tmpl)
        sentiment  = random.choice(SENTIMENT_WEIGHTS[dept])
        priority   = sentiment_to_priority(sentiment)
        if sentiment == "Critical/Urgent":
            priority = 5
        elif sentiment == "Negative":
            priority = random.choice([3, 4])

        date = start + pd.Timedelta(days=random.randint(0, (end - start).days))

        records.append({
            "complaint_id"       : f"CMP{str(i+1).zfill(5)}",
            "date"               : date.date(),
            "complaint_text"     : text,
            "department"         : dept,
            "sentiment"          : sentiment,
            "city"               : random.choice(CITIES),
            "priority_level"     : priority,
            "resolution_days"    : random.randint(1, 45) if sentiment != "Critical/Urgent"
                                   else random.randint(1, 10),
            "is_repeat_complaint": random.choice([0, 0, 0, 1]),
            "submission_channel" : random.choice(CHANNELS),
        })

    df = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Dataset saved to '{OUTPUT_PATH}'")
    print(f"   Rows            : {len(df):,}")
    print(f"   Columns         : {df.columns.tolist()}")
    print(f"   Departments     : {df['department'].unique().tolist()}")
    print(f"   Sentiments      : {df['sentiment'].value_counts().to_dict()}")
    print(f"   Cities          : {df['city'].nunique()} cities")
    return df


if __name__ == "__main__":
    generate_dataset(n_rows=3000)