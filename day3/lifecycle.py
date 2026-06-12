# Core imports for the whole lab
from email.mime import audio

from matplotlib.pylab import shape
import pandas as pd
import json

print('Setup complete. pandas', pd.__version__)

lifecycle = [
    'Define',    # frame the business problem
    'Collect',   # gather & ingest data
    'Prepare',   # clean & transform
    'Explore',   # EDA & visualise
    'Model',     # train algorithms
    'Evaluate',  # test & validate
    'Deploy',    # serve predictions
    'Monitor',   # watch & maintain  (then loop back!)
]
for i, stage in enumerate(lifecycle, start=1):
    print(f'{i}. {stage}')

    what_happens = {
    'Define':   'turn a business question into an ML problem',
    'Collect':  'pull data from files, databases, APIs',
    'Prepare':  'clean missing values, fix types, engineer features',
    'Explore':  'summarise & visualise to find patterns',
    'Model':    'train and tune algorithms',
    'Evaluate': 'measure performance on unseen data',
    'Deploy':   'put the model into production',
    'Monitor':  'track performance and retrain as needed',
}
for stage, desc in what_happens.items():
    print(f'{stage:9s} -> {desc}')



shuffled = ['Model', 'Define', 'Deploy', 'Collect', 'Monitor',
            'Prepare', 'Evaluate', 'Explore']

# 1 & 2. sort 'shuffled' by position in 'lifecycle'
#   hint: sorted(shuffled, key=lifecycle.index)
# YOUR CODE HERE
sorted_lifecycle = sorted(shuffled, key=lifecycle.index)
print(sorted_lifecycle)

crisp_dm = [
    'Business Understanding',
    'Data Understanding',
    'Data Preparation',
    'Modeling',
    'Evaluation',
    'Deployment',
]
for i, phase in enumerate(crisp_dm, start=1):
    print(f'{i}. {phase}')

    # Several lifecycle stages can fall under one CRISP-DM phase.
mapping = {
    'Define':   'Business Understanding',
    'Collect':  'Data Understanding',
    'Explore':  'Data Understanding',
    'Prepare':  'Data Preparation',
    'Model':    'Modeling',
    'Evaluate': 'Evaluation',
    'Deploy':   'Deployment',
    'Monitor':  'Deployment',
}
for stage, phase in mapping.items():
    print(f'{stage:9s} -> {phase}')

    
# 1. invert the mapping: group stages by phase
#   hint: build a dict, append each stage to mapping[stage]'s list
grouped = {}
# YOUR CODE HERE
for stage, phase in mapping.items():
    if phase not in grouped:
        grouped[phase] = []
    grouped[phase].append(stage)

print(grouped)


# 2. print phase -> stages
# YOUR CODE HERE
for phase, stages in grouped.items():
    print(f'{phase:20s} -> {stages}')

    # 3. which phase has the most stages?  (max by length)
# YOUR CODE HERE
most_stages = max(grouped.items(), key=lambda item: len(item[1]))
print(f'Phase with most stages: {most_stages[0]} ({len(most_stages[1])} stages)')

examples = {
    'a SQL users table':        'structured',
    'a CSV of sales':           'structured',
    'a JSON API response':      'semi-structured',
    'an XML config file':       'semi-structured',
    'a folder of photos':       'unstructured',
    'customer review text':     'unstructured',
}
for item, kind in examples.items():
    print(f'{kind:16s} <- {item}')


    # -----------------------------------------------------------
# 🔹 3B. A SIMPLE CLASSIFIER
# -----------------------------------------------------------

def classify(fmt):
    fmt = fmt.lower()
    if fmt in ('csv', 'sql', 'parquet', 'excel', 'xlsx'):
        return 'structured'
    if fmt in ('json', 'xml', 'yaml', 'log'):
        return 'semi-structured'
    return 'unstructured'   # text, image, audio, video, ...

for fmt in ['csv', 'json', 'jpg', 'sql', 'xml', 'mp3']:
    print(f'{fmt:6s} -> {classify(fmt)}')

    
to_label = ['parquet', 'yaml', 'png', 'csv', 'wav', 'json', 'txt']

# 1. classify each one
# YOUR CODE HERE
labels = [classify(fmt) for fmt in to_label]
print(labels)

# 2. count per category  (hint: collections.Counter)
# YOUR CODE HERE    
from collections import Counter
counts = Counter(labels)
print(counts)

# 3. An ML task that uses unstructured data: ...   (write your answer)  
# YOUR ANSWER HERE
# An ML task that uses unstructured data: image classification, sentiment analysis, speech recognition, etc.

# -----------------------------------------------------------
# 🔹 4A. WRITE A CSV, THEN READ IT BACK
# -----------------------------------------------------------

# (in real life the CSV already exists; here we create one so the lab is self-contained)
sample = pd.DataFrame({
    'name':   ['Ana', 'Bo', 'Cy', 'Di'],
    'age':    [30, 25, 41, 38],
    'city':   ['Pune', 'Delhi', 'Pune', 'Hyderabad'],
    'spend':  [120.5, 80.0, 200.2, 150.0],
})
sample.to_csv('people.csv', index=False)   # write

df = pd.read_csv('people.csv')             # read back
df.head()

# -----------------------------------------------------------
# 🔹 4B. ALWAYS INSPECT A NEW DATAFRAME
# -----------------------------------------------------------

print('shape :', df.shape)        # (rows, columns)
print()
print('dtypes:')
print(df.dtypes)                  # the type of each column
print()
df.describe(include='all')        # quick summary

# -----------------------------------------------------------
# 🔹 4C. READING JSON (semi-structured -> table)
# -----------------------------------------------------------

# JSON often comes nested; json_normalize flattens it into columns
records = [
    {'id': 1, 'user': {'name': 'Ana', 'city': 'Pune'}, 'spend': 120.5},
    {'id': 2, 'user': {'name': 'Bo',  'city': 'Delhi'}, 'spend': 80.0},
]
json_df = pd.json_normalize(records)   # note the 'user.name', 'user.city' columns
json_df

# 1. build the DataFrame
# YOUR CODE HERE
records = [
    {'id': 1, 'user': {'name': 'Ana', 'city': 'Pune'}, 'spend': 120.5},
    {'id': 2, 'user': {'name': 'Bo',  'city': 'Delhi'}, 'spend': 80.0},
]
json_df = pd.json_normalize(records)
print(json_df)

    # 2. write to 'products.csv' then read it back
# YOUR CODE HERE
json_df.to_csv('products.csv', index=False)
products_df = pd.read_csv('products.csv')
print(products_df)


# -----------------------------------------------------------
# 🔹 5A. FETCH JSON FROM A PUBLIC API
# -----------------------------------------------------------
import requests

API_URL = 'https://jsonplaceholder.typicode.com/users'   # a free demo API

# Pattern: GET the URL -> parse JSON -> build a DataFrame.
# We wrap it in try/except so the lab still runs if you're offline;
# in that case we fall back to a small inline sample.
try:
    resp = requests.get(API_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    print('Fetched', len(data), 'records from the live API.')
except Exception as e:
    print('Network unavailable (', type(e).__name__, ') - using inline sample.')
    data = [
        {'id': 1, 'name': 'Leanne Graham',  'username': 'Bret',     'email': 'sincere@april.biz'},
        {'id': 2, 'name': 'Ervin Howell',   'username': 'Antonette','email': 'shanna@melissa.tv'},
        {'id': 3, 'name': 'Clementine Bauch','username': 'Samantha','email': 'nathan@yesenia.net'},
    ]

api_df = pd.json_normalize(data)
api_df.head()

# -----------------------------------------------------------
# 🔹 5B. INSPECT & SELECT FROM THE API DATA
# -----------------------------------------------------------

print('shape:', api_df.shape)
# keep just a few useful columns (these exist in both live & sample data)
cols = [c for c in ['id', 'name', 'username', 'email'] if c in api_df.columns]
api_df[cols].head()


POSTS_URL = 'https://jsonplaceholder.typicode.com/posts'

# 1. fetch (try/except with a small inline fallback)
# YOUR CODE HERE
try:
    resp = requests.get(POSTS_URL, timeout=10)
    resp.raise_for_status()
    posts_data = resp.json()
    print('Fetched', len(posts_data), 'posts from the live API.')

except Exception as e:
    print('Network unavailable (', type(e).__name__, ') - using inline sample.')
    posts_data = [
        {'userId': 1, 'id': 1, 'title': 'Post 1', 'body': 'Lorem ipsum...'},
        {'userId': 1, 'id': 2, 'title': 'Post 2', 'body': 'Dolor sit amet...'},
        {'userId': 2, 'id': 3, 'title': 'Post 3', 'body': 'Consectetur adipiscing...'},
    ]

# 2. build a DataFrame
# YOUR CODE HERE
posts_df = pd.json_normalize(posts_data)
print(posts_df.head())

# 2. build the DataFrame
# YOUR CODE HERE
posts_df = pd.json_normalize(posts_data)
print(posts_df.head())

# 3. shape + first 3 rows
# YOUR CODE HERE
print('shape:', posts_df.shape)
print(posts_df.head(3))
