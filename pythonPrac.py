# ── Adding, updating, and removing entries ───────────────────────────
marks = {'Math': 88, 'Physics': 75, 'Chemistry': 82}  # initial marks

marks['Biology'] = 91           # add a new subject
marks['Math']    = 95           # update existing value
del marks['Chemistry']          # remove a key

print('Updated marks:', marks)  # {'Math': 95, 'Physics': 75, 'Biology': 91}

# Iterating over key-value pairs
for subject, score in marks.items():       # .items() gives (key, value) pairs
    print(f'  {subject:10s}: {score}')     # formatted output