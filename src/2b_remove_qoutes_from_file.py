file_name = "dataset/1M_sv.csv/1.csv"
with open(file_name) as f:
    newText = f.read().replace('"', '')

with open(file_name, "w") as f:
    f.write(newText)
