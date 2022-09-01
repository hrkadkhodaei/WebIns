L = []
cnt = 0
with open("dataset/similar_pages/22.csv") as fin:
    Lines = fin.readlines()
    for line in Lines:
        cnt += 1
        x = line.replace('"', '').replace('[', '').replace(']', '')
        L.append(x)
        t = cnt % 1000
        if t == 0:
            print(int(cnt / 1000))

with open("dataset/similar_pages/33.csv", "w+") as fp:
    fp.writelines(L)

