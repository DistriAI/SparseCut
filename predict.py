import json
import os

# data = []
# for i in range(45):
#     with open(os.path.join(f"./playground/vqav2/chunk{i+1}.json")) as f:
#         part = json.load(f)
#     data.extend(part)
#
# print(len(data))
# with open("./playground/vqav2.json", "w") as fout:
#     json.dump(data, fout)

with open("./playground/vqav2.json", "r") as f:
    data = json.load(f)

data4save = []
for item in data:
    for key, value in item.items():  # 因为只有一个，所以循环一次
        key = int(key)
        data4save.append({"question_id":key, "answer":value})

with open("./playground/vqav2_submit.json", "w") as fout:
    json.dump(data4save, fout)


