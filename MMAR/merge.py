import re
import json

def extract_records(raw_text):
    records = []
    pattern = re.compile(
        r'"id"\s*:\s*"([^"]+)"[\s\S]*?"answer_prediction"\s*:\s*"([\s\S]*?)"\s*(?:,|\})'
    )
    for m in pattern.finditer(raw_text):
        _id = m.group(1)
        pred = m.group(2)
        records.append((_id, pred))
    return records


def main(a_path, b_path, out_path):
    with open(a_path, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    with open(b_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_b = f.read()

    id2pred = {i: p for i, p in extract_records(raw_b)}

    for sample in data_a:
        _id = sample.get("id")
        if _id in id2pred:
            sample["model_prediction"] = id2pred[_id]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_a, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])

