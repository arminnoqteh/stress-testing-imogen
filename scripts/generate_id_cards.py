"""
Generate synthetic ID card images and JSON labels for benchmarking.
"""
import random
from benchmark.utils import save_id_card_pair

N = 20
OUT_DIR = "test_id_cards"

first_names = ["Alice", "Bob", "Carlos", "Diana", "Eve", "Farhan", "Gina", "Hassan"]
last_names = ["Khan", "Smith", "Ali", "Garcia", "Patel", "Zhang"]
cities = ["Lahore", "Karachi", "Islamabad", "Multan", "Peshawar"]
provinces = ["Punjab", "Sindh", "Islamabad", "Khyber Pakhtunkhwa"]

for i in range(N):
    fn = random.choice(first_names)
    ln = random.choice(last_names)
    name = f"{fn} {ln}"
    social_no = f"{random.randint(1000000000,9999999999)}"
    father = f"{random.choice(first_names)} {random.choice(last_names)}"
    city = random.choice(cities)
    province = random.choice(provinces)
    img_path, json_path = save_id_card_pair(OUT_DIR, i, name, social_no, father, city, province)
    print(f"Saved: {img_path}, {json_path}")

print(f"Generated {N} id cards in {OUT_DIR}")
