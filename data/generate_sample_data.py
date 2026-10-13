"""
generate_sample_data.py
Generates a realistic raw sales dataset with real-world data quality issues:
- Missing values (NaN / nulls in multiple columns)
- Duplicate records
- Inconsistent data types & formatting (e.g., currency symbols, mixed date formats, messy casing)
- Outliers in numerical fields (IQR anomalies)
- Invalid records (negative quantities, zero/negative prices)
"""

import csv
import random
from datetime import datetime, timedelta

def generate_sales_data(filepath="data/sample_sales.csv", num_records=350, seed=42):
    random.seed(seed)

    products = [
        {"name": "Apple MacBook Pro", "category": "Technology", "price_range": (1299.99, 1999.99)},
        {"name": "Dell XPS 15", "category": "Technology", "price_range": (999.99, 1499.99)},
        {"name": "Logitech Wireless Mouse", "category": "Accessories", "price_range": (24.99, 49.99)},
        {"name": "Keychron Mechanical Keyboard", "category": "Accessories", "price_range": (79.99, 129.99)},
        {"name": "Sony WH-1000XM5 Headphones", "category": "Audio", "price_range": (299.99, 399.99)},
        {"name": "Standing Desk Pro", "category": "Furniture", "price_range": (399.99, 699.99)},
        {"name": "Ergonomic Office Chair", "category": "Furniture", "price_range": (199.99, 349.99)},
        {"name": "Anker USB-C Hub", "category": "Accessories", "price_range": (39.99, 69.99)},
        {"name": "Samsung 4K Monitor", "category": "Technology", "price_range": (329.99, 499.99)},
        {"name": "Bose SoundLink Speaker", "category": "Audio", "price_range": (129.99, 179.99)},
        {"name": "Filing Storage Cabinet", "category": "Furniture", "price_range": (119.99, 189.99)},
        {"name": "Desk Organizer Set", "category": "Office Supplies", "price_range": (15.99, 34.99)},
        {"name": "Gel Pen Pack (12-pack)", "category": "Office Supplies", "price_range": (9.99, 18.99)},
        {"name": "Heavy Duty Paper Shredder", "category": "Office Supplies", "price_range": (69.99, 119.99)},
        {"name": "Smart Fitness Watch", "category": "Technology", "price_range": (149.99, 299.99)},
    ]

    regions = ["North", "South", "East", "West"]
    customers = [
        "Emma Watson", "Liam Smith", "Olivia Brown", "Noah Davis", "Sophia Wilson",
        "James Taylor", "Isabella Martinez", "Benjamin Anderson", "Mia Thomas",
        "Lucas Jackson", "Charlotte White", "Henry Harris", "Amelia Martin",
        "Alexander Clark", "Harper Lewis", "Ethan Robinson", "Evelyn Walker",
        "Daniel Hall", "Abigail Allen", "Matthew Young"
    ]

    start_date = datetime(2023, 1, 1)
    rows = []

    for i in range(1, num_records + 1):
        prod = random.choice(products)
        customer = random.choice(customers)
        region = random.choice(regions)
        date_offset = random.randint(0, 500)
        curr_date = start_date + timedelta(days=date_offset)

        # Date formatting variations
        date_format_choice = random.random()
        if date_format_choice < 0.70:
            order_date_str = curr_date.strftime("%Y-%m-%d")
        elif date_format_choice < 0.85:
            order_date_str = curr_date.strftime("%m/%d/%Y")
        else:
            order_date_str = curr_date.strftime("%d-%m-%Y")

        unit_price = round(random.uniform(*prod["price_range"]), 2)
        quantity = random.randint(1, 8)
        total_sales = round(unit_price * quantity, 2)
        category = prod["category"]
        product_name = prod["name"]
        order_id = f"ORD-{1000 + i}"

        # Inconsistent category casing / whitespace (approx 8% of rows)
        if random.random() < 0.08:
            cat_mod = random.choice([category.lower(), f"  {category}  ", category.upper()])
            category = cat_mod

        # Currency formatting quirk (approx 10% of rows have '$' in unit price or total sales)
        unit_price_str = f"${unit_price}" if random.random() < 0.10 else str(unit_price)
        total_sales_str = f"${total_sales}" if random.random() < 0.10 else str(total_sales)

        # Introduce Missing values intentionally
        # Missing Customer Name (4%)
        if random.random() < 0.04:
            customer = ""
        # Missing Region (4%)
        if random.random() < 0.04:
            region = ""
        # Missing Unit Price (3%)
        if random.random() < 0.03:
            unit_price_str = ""
        # Missing Quantity (3%)
        if random.random() < 0.03:
            quantity = ""
        # Missing Total Sales (5%)
        if random.random() < 0.05:
            total_sales_str = ""

        # Introduce Outliers intentionally
        # Quantity outlier (e.g. bulk order of 85 or 120)
        if random.random() < 0.02:
            quantity = random.choice([75, 95, 120])
            if unit_price_str and unit_price_str.replace('$', '').strip():
                try:
                    p = float(unit_price_str.replace('$', '').strip())
                    total_sales_str = str(round(p * quantity, 2))
                except ValueError:
                    pass

        # Price outlier (luxury custom spec or data entry typo)
        if random.random() < 0.015:
            unit_price_str = str(random.choice([8500.0, 9999.0]))
            if quantity != "":
                total_sales_str = str(round(float(unit_price_str) * float(quantity), 2))

        # Invalid values (negative quantity / price, approx 2%)
        if random.random() < 0.02:
            quantity = random.choice([-1, -4])

        rows.append({
            "Order ID": order_id,
            "Order Date": order_date_str,
            "Product Name": product_name,
            "Category": category,
            "Quantity": quantity,
            "Unit Price": unit_price_str,
            "Total Sales": total_sales_str,
            "Region": region,
            "Customer Name": customer
        })

    # Add duplicate records intentionally (approx 15 duplicate rows)
    duplicate_rows = random.sample(rows[:100], 15)
    rows.extend(duplicate_rows)

    # Shuffle to disperse duplicates throughout the dataset
    random.shuffle(rows)

    fieldnames = [
        "Order ID", "Order Date", "Product Name", "Category",
        "Quantity", "Unit Price", "Total Sales", "Region", "Customer Name"
    ]

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sample dataset successfully generated with {len(rows)} records at: {filepath}")

if __name__ == "__main__":
    generate_sales_data()
