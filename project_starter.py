import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
import json
import re
import threading
from typing import Optional

from smolagents import OpenAIServerModel, ToolCallingAgent, tool

dotenv.load_dotenv()

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
)

# Exact catalog names -> catalog entries, used to validate item names in every tool
CATALOG = {item["item_name"]: item for item in paper_supplies}

# Bulk discount policy: (minimum total units in the order, discount rate)
BULK_DISCOUNTS = [(5000, 0.15), (2000, 0.10), (500, 0.05), (0, 0.0)]


def bulk_discount_rate(total_units: int) -> float:
    """Return the discount rate for an order with the given total number of units."""
    for min_units, rate in BULK_DISCOUNTS:
        if total_units >= min_units:
            return rate
    return 0.0


def discount_reason(total_units: int) -> str:
    """Explain in plain words which bulk discount applies to an order of this size."""
    for min_units, rate in BULK_DISCOUNTS:
        if total_units >= min_units and rate > 0:
            return (f"{int(rate * 100)}% bulk discount because the order has {total_units} units "
                    f"({min_units} or more)")
    return f"no bulk discount, because discounts start at {BULK_DISCOUNTS[-2][0]} units"


def current_stock(item_name: str, as_of_date: str) -> int:
    """Return the stock of one item as a plain integer (wraps get_stock_level)."""
    stock_df = get_stock_level(item_name, as_of_date)
    return int(stock_df["current_stock"].iloc[0])


def parse_json(answer) -> dict:
    """Turn an agent's final answer into a dict (agents are asked to answer with JSON)."""
    if isinstance(answer, dict):
        return answer
    match = re.search(r"\{.*\}", str(answer), re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# Shared state for the request being processed: the last calculated quote and every sale attempt.
# Tools write the real results here, so the customer reply is based on facts, not on LLM wording.
ORDER_STATE = {"request_date": "", "inventory": {}, "unmatched": [], "quote": {}, "sales": []}

# Risk limit: one restock may cost at most this share of the cash currently available
MAX_RESTOCK_SHARE_OF_CASH = 0.10

# State coordination: agents can run tool calls in parallel, but SQLite allows one writer at a time.
# Tools that check and then change stock run under this lock, so "check, then reserve" is atomic.
DB_LOCK = threading.Lock()


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent


@tool
def list_catalog() -> str:
    """List every product the company sells, with its exact catalog name, category and unit price.

    Returns:
        A JSON list of catalog items.
    """
    return json.dumps(paper_supplies)


@tool
def check_inventory(request_date: Optional[str] = None) -> str:
    """Show all items that are currently in stock and how many units of each (on the request date).

    Args:
        request_date: Optional and ignored; the request date is already known to the system.

    Returns:
        A JSON object mapping item names to units in stock.
    """
    inventory = get_all_inventory(ORDER_STATE["request_date"])
    return json.dumps({name: int(units) for name, units in inventory.items()})


@tool
def check_item_stock(item_name: str, request_date: Optional[str] = None) -> str:
    """Return how many units of one catalog item are in stock on the request date.

    Args:
        item_name: The exact catalog item name.
        request_date: Optional and ignored; the request date is already known to the system.

    Returns:
        A short text with the stock level.
    """
    if item_name not in CATALOG:
        return f"'{item_name}' is not a catalog item."
    return f"{item_name}: {current_stock(item_name, ORDER_STATE['request_date'])} units in stock."


@tool
def record_unavailable_item(requested: str, reason: str) -> str:
    """Record a requested item that cannot be matched to any catalog product (we do not sell it).

    Args:
        requested: The item as the customer wrote it, e.g. "200 balloons".
        reason: A short customer-friendly reason, e.g. "not an item we sell".

    Returns:
        A confirmation text.
    """
    key = requested.strip().lower()
    if all(item["requested"].strip().lower() != key for item in ORDER_STATE["unmatched"]):
        ORDER_STATE["unmatched"].append({"requested": requested, "reason": reason})
    return f"Recorded as unavailable: {requested}."


def _check_and_restock_locked(item_name: str, requested: str, quantity: int, needed_by: str) -> str:
    """Stock check and restock rules for check_and_restock; must be called while holding DB_LOCK."""
    request_date = ORDER_STATE["request_date"]  # the system's request date always wins
    if item_name not in CATALOG:
        return json.dumps({"available": False, "reason": "not an item we sell"})
    if quantity <= 0:
        return json.dumps({"available": False, "reason": "the quantity must be positive"})
    entry = ORDER_STATE["inventory"].get(item_name, {"lines": {}, "restocked": 0})
    key = requested.strip().lower()
    # A repeated call may raise the quantity but never lower it (no shrinking orders to "whatever is in stock")
    entry["lines"][key] = max(entry["lines"].get(key, 0), quantity)
    # Use the largest requested amount, so the same item described twice is never counted twice
    total = max(entry["lines"].values())
    entry.update({"quantity": total, "available": True, "ready_date": request_date, "reason": ""})
    stock = current_stock(item_name, request_date)
    if stock < total:
        shortfall = total - stock
        arrival = get_supplier_delivery_date(request_date, shortfall)
        cost = round(shortfall * CATALOG[item_name]["unit_price"], 2)
        if arrival > needed_by:
            entry.update({"available": False, "reason": f"our supplier can only deliver on {arrival}, "
                                                        f"after your deadline of {needed_by}"})
        elif cost > MAX_RESTOCK_SHARE_OF_CASH * get_cash_balance(request_date):
            entry.update({"available": False, "reason": "we cannot restock this quantity in time"})
        else:
            create_transaction(item_name, "stock_orders", shortfall, cost, request_date)
            entry.update({"restocked": entry["restocked"] + shortfall, "ready_date": arrival})
    ORDER_STATE["inventory"][item_name] = entry
    return json.dumps({k: entry[k] for k in ("available", "ready_date", "reason")})


@tool
def check_and_restock(item_name: str, requested: str, quantity: int, needed_by: str,
                      request_date: Optional[str] = None) -> str:
    """Make sure enough units of an item are available for the customer's deadline.
    If stock is short, orders the missing units from the supplier, but only if they arrive
    before the deadline and the restock stays within the company's risk limit. Calling it again
    for the same requested item replaces the earlier quantity (it never restocks the same units twice).

    Args:
        item_name: The exact catalog item name.
        requested: The item as the customer wrote it, e.g. "heavy cardstock (white)".
        quantity: The full number of units the customer asked for (never reduce it to the stock level).
        needed_by: The date the customer needs the items, in YYYY-MM-DD format.
        request_date: Optional and ignored; the request date is already known to the system.

    Returns:
        A JSON object with available (true/false), ready_date and, if unavailable, a customer-friendly reason.
    """
    with DB_LOCK:  # check and restock happen as one uninterrupted step
        return _check_and_restock_locked(item_name, requested, quantity, needed_by)


# Tools for quoting agent


@tool
def find_similar_quotes(search_terms: list[str]) -> str:
    """Search past quotes for similar requests, to keep prices consistent with history.

    Args:
        search_terms: One or two keywords from the request, e.g. ["cardstock"] or ["ceremony"].

    Returns:
        A JSON list of up to 5 matching past quotes with their totals and explanations.
    """
    quotes = search_quote_history(search_terms, limit=5)
    return json.dumps(quotes, default=str)


@tool
def calculate_quote(request_date: Optional[str] = None) -> str:
    """Price every item the inventory step secured for this request, using catalog unit prices
    and the bulk discount policy. Items and quantities come from the system.

    Args:
        request_date: Optional and ignored; the request date is already known to the system.

    Returns:
        A JSON object with each line's price after discount, the discount rate and the total.
    """
    lines = [{"item_name": name, "quantity": entry["quantity"]}
             for name, entry in ORDER_STATE["inventory"].items() if entry["available"]]
    total_units = sum(int(line["quantity"]) for line in lines)
    rate = bulk_discount_rate(total_units)
    priced_lines = []
    for line in lines:
        unit_price = CATALOG[line["item_name"]]["unit_price"]
        quantity = int(line["quantity"])
        priced_lines.append({
            "item_name": line["item_name"],
            "quantity": quantity,
            "unit_price": unit_price,
            "line_total": round(quantity * unit_price * (1 - rate), 2),
        })
    subtotal = round(sum(line["quantity"] * line["unit_price"] for line in priced_lines), 2)
    quote = {
        "lines": priced_lines,
        "total_units": total_units,
        "discount_rate": rate,
        "discount_reason": discount_reason(total_units),
        "subtotal": subtotal,
        "total": round(sum(line["line_total"] for line in priced_lines), 2),
    }
    ORDER_STATE["quote"] = quote
    return json.dumps(quote)


# Tools for ordering agent


def _finalize_sale_locked(item_name: str) -> str:
    """Sale rules for finalize_sale; must be called while holding DB_LOCK."""
    sale_date = ORDER_STATE["request_date"]
    line = next((l for l in ORDER_STATE["quote"].get("lines", []) if l["item_name"] == item_name), None)
    if line is None:
        return f"FAILED: '{item_name}' is not in the current quote."
    if any(s["item_name"] == item_name and s["status"] == "sold" for s in ORDER_STATE["sales"]):
        return f"ALREADY SOLD: {item_name}."
    stock = current_stock(item_name, sale_date)
    if stock < line["quantity"]:
        reason = "not enough stock available by your deadline"
        ORDER_STATE["sales"].append({"item_name": item_name, "status": "failed", "reason": reason})
        return f"FAILED: only {stock} units in stock, {line['quantity']} needed."
    transaction_id = create_transaction(
        item_name, "sales", line["quantity"], line["line_total"], sale_date)
    ORDER_STATE["sales"].append({"item_name": item_name, "status": "sold",
                                 "quantity": line["quantity"], "price": line["line_total"]})
    return (f"SOLD: {line['quantity']} {item_name} for ${line['line_total']:.2f} "
            f"(transaction {transaction_id}).")


@tool
def finalize_sale(item_name: str, request_date: Optional[str] = None) -> str:
    """Record the sale of one quoted line in the database, only if enough stock is available.
    The quantity and price are taken from the current quote; the date is the request date.

    Args:
        item_name: The exact catalog item name of a line in the current quote.
        request_date: Optional and ignored; the request date is already known to the system.

    Returns:
        A text confirming the sale, or explaining why it could not be completed.
    """
    with DB_LOCK:  # stock check and sale happen as one uninterrupted step
        return _finalize_sale_locked(item_name)


@tool
def get_financial_summary(request_date: Optional[str] = None) -> str:
    """Report the company's cash balance, inventory value and total assets on the request date
    (internal use only).

    Args:
        request_date: Optional and ignored; the request date is already known to the system.

    Returns:
        A JSON object with cash_balance, inventory_value and total_assets.
    """
    report = generate_financial_report(ORDER_STATE["request_date"])
    return json.dumps({
        "cash_balance": round(float(report["cash_balance"]), 2),
        "inventory_value": round(float(report["inventory_value"]), 2),
        "total_assets": round(float(report["total_assets"]), 2),
    })


# Set up your agents and create an orchestration agent that will manage them.

INVENTORY_TASK = """You are the inventory agent of a paper supply company.
Customer request: {request}

Steps:
1. Find the date the customer needs the items (needed_by, YYYY-MM-DD). If no deadline is given,
   use 7 days after the request date ('Date of request: YYYY-MM-DD').
2. Call list_catalog and check_inventory. Match every requested item to ONE exact catalog item_name.
   Only match the same kind of product (paper size words like A3/A4/A5 can be ignored, but tickets
   are not invitation cards and balloons are not paper). Convert quantities to units
   (1 ream = 500 sheets). Always use the FULL quantity the customer asked for; never reduce it.
3. For each matched item, call check_and_restock(item_name, requested, quantity, needed_by) once,
   where requested is the item as the customer wrote it. Do not retry with a smaller quantity.
4. For each item that matches nothing in the catalog, call record_unavailable_item(requested,
   "not an item we sell").
(You may use check_item_stock to look up a stock level, but only check_and_restock restocks.)

Call final_answer with one short sentence summarizing what you did."""

QUOTE_TASK = """You are the quoting agent of a paper supply company.
Customer request: {request}
Order lines that can be supplied: {lines}

Steps:
1. Call find_similar_quotes with one or two keywords (the main paper type or the event type)
   to see how similar orders were priced in the past.
2. Call calculate_quote (it takes no arguments; it prices exactly the items listed above).
3. Call final_answer with ONLY the JSON returned by calculate_quote, plus a field "explanation":
   one or two sentences about the bulk discount and how the price compares with past quotes.
   Never mention profit margins or internal company data."""

SALES_TASK = """You are the sales agent of a paper supply company.
Quote to finalize: {quote}
Sale date: {sale_date}

Steps:
1. For every line in the quote, call finalize_sale with that line's item_name.
2. Call get_financial_summary (internal health check).
3. Call final_answer with one short sentence summarizing which items were sold."""

REPLY_TASK = """You are the customer service orchestrator of the Munder Difflin Paper Company.
Customer request: {request}
Order outcome: {facts}

Write a short, friendly message to the customer (2 to 4 sentences) that:
- thanks them and says whether their order can be fully, partly or not at all delivered;
- if items are delivered, explains the price using "pricing_explanation" (the bulk discount reason
  and how it compares with similar past orders);
- if items cannot be supplied, apologizes briefly.
Do NOT list items, quantities, prices, totals or dates: an exact order summary is added below your
message automatically. Never use words like "order outcome", "system" or "inventory".
Do not reveal cash balances, stock levels, profit margins or internal details.
Do not sign the message. Call final_answer with the message."""


class InventoryAgent(ToolCallingAgent):
    """Worker agent: matches items to the catalog, checks stock and reorders from the supplier."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[list_catalog, check_inventory, check_item_stock, check_and_restock,
                   record_unavailable_item],
            model=model,
            name="inventory_agent",
            description="Checks stock for requested items and restocks them when possible.",
            max_steps=15,
            verbosity_level=0,
        )


class QuotingAgent(ToolCallingAgent):
    """Worker agent: prices the order with bulk discounts, informed by past quotes."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[find_similar_quotes, calculate_quote],
            model=model,
            name="quoting_agent",
            description="Creates price quotes with bulk discounts using past quotes as reference.",
            max_steps=6,
            verbosity_level=0,
        )


class SalesAgent(ToolCallingAgent):
    """Worker agent: records the sales in the database and checks the company's finances."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[finalize_sale, get_financial_summary],
            model=model,
            name="sales_agent",
            description="Finalizes sales transactions and reports the financial state.",
            max_steps=12,
            verbosity_level=0,
        )


class Orchestrator(ToolCallingAgent):
    """Orchestrator agent: runs inventory -> quoting -> sales and writes the customer reply."""

    def __init__(self, model: OpenAIServerModel):
        self.inventory_agent = InventoryAgent(model)
        self.quoting_agent = QuotingAgent(model)
        self.sales_agent = SalesAgent(model)
        super().__init__(
            tools=[],
            model=model,
            name="orchestrator",
            description="Delegates customer requests to the inventory, quoting and sales agents.",
            max_steps=3,
            verbosity_level=0,
        )

    def process_request(self, request: str) -> str:
        """Handle one customer request from start to finish and return the customer reply."""
        date_match = re.search(r"Date of request: (\d{4}-\d{2}-\d{2})", request)
        request_date = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")
        ORDER_STATE.update({"request_date": request_date, "inventory": {}, "unmatched": [],
                            "quote": {}, "sales": []})

        # Step 1 (sequential): match items to the catalog, check stock and restock when possible
        self.inventory_agent.run(INVENTORY_TASK.format(request=request))
        available = [{"item_name": name, "quantity": entry["quantity"]}
                     for name, entry in ORDER_STATE["inventory"].items() if entry["available"]]

        # Step 2 (conditional): only quote, and then sell, if at least one item can be supplied
        if available:
            quote_answer = parse_json(self.quoting_agent.run(
                QUOTE_TASK.format(request=request, lines=json.dumps(available))))
            if ORDER_STATE["quote"]:
                ORDER_STATE["quote"]["history_note"] = quote_answer.get("explanation", "")
            if ORDER_STATE["quote"].get("lines"):
                self.sales_agent.run(SALES_TASK.format(
                    quote=json.dumps(ORDER_STATE["quote"]), sale_date=request_date))

        # Step 3: the orchestrator writes the customer reply from the recorded facts only
        facts = self.build_order_facts()
        message = str(self.run(REPLY_TASK.format(request=request, facts=json.dumps(facts))))
        return f"{message.strip()}\n\n{self.format_order_summary(facts)}\n\nMunder Difflin Paper Company"

    @staticmethod
    def format_order_summary(facts: dict) -> str:
        """Write the exact order summary for the customer from the recorded facts (no LLM involved)."""
        lines = ["Order summary:"]
        for item in facts["delivered_items"]:
            lines.append(f"- {item['item']}: {item['quantity']} units, ${item['price']:.2f}")
        if facts["delivered_items"]:
            lines.append(f"Bulk discount: {int(facts['bulk_discount_rate'] * 100)}% "
                         f"({facts['pricing_explanation']['discount']})")
            lines.append(f"Order total: ${facts['order_total']:.2f}")
            lines.append(f"Delivery by: {facts['delivery_date']}")
        else:
            lines.append("- No items can be delivered for this request.")
        if facts["not_supplied"]:
            lines.append("Not supplied:")
            for item in facts["not_supplied"]:
                quantity = f" ({item['quantity']} units)" if item.get("quantity") else ""
                lines.append(f"- {item['item']}{quantity}: {item['reason']}")
        return "\n".join(lines)

    @staticmethod
    def build_order_facts() -> dict:
        """Combine the recorded inventory results, quote and sales into the facts for the reply."""
        inventory = ORDER_STATE["inventory"]
        ready_dates = {name: entry["ready_date"] for name, entry in inventory.items()}
        sold = [sale for sale in ORDER_STATE["sales"] if sale["status"] == "sold"]
        sold_names = {sale["item_name"] for sale in sold}
        failed = {s["item_name"]: s["reason"] for s in ORDER_STATE["sales"] if s["status"] == "failed"}
        matched_requests = {key for entry in inventory.values() for key in entry["lines"]}
        not_supplied = [{"item": item["requested"], "reason": item["reason"]}
                        for item in ORDER_STATE["unmatched"]
                        if item["requested"].strip().lower() not in matched_requests]
        for name, entry in inventory.items():
            if name in sold_names:
                continue
            if not entry["available"]:
                reason = entry["reason"]
            else:
                reason = failed.get(name, "the order for this item could not be completed")
            not_supplied.append({"item": name, "quantity": entry["quantity"], "reason": reason})
        return {
            "delivered_items": [{"item": s["item_name"], "quantity": s["quantity"],
                                 "price": s["price"], "ready_date": ready_dates.get(s["item_name"])}
                                for s in sold],
            "bulk_discount_rate": ORDER_STATE["quote"].get("discount_rate", 0) if sold else 0,
            "pricing_explanation": {
                "discount": ORDER_STATE["quote"].get("discount_reason", "") if sold else "",
                "compared_with_past_quotes": ORDER_STATE["quote"].get("history_note", "") if sold else "",
            },
            "order_total": round(sum(s["price"] for s in sold), 2),
            "delivery_date": max((ready_dates.get(s["item_name"]) or "" for s in sold), default=None),
            "not_supplied": not_supplied,
        }


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    orchestrator = Orchestrator(model)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        try:
            response = orchestrator.process_request(request_with_date)
        except Exception as e:
            # Log the technical error internally; give the customer a polite message only
            print(f"ERROR while processing request {idx+1}: {e}")
            response = ("We're sorry, we could not process your request right now. "
                        "Our team will contact you shortly.")

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
