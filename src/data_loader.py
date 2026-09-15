"""
Data loader: reads raw TWCS dataset and filters for the selected brand.
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_PATH, BRAND_NAME


def _clean_id(series):
    """Clean ID column converting float strings like '1234.0' to '1234'."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r'\.0$', '', regex=True)
    return s.replace({'nan': '', 'None': '', '<NA>': '', 'NaN': ''})


def load_raw_dataset(path=None):
    """Load the full TWCS dataset from CSV."""
    path = path or RAW_DATA_PATH
    df = pd.read_csv(path, dtype=str)
    
    # Clean ID columns
    df['tweet_id'] = _clean_id(df['tweet_id'])
    df['response_tweet_id'] = _clean_id(df['response_tweet_id'])
    df['in_response_to_tweet_id'] = _clean_id(df['in_response_to_tweet_id'])
    
    # Boolean and text cleaning
    df['inbound'] = df['inbound'].astype(str).str.lower().map({'true': True, '1': True, 't': True}).fillna(False)
    df['text'] = df['text'].fillna('')
    df['author_id'] = df['author_id'].fillna('').astype(str)
    return df


def get_brand_stats(df):
    """Get message counts per brand (outbound only)."""
    outbound = df[df['inbound'] == False]
    return outbound['author_id'].value_counts().to_dict()


def filter_brand_conversations(df, brand_name=None):
    """
    Filter dataset to only include conversations involving the selected brand.
    Returns both inbound (customer) and outbound (brand) messages.
    """
    brand_name = brand_name or BRAND_NAME

    # Get all outbound messages from this brand
    brand_outbound = df[
        (df['inbound'] == False) & (df['author_id'] == brand_name)
    ]

    # Get tweet IDs that the brand responded to
    customer_tweet_ids = set(
        brand_outbound['in_response_to_tweet_id'].dropna().unique()
    ) - {'', 'nan'}

    # Get those customer tweets
    brand_inbound = df[df['tweet_id'].isin(customer_tweet_ids)]

    # Also get customer tweets that brand outbound messages are responding to
    brand_response_ids = set(
        brand_outbound['response_tweet_id'].str.split(',').explode().dropna().unique()
    ) - {'', 'nan'}

    # Combine: all outbound from brand + all inbound they responded to + follow-ups
    all_relevant_ids = (
        set(brand_outbound['tweet_id'].unique()) |
        customer_tweet_ids |
        brand_response_ids
    )

    brand_df = df[df['tweet_id'].isin(all_relevant_ids)].copy()

    return brand_df


def build_conversation_pairs(df, brand_name=None):
    """
    Build customer_message → brand_response pairs from the filtered dataset.
    Each pair represents a single exchange: customer asks, brand responds.

    Returns a list of dicts:
    {
        'customer_tweet_id': str,
        'customer_message': str,
        'customer_author_id': str,
        'brand_tweet_id': str,
        'brand_response': str,
        'created_at': str  (of brand response)
    }
    """
    brand_name = brand_name or BRAND_NAME

    # Brand outbound messages
    brand_msgs = df[
        (df['inbound'] == False) & (df['author_id'] == brand_name)
    ].copy()

    # Build lookup: tweet_id → row
    tweet_lookup = df.set_index('tweet_id')

    pairs = []
    for _, brand_row in brand_msgs.iterrows():
        cust_id = brand_row['in_response_to_tweet_id']
        if not cust_id or cust_id == 'nan' or cust_id == '':
            continue

        if cust_id not in tweet_lookup.index:
            continue

        cust_row = tweet_lookup.loc[cust_id]
        # Handle case where cust_id maps to multiple rows
        if isinstance(cust_row, pd.DataFrame):
            cust_row = cust_row.iloc[0]

        # Skip if "customer" is actually the brand (self-replies)
        if cust_row.get('author_id', '') == brand_name:
            continue

        pairs.append({
            'customer_tweet_id': str(cust_id),
            'customer_message': str(cust_row.get('text', '')),
            'customer_author_id': str(cust_row.get('author_id', '')),
            'brand_tweet_id': str(brand_row['tweet_id']),
            'brand_response': str(brand_row['text']),
            'created_at': str(brand_row.get('created_at', '')),
        })

    return pairs


def build_multi_turn_conversations(df, brand_name=None):
    """
    Reconstruct multi-turn conversations by chaining
    in_response_to_tweet_id → tweet_id links.

    Returns list of conversations, each being a list of messages in order:
    [
        {'role': 'customer'|'brand', 'text': ..., 'tweet_id': ...},
        ...
    ]
    """
    brand_name = brand_name or BRAND_NAME

    tweet_lookup = {}
    for _, row in df.iterrows():
        tweet_lookup[str(row['tweet_id'])] = row

    # Build reply chains: child → parent
    child_to_parent = {}
    parent_to_children = {}
    for _, row in df.iterrows():
        tid = str(row['tweet_id'])
        parent_id = str(row['in_response_to_tweet_id'])
        if parent_id and parent_id != 'nan' and parent_id != '':
            child_to_parent[tid] = parent_id
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            parent_to_children[parent_id].append(tid)

    # Find conversation roots (messages with no parent in the dataset)
    all_ids = set(tweet_lookup.keys())
    roots = set()
    for tid in all_ids:
        # Walk up to find root
        current = tid
        visited = set()
        while current in child_to_parent and current not in visited:
            visited.add(current)
            current = child_to_parent[current]
        roots.add(current)

    # Build conversations from roots via BFS
    conversations = []
    for root_id in roots:
        if root_id not in tweet_lookup:
            continue
        conv = []
        queue = [root_id]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited or current not in tweet_lookup:
                continue
            visited.add(current)
            row = tweet_lookup[current]
            role = 'brand' if (not row['inbound'] and str(row['author_id']) == brand_name) else 'customer'
            conv.append({
                'role': role,
                'text': str(row.get('text', '')),
                'tweet_id': current,
                'author_id': str(row.get('author_id', '')),
            })
            # Add children
            if current in parent_to_children:
                queue.extend(parent_to_children[current])

        if len(conv) >= 2:  # At least one exchange
            conversations.append(conv)

    return conversations


if __name__ == "__main__":
    print(f"Loading raw dataset from {RAW_DATA_PATH}...")
    df = load_raw_dataset()
    print(f"Total rows: {len(df)}")

    stats = get_brand_stats(df)
    print(f"\nTop 10 brands by outbound messages:")
    for brand, count in sorted(stats.items(), key=lambda x: -x[1])[:10]:
        print(f"  {brand}: {count}")

    print(f"\nFiltering for {BRAND_NAME}...")
    brand_df = filter_brand_conversations(df)
    print(f"Brand-related messages: {len(brand_df)}")

    pairs = build_conversation_pairs(df)
    print(f"Customer→Brand pairs: {len(pairs)}")
