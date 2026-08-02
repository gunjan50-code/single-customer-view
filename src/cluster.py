"""
Step 7 of the pipeline: turn pairwise decisions into people, then into
golden records.

Two separate jobs live here.

CLUSTERING. We have a pile of "record A is record B" decisions. What we need
is groups. Treat every record as a node and every auto-merged pair as an
edge, and the connected components of that graph are our entities.

The catch is that matching is not transitive. The model may say A=B and B=C
while explicitly rejecting A=C. Connected components merges all three
regardless. Sometimes that is right -- A and C really are the same person and
we simply failed to see it directly. Sometimes it is catastrophic: one bad
edge chains hundreds of unrelated people into a single blob. That failure has
a name in production systems, a runaway cluster, and it is the reason nobody
ships raw connected components without guards.

We measure how often transitivity is violated, cap cluster size, and split
clusters whose internal agreement is too weak to believe.

SURVIVORSHIP. The cluster says these five records are one person. Now which
name do we keep? Which address? Each field gets its own rule, and every
winning value keeps a pointer back to the record it came from. That lineage
is not decoration -- an auditor asking "where did this address come from?"
will not accept "the system merged it".

Run:  python -m src.cluster
"""

import sqlite3
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd

import config
from src.load import load_answer_key, load_standardized

# Which source systems to trust, highest first. Used to break ties on fields
# where recency is not the right rule.
TRUST = {name: settings["trust"] for name, settings in config.SOURCE_SYSTEMS.items()}


# ------------------------------------------------------------- clustering

def build_graph(decisions: pd.DataFrame) -> nx.Graph:
    """One node per record, one edge per auto-merged pair."""
    merged = decisions[decisions.band == "auto_merge"]
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        zip(merged.record_id_a, merged.record_id_b, merged.score)
    )
    return graph


def cohesion(graph: nx.Graph, nodes: set) -> float:
    """Average score of the edges actually present inside a cluster.

    A tight cluster of the same person has many high-scoring internal edges.
    A cluster held together by one weak bridge does not, and that is the
    signature of a runaway.
    """
    subgraph = graph.subgraph(nodes)
    if subgraph.number_of_edges() == 0:
        return 0.0
    return sum(d["weight"] for _, _, d in subgraph.edges(data=True)) / subgraph.number_of_edges()


def split_weak_cluster(graph: nx.Graph, nodes: set) -> list[set]:
    """Break a suspicious cluster at its weakest links.

    Rather than throwing the cluster away, we repeatedly remove the
    lowest-scoring edge until the component falls apart, which tends to cut
    exactly the accidental bridge that fused two real people together.
    """
    subgraph = graph.subgraph(nodes).copy()
    while subgraph.number_of_edges() > 0:
        components = list(nx.connected_components(subgraph))
        if len(components) > 1:
            result = []
            for component in components:
                if len(component) > config.MAX_CLUSTER_SIZE:
                    result.extend(split_weak_cluster(subgraph, component))
                else:
                    result.append(component)
            return result
        weakest = min(subgraph.edges(data=True), key=lambda e: e[2]["weight"])
        subgraph.remove_edge(weakest[0], weakest[1])
    return [{n} for n in nodes]


def find_clusters(graph: nx.Graph, all_record_ids: list[str]) -> tuple[dict, dict]:
    """Connected components, with guards. Returns (record -> entity, stats)."""
    stats = {"runaway_split": 0, "weak_split": 0, "raw_components": 0}
    clusters: list[set] = []

    for component in nx.connected_components(graph):
        stats["raw_components"] += 1
        if len(component) > config.MAX_CLUSTER_SIZE:
            stats["runaway_split"] += 1
            clusters.extend(split_weak_cluster(graph, component))
        elif len(component) > 2 and cohesion(graph, component) < config.MIN_CLUSTER_COHESION:
            stats["weak_split"] += 1
            clusters.extend(split_weak_cluster(graph, component))
        else:
            clusters.append(component)

    entity_of_record: dict[str, int] = {}
    for entity_id, nodes in enumerate(clusters, start=1):
        for record_id in nodes:
            entity_of_record[record_id] = entity_id

    # Records with no edges at all are their own entity -- a customer who
    # appears exactly once is still a customer.
    next_id = len(clusters) + 1
    for record_id in all_record_ids:
        if record_id not in entity_of_record:
            entity_of_record[record_id] = next_id
            next_id += 1

    stats["final_clusters"] = next_id - 1
    return entity_of_record, stats


def measure_transitivity(decisions: pd.DataFrame, entity_of_record: dict) -> dict:
    """How often did clustering assert something the model explicitly denied?"""
    rejected = decisions[decisions.band == "auto_reject"]
    same_entity = [
        entity_of_record.get(a) == entity_of_record.get(b)
        for a, b in zip(rejected.record_id_a, rejected.record_id_b)
    ]
    contradictions = int(sum(same_entity))
    return {
        "rejected_pairs": len(rejected),
        "contradicted_by_transitivity": contradictions,
        "rate": contradictions / max(len(rejected), 1),
    }


# ----------------------------------------------------------- survivorship

def pick_most_recent(rows: pd.DataFrame, field: str) -> tuple[str, str]:
    """Latest non-empty value. Right for anything that changes over time."""
    candidates = rows[rows[field].astype(str).str.len() > 0]
    if candidates.empty:
        return "", ""
    winner = candidates.sort_values("created_date", ascending=False).iloc[0]
    return winner[field], winner["record_id"]


def pick_most_frequent(rows: pd.DataFrame, field: str) -> tuple[str, str]:
    """Most common non-empty value. Right for facts that should never change.

    A mistyped date of birth appears once; the true one appears three times.
    Recency would happily pick the typo if it arrived last.
    """
    values = [v for v in rows[field].astype(str) if v]
    if not values:
        return "", ""
    winner = Counter(values).most_common(1)[0][0]
    source = rows[rows[field].astype(str) == winner].iloc[0]["record_id"]
    return winner, source


def pick_longest(rows: pd.DataFrame, field: str) -> tuple[str, str]:
    """Most complete value. 'Rajesh Kumar Sharma' beats 'R. K. Sharma'."""
    candidates = rows[rows[field].astype(str).str.len() > 0]
    if candidates.empty:
        return "", ""
    winner = candidates.loc[candidates[field].astype(str).str.len().idxmax()]
    return winner[field], winner["record_id"]


def pick_most_trusted(rows: pd.DataFrame, field: str) -> tuple[str, str]:
    """Value from the highest-trust source system that has one."""
    candidates = rows[rows[field].astype(str).str.len() > 0].copy()
    if candidates.empty:
        return "", ""
    candidates["_trust"] = candidates.source_system.map(TRUST).fillna(0)
    winner = candidates.sort_values(["_trust", "created_date"],
                                    ascending=[False, False]).iloc[0]
    return winner[field], winner["record_id"]


# Each field, the rule that governs it, and why.
SURVIVORSHIP_RULES = {
    "first_name":   (pick_longest,       "most complete form wins over initials"),
    "last_name":    (pick_longest,       "most complete form wins over initials"),
    "dob_clean":    (pick_most_frequent, "a typo appears once, the truth repeats"),
    "email_clean":  (pick_most_trusted,  "self-entered beats agent-typed"),
    "phone_clean":  (pick_most_recent,   "people change numbers"),
    "address_clean": (pick_most_recent,  "people move"),
    "city_clean":   (pick_most_recent,   "people move"),
    "pincode_clean": (pick_most_recent,  "people move"),
}


def build_golden_records(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse each entity into one record, keeping field-level lineage."""
    golden_rows, lineage_rows = [], []

    for entity_id, rows in records.groupby("entity_id", sort=False):
        golden = {"entity_id": entity_id, "n_source_records": len(rows),
                  "source_systems": "|".join(sorted(set(rows.source_system)))}

        # Roughly three quarters of entities are a single record, and there is
        # nothing for survivorship to arbitrate. Short-circuiting them skips
        # eight rule evaluations and eight dataframe scans per entity, which
        # is where almost all the runtime of this stage was going.
        if len(rows) == 1:
            only = rows.iloc[0]
            source_record, system = only.record_id, only.source_system
            for field in SURVIVORSHIP_RULES:
                value = only[field]
                golden[field] = value
                if value:
                    lineage_rows.append({
                        "entity_id": entity_id,
                        "field": field,
                        "value": value,
                        "source_record_id": source_record,
                        "source_system": system,
                        "rule": "single_source",
                    })
            golden_rows.append(golden)
            continue

        system_of_record = dict(zip(rows.record_id, rows.source_system))
        for field, (rule, _reason) in SURVIVORSHIP_RULES.items():
            value, source_record = rule(rows, field)
            golden[field] = value
            if value:
                lineage_rows.append({
                    "entity_id": entity_id,
                    "field": field,
                    "value": value,
                    "source_record_id": source_record,
                    "source_system": system_of_record.get(source_record, ""),
                    "rule": rule.__name__,
                })
        golden_rows.append(golden)

    return pd.DataFrame(golden_rows), pd.DataFrame(lineage_rows)


# ------------------------------------------------------------- evaluation

def evaluate_clusters(records: pd.DataFrame, answer_key: pd.DataFrame,
                      test_people: set) -> None:
    """Compare discovered entities against the truth, at cluster level."""
    truth = dict(zip(answer_key.record_id, answer_key.person_id))
    frame = records.copy()
    frame["person_id"] = frame.record_id.map(truth)

    for label, subset in [
        ("HELD-OUT PEOPLE (the honest number)", frame[frame.person_id.isin(test_people)]),
        ("ALL RECORDS", frame),
    ]:
        true_groups = subset.groupby("person_id").record_id.apply(frozenset)
        found_groups = subset.groupby("entity_id").record_id.apply(frozenset)
        found_set = set(found_groups)
        exact = sum(1 for g in true_groups if g in found_set)

        # Pairwise scoring: of the pairs we put together, how many belong
        # together, and of the pairs that belong together, how many did we find?
        def pairs_of(groups):
            out = set()
            for group in groups:
                members = sorted(group)
                for i, a in enumerate(members):
                    for b in members[i + 1:]:
                        out.add((a, b))
            return out

        true_pairs = pairs_of(true_groups)
        found_pairs = pairs_of(found_groups)
        overlap = len(true_pairs & found_pairs)
        precision = overlap / len(found_pairs) if found_pairs else 0.0
        recall = overlap / len(true_pairs) if true_pairs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        print(f"\n  {label}")
        print(f"    Records                    {len(subset):>10,}")
        print(f"    Real people                {len(true_groups):>10,}")
        print(f"    Entities discovered        {len(found_groups):>10,}")
        print(f"    Perfectly reconstructed    {exact:>10,}  "
              f"({exact / len(true_groups):.1%} of people, every record and no extras)")
        print(f"    Pairwise precision         {precision:>10.4f}")
        print(f"    Pairwise recall            {recall:>10.4f}")
        print(f"    Pairwise F1                {f1:>10.4f}")


def write_database(golden: pd.DataFrame, lineage: pd.DataFrame,
                   members: pd.DataFrame) -> None:
    """Publish to SQLite -- the single source of truth downstream systems read."""
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    with sqlite3.connect(config.DB_PATH) as connection:
        golden.to_sql("golden_records", connection, index=False)
        lineage.to_sql("record_lineage", connection, index=False)
        members.to_sql("cluster_members", connection, index=False)
        connection.execute(
            "CREATE INDEX idx_lineage_entity ON record_lineage(entity_id)")
        connection.execute(
            "CREATE INDEX idx_members_entity ON cluster_members(entity_id)")
        connection.execute(
            "CREATE INDEX idx_members_record ON cluster_members(record_id)")


def main() -> None:
    decisions = pd.read_csv(config.DATA_PROCESSED / "decisions.csv")
    records = load_standardized()
    answer_key = load_answer_key()
    test_people = set(pd.read_csv(config.DATA_PROCESSED / "test_people.csv").person_id)

    print(f"Clustering {len(records):,} records from "
          f"{int((decisions.band == 'auto_merge').sum()):,} auto-merged pairs...")

    graph = build_graph(decisions)
    entity_of_record, stats = find_clusters(graph, records.record_id.tolist())
    records["entity_id"] = records.record_id.map(entity_of_record)

    print("\n" + "=" * 74)
    print("CLUSTERING")
    print("=" * 74)
    print(f"  Raw connected components       {stats['raw_components']:>10,}")
    print(f"  Split for exceeding size cap   {stats['runaway_split']:>10,}  "
          f"(runaway guard, cap {config.MAX_CLUSTER_SIZE})")
    print(f"  Split for weak cohesion        {stats['weak_split']:>10,}  "
          f"(below {config.MIN_CLUSTER_COHESION})")
    print(f"  Final entities                 {stats['final_clusters']:>10,}")

    sizes = records.groupby("entity_id").size()
    print(f"  Largest entity                 {sizes.max():>10,} records")
    print(f"  Singletons                     {int((sizes == 1).sum()):>10,} "
          f"({(sizes == 1).mean():.1%})")

    transitivity = measure_transitivity(decisions, entity_of_record)
    print(f"\n  Transitivity contradictions    "
          f"{transitivity['contradicted_by_transitivity']:>10,} "
          f"({transitivity['rate']:.3%} of auto-rejected pairs)")
    print("  These are pairs the model said were different people, but which")
    print("  clustering fused anyway via a chain of other matches. A non-zero")
    print("  rate is expected and healthy; a large one means runaway clusters.")

    print("\n" + "=" * 74)
    print("ENTITY RESOLUTION QUALITY")
    print("=" * 74)
    evaluate_clusters(records, answer_key, test_people)

    print("\n" + "=" * 74)
    print("GOLDEN RECORDS")
    print("=" * 74)
    golden, lineage = build_golden_records(records)
    members = records[["entity_id", "record_id", "source_system"]]
    write_database(golden, lineage, members)

    print(f"  {len(records):,} raw records collapsed into {len(golden):,} golden records")
    print(f"  {1 - len(golden) / len(records):.1%} of the customer file was redundant")
    print(f"  {len(lineage):,} field-level lineage entries recorded")
    print(f"\n  Survivorship rules applied:")
    for field, (rule, reason) in SURVIVORSHIP_RULES.items():
        print(f"    {field:<16} {rule.__name__:<20} {reason}")

    multi = golden[golden.n_source_records > 1]
    print(f"\n  {len(multi):,} golden records were assembled from 2+ systems")
    print(f"  Written to {config.DB_PATH.name} "
          f"(golden_records, record_lineage, cluster_members)")


if __name__ == "__main__":
    main()
