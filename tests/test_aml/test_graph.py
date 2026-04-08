"""
tests/test_aml/test_graph.py
==============================
Unit tests for the AML transaction network graph builder.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import networkx as nx
import pandas as pd
import pytest

from fincrime_ml.aml.graph import (
    GraphStats,
    NodeFeatures,
    TransactionGraphBuilder,
)
from fincrime_ml.core.data.synth_aml import SyntheticAMLGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def aml_df() -> pd.DataFrame:
    gen = SyntheticAMLGenerator(n_accounts=200, seed=7)
    return gen.generate(n_transactions=1_500, suspicious_rate=0.08)


@pytest.fixture(scope="module")
def builder(aml_df) -> TransactionGraphBuilder:
    b = TransactionGraphBuilder()
    b.build(aml_df)
    return b


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Minimal 4-node, 3-edge graph: A->B->C, A->D."""
    t0 = datetime(2024, 1, 1)
    return pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "sender_account_id": "A",
                "receiver_account_id": "B",
                "amount_gbp": 1_000.0,
                "timestamp": t0,
                "is_mule_sender": 0,
                "is_mule_receiver": 1,
            },
            {
                "transaction_id": "T2",
                "sender_account_id": "B",
                "receiver_account_id": "C",
                "amount_gbp": 900.0,
                "timestamp": t0 + timedelta(hours=2),
                "is_mule_sender": 1,
                "is_mule_receiver": 0,
            },
            {
                "transaction_id": "T3",
                "sender_account_id": "A",
                "receiver_account_id": "D",
                "amount_gbp": 500.0,
                "timestamp": t0 + timedelta(hours=1),
                "is_mule_sender": 0,
                "is_mule_receiver": 0,
            },
        ]
    )


@pytest.fixture
def multi_edge_df() -> pd.DataFrame:
    """Same sender/receiver pair across multiple transactions."""
    t0 = datetime(2024, 2, 1)
    rows = []
    for i in range(5):
        rows.append(
            {
                "transaction_id": f"M-{i}",
                "sender_account_id": "X",
                "receiver_account_id": "Y",
                "amount_gbp": 200.0 * (i + 1),
                "timestamp": t0 + timedelta(hours=i),
                "is_mule_sender": 0,
                "is_mule_receiver": 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GraphStats dataclass tests
# ---------------------------------------------------------------------------


def test_graph_stats_fields():
    gs = GraphStats(
        n_nodes=10,
        n_edges=8,
        total_volume_gbp=50_000.0,
        density=0.09,
        n_weakly_connected_components=2,
        n_strongly_connected_components=8,
        max_in_degree=4,
        max_out_degree=3,
        top_betweenness_node="ACC-001",
        top_pagerank_node="ACC-002",
    )
    assert gs.n_nodes == 10
    assert gs.density == 0.09
    assert gs.metadata == {}


# ---------------------------------------------------------------------------
# NodeFeatures dataclass tests
# ---------------------------------------------------------------------------


def test_node_features_fields():
    nf = NodeFeatures(
        node_id="ACC-001",
        in_degree=3,
        out_degree=2,
        total_inflow_gbp=9_000.0,
        total_outflow_gbp=8_000.0,
        net_position_gbp=1_000.0,
        pass_through_ratio=0.89,
        betweenness_centrality=0.15,
        pagerank=0.08,
        in_txn_count=5,
        out_txn_count=4,
    )
    assert nf.node_id == "ACC-001"
    assert nf.is_mule is False


# ---------------------------------------------------------------------------
# TransactionGraphBuilder.build() tests
# ---------------------------------------------------------------------------


def test_build_returns_self(simple_df):
    b = TransactionGraphBuilder()
    result = b.build(simple_df)
    assert result is b


def test_build_creates_graph(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    assert b.graph is not None
    assert isinstance(b.graph, nx.DiGraph)


def test_build_node_count(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    # A, B, C, D = 4 nodes
    assert b.graph.number_of_nodes() == 4


def test_build_edge_count(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    # A->B, B->C, A->D = 3 edges
    assert b.graph.number_of_edges() == 3


def test_build_edge_weight_correct(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    # A->B weight = 1000
    assert b.graph["A"]["B"]["weight"] == 1_000.0


def test_build_multi_edge_aggregated(multi_edge_df):
    b = TransactionGraphBuilder()
    b.build(multi_edge_df)
    # 5 transactions X->Y, should be 1 edge with aggregated weight
    assert b.graph.number_of_edges() == 1
    assert b.graph["X"]["Y"]["weight"] == pytest.approx(3_000.0)  # 200+400+600+800+1000
    assert b.graph["X"]["Y"]["txn_count"] == 5


def test_build_edge_txn_count(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    assert b.graph["A"]["B"]["txn_count"] == 1


def test_build_mule_labels_applied(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    # B is mule_receiver (T1) and mule_sender (T2)
    assert b.graph.nodes["B"]["is_mule"] is True
    assert b.graph.nodes["D"]["is_mule"] is False


def test_build_node_inflow_set(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    # B receives 1000 from A
    assert b.graph.nodes["B"]["total_inflow"] == pytest.approx(1_000.0)


def test_build_node_outflow_set(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    # A sends 1000 + 500 = 1500
    assert b.graph.nodes["A"]["total_outflow"] == pytest.approx(1_500.0)


def test_build_min_edge_weight_filter(multi_edge_df):
    """Edges below min_edge_weight should be excluded."""
    b = TransactionGraphBuilder()
    b.build(multi_edge_df, min_edge_weight=5_000.0)
    # Aggregate is 3000, below 5000 => no edges
    assert b.graph.number_of_edges() == 0


def test_build_missing_column_raises(simple_df):
    df = simple_df.drop(columns=["amount_gbp"])
    b = TransactionGraphBuilder()
    with pytest.raises(KeyError, match="amount_gbp"):
        b.build(df)


def test_build_on_aml_data(aml_df):
    b = TransactionGraphBuilder()
    b.build(aml_df)
    assert b.graph.number_of_nodes() > 0
    assert b.graph.number_of_edges() > 0


def test_build_clears_cached_metrics(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    _ = b.node_features()  # populates cache
    b.build(simple_df)  # rebuild should clear cache
    assert b._betweenness is None
    assert b._pagerank is None


# ---------------------------------------------------------------------------
# graph_stats() tests
# ---------------------------------------------------------------------------


def test_graph_stats_returns_dataclass(builder):
    stats = builder.graph_stats()
    assert isinstance(stats, GraphStats)


def test_graph_stats_node_count(builder, aml_df):
    stats = builder.graph_stats()
    assert stats.n_nodes > 0


def test_graph_stats_edge_count(builder):
    stats = builder.graph_stats()
    assert stats.n_edges > 0


def test_graph_stats_density_in_range(builder):
    stats = builder.graph_stats()
    assert 0.0 <= stats.density <= 1.0


def test_graph_stats_volume_positive(builder):
    stats = builder.graph_stats()
    assert stats.total_volume_gbp > 0.0


def test_graph_stats_wcc_positive(builder):
    stats = builder.graph_stats()
    assert stats.n_weakly_connected_components >= 1


def test_graph_stats_top_betweenness_is_string(builder):
    stats = builder.graph_stats()
    assert isinstance(stats.top_betweenness_node, str)


def test_graph_stats_top_pagerank_is_string(builder):
    stats = builder.graph_stats()
    assert isinstance(stats.top_pagerank_node, str)


def test_graph_stats_before_build_raises():
    b = TransactionGraphBuilder()
    with pytest.raises(RuntimeError, match="build()"):
        b.graph_stats()


def test_graph_stats_simple(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    stats = b.graph_stats()
    assert stats.n_nodes == 4
    assert stats.n_edges == 3
    assert stats.total_volume_gbp == pytest.approx(2_400.0)


# ---------------------------------------------------------------------------
# node_features() tests
# ---------------------------------------------------------------------------


def test_node_features_returns_dataframe(builder):
    df = builder.node_features()
    assert isinstance(df, pd.DataFrame)


def test_node_features_row_count(builder):
    df = builder.node_features()
    assert len(df) == builder.graph.number_of_nodes()


def test_node_features_columns(builder):
    df = builder.node_features()
    expected = [
        "node_id",
        "in_degree",
        "out_degree",
        "total_inflow_gbp",
        "total_outflow_gbp",
        "net_position_gbp",
        "pass_through_ratio",
        "betweenness_centrality",
        "pagerank",
        "in_txn_count",
        "out_txn_count",
        "is_mule",
    ]
    for col in expected:
        assert col in df.columns


def test_node_features_betweenness_in_range(builder):
    df = builder.node_features()
    assert (df["betweenness_centrality"] >= 0).all()
    assert (df["betweenness_centrality"] <= 1).all()


def test_node_features_pagerank_sums_to_one(builder):
    df = builder.node_features()
    assert df["pagerank"].sum() == pytest.approx(1.0, abs=1e-4)


def test_node_features_pass_through_non_negative(builder):
    df = builder.node_features()
    assert (df["pass_through_ratio"] >= 0).all()


def test_node_features_mule_column_is_bool(builder):
    df = builder.node_features()
    assert df["is_mule"].dtype == bool


def test_node_features_sorted_by_betweenness(builder):
    df = builder.node_features()
    assert df["betweenness_centrality"].is_monotonic_decreasing


def test_node_features_before_build_raises():
    b = TransactionGraphBuilder()
    with pytest.raises(RuntimeError):
        b.node_features()


def test_node_features_pass_through_simple(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    df = b.node_features()
    node_b = df[df["node_id"] == "B"].iloc[0]
    # B receives 1000, sends 900 => pass_through = 0.9
    assert node_b["pass_through_ratio"] == pytest.approx(0.9, abs=0.01)


# ---------------------------------------------------------------------------
# edge_dataframe() tests
# ---------------------------------------------------------------------------


def test_edge_dataframe_returns_dataframe(builder):
    df = builder.edge_dataframe()
    assert isinstance(df, pd.DataFrame)


def test_edge_dataframe_row_count(builder):
    df = builder.edge_dataframe()
    assert len(df) == builder.graph.number_of_edges()


def test_edge_dataframe_columns(builder):
    df = builder.edge_dataframe()
    for col in ("source", "target", "weight", "txn_count", "mean_amount", "first_txn", "last_txn"):
        assert col in df.columns


def test_edge_dataframe_weight_positive(builder):
    df = builder.edge_dataframe()
    assert (df["weight"] > 0).all()


def test_edge_dataframe_sorted_by_weight(builder):
    df = builder.edge_dataframe()
    assert df["weight"].is_monotonic_decreasing


def test_edge_dataframe_txn_count_positive(builder):
    df = builder.edge_dataframe()
    assert (df["txn_count"] > 0).all()


def test_edge_dataframe_before_build_raises():
    b = TransactionGraphBuilder()
    with pytest.raises(RuntimeError):
        b.edge_dataframe()


def test_edge_dataframe_multi_edge(multi_edge_df):
    b = TransactionGraphBuilder()
    b.build(multi_edge_df)
    df = b.edge_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["txn_count"] == 5


# ---------------------------------------------------------------------------
# subgraph() tests
# ---------------------------------------------------------------------------


def test_subgraph_returns_digraph(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    sg = b.subgraph("A", radius=1)
    assert isinstance(sg, nx.DiGraph)


def test_subgraph_contains_focal_node(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    sg = b.subgraph("A", radius=1)
    assert "A" in sg.nodes


def test_subgraph_radius_1_includes_neighbours(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    sg = b.subgraph("A", radius=1)
    # A->B and A->D, so B and D should be in radius-1 subgraph
    assert "B" in sg.nodes
    assert "D" in sg.nodes


def test_subgraph_invalid_node_raises(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    with pytest.raises(KeyError, match="not found"):
        b.subgraph("NONEXISTENT")


def test_subgraph_before_build_raises():
    b = TransactionGraphBuilder()
    with pytest.raises(RuntimeError):
        b.subgraph("A")


# ---------------------------------------------------------------------------
# flag_high_risk_nodes() tests
# ---------------------------------------------------------------------------


def test_flag_high_risk_nodes_returns_dataframe(builder):
    df = builder.flag_high_risk_nodes()
    assert isinstance(df, pd.DataFrame)


def test_flag_high_risk_nodes_columns(builder):
    df = builder.flag_high_risk_nodes()
    for col in (
        "node_id",
        "flag_reason",
        "betweenness_centrality",
        "pagerank",
        "pass_through_ratio",
        "is_mule",
    ):
        assert col in df.columns


def test_flag_high_risk_nodes_flag_reason_non_empty(builder):
    df = builder.flag_high_risk_nodes()
    if len(df) > 0:
        assert (df["flag_reason"].str.len() > 0).all()


def test_flag_high_risk_nodes_pass_through_flagged(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    # B has pass_through = 0.9 which exceeds threshold of 0.8
    df = b.flag_high_risk_nodes(pass_through_threshold=0.80)
    flagged_ids = df["node_id"].tolist()
    assert "B" in flagged_ids


def test_flag_high_risk_nodes_before_build_raises():
    b = TransactionGraphBuilder()
    with pytest.raises(RuntimeError):
        b.flag_high_risk_nodes()


# ---------------------------------------------------------------------------
# find_mule_chains() tests
# ---------------------------------------------------------------------------


def test_find_mule_chains_returns_list(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    chains = b.find_mule_chains()
    assert isinstance(chains, list)


def test_find_mule_chains_detects_chain():
    """Two mule nodes connected by a directed path should form a chain."""
    t0 = datetime(2024, 1, 1)
    df = pd.DataFrame(
        [
            {
                "transaction_id": "C1",
                "sender_account_id": "MULE1",
                "receiver_account_id": "MULE2",
                "amount_gbp": 5_000.0,
                "timestamp": t0,
                "is_mule_sender": 1,
                "is_mule_receiver": 1,
            }
        ]
    )
    b = TransactionGraphBuilder()
    b.build(df)
    chains = b.find_mule_chains(min_chain_length=1)
    assert len(chains) > 0
    all_nodes = {node for chain in chains for node in chain}
    assert "MULE1" in all_nodes or "MULE2" in all_nodes


def test_find_mule_chains_no_mule_accounts():
    t0 = datetime(2024, 1, 1)
    df = pd.DataFrame(
        [
            {
                "transaction_id": "N1",
                "sender_account_id": "P",
                "receiver_account_id": "Q",
                "amount_gbp": 500.0,
                "timestamp": t0,
                "is_mule_sender": 0,
                "is_mule_receiver": 0,
            }
        ]
    )
    b = TransactionGraphBuilder()
    b.build(df)
    chains = b.find_mule_chains()
    assert chains == []


def test_find_mule_chains_before_build_raises():
    b = TransactionGraphBuilder()
    with pytest.raises(RuntimeError):
        b.find_mule_chains()


# ---------------------------------------------------------------------------
# apply_filter() tests
# ---------------------------------------------------------------------------


def test_apply_filter_returns_digraph(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    sg = b.apply_filter(lambda attrs: attrs.get("is_mule", False))
    assert isinstance(sg, nx.DiGraph)


def test_apply_filter_mule_only(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    sg = b.apply_filter(lambda attrs: attrs.get("is_mule", False))
    # Only B is mule
    assert set(sg.nodes) == {"B"}


def test_apply_filter_inflow_threshold(simple_df):
    b = TransactionGraphBuilder()
    b.build(simple_df)
    # Only nodes with inflow >= 900
    sg = b.apply_filter(lambda attrs: attrs.get("total_inflow", 0.0) >= 900.0)
    for node in sg.nodes:
        assert b.graph.nodes[node]["total_inflow"] >= 900.0


def test_apply_filter_before_build_raises():
    b = TransactionGraphBuilder()
    with pytest.raises(RuntimeError):
        b.apply_filter(lambda _: True)


# ---------------------------------------------------------------------------
# _check_columns() tests
# ---------------------------------------------------------------------------


def test_check_columns_raises_on_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError, match="b"):
        TransactionGraphBuilder._check_columns(df, ["a", "b"])


def test_check_columns_passes_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    TransactionGraphBuilder._check_columns(df, ["a", "b"])  # no exception
