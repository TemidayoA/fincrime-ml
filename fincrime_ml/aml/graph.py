"""
aml/graph.py
=============
Transaction network graph builder for AML entity relationship mapping.

Purpose
    Money laundering is inherently a network phenomenon. Individual transaction
    screening (rule-based or ML) misses patterns that only become visible when
    the full web of entity relationships is analysed: mule chains appear as
    directed paths, structuring rings appear as star subgraphs, and integration
    flows appear as high-betweenness nodes connecting layered accounts to
    the legitimate economy.

    This module builds a directed weighted graph from a transaction DataFrame,
    annotates each node and edge with AML-relevant attributes, and exposes
    graph-analytic features (degree centrality, betweenness, PageRank, flow
    metrics) that downstream ML models consume as features.

Graph model
    - Nodes: financial entities (accounts, merchants, counterparties)
    - Edges: directed transaction flows (sender -> receiver)
    - Edge weight: total GBP transferred in the observation window
    - Edge attributes: transaction count, mean amount, time span, typology flags
    - Node attributes: in-degree, out-degree, total inflow, total outflow,
      net position, betweenness centrality, PageRank, is_mule (if labelled)

Regulatory alignment
    FATF Recommendation R.10 requires firms to understand the nature and
    purpose of business relationships. Network analysis operationalises this
    by mapping the full entity graph rather than individual transactions.
    JMLSG Part I para 5.3.17 specifically references network analysis as an
    advanced transaction monitoring technique.

    FCA Financial Crime Guide (FCG 3.2) notes that effective AML systems
    should identify complex ownership structures and fund flow patterns.

Architecture note
    Imports only from fincrime_ml.core. No imports from fincrime_ml.fraud
    permitted (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default column name constants
# ---------------------------------------------------------------------------

DEFAULT_SENDER_COL: str = "sender_account_id"
DEFAULT_RECEIVER_COL: str = "receiver_account_id"
DEFAULT_AMOUNT_COL: str = "amount_gbp"
DEFAULT_TIMESTAMP_COL: str = "timestamp"
DEFAULT_TXN_ID_COL: str = "transaction_id"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GraphStats:
    """Summary statistics for a built transaction network graph.

    Attributes:
        n_nodes: Total number of entity nodes in the graph.
        n_edges: Total number of directed edges (unique account pairs).
        total_volume_gbp: Sum of all transaction amounts in the graph.
        density: Graph density (edges / possible edges).
        n_weakly_connected_components: Number of weakly connected components.
        n_strongly_connected_components: Number of strongly connected components.
        max_in_degree: Highest in-degree (most-received-from account).
        max_out_degree: Highest out-degree (most-sent-to account).
        top_betweenness_node: Node with the highest betweenness centrality.
        top_pagerank_node: Node with the highest PageRank score.
    """

    n_nodes: int
    n_edges: int
    total_volume_gbp: float
    density: float
    n_weakly_connected_components: int
    n_strongly_connected_components: int
    max_in_degree: int
    max_out_degree: int
    top_betweenness_node: str
    top_pagerank_node: str
    metadata: dict = field(default_factory=dict)


@dataclass
class NodeFeatures:
    """Per-node AML-relevant feature vector.

    Attributes:
        node_id: Entity identifier.
        in_degree: Number of distinct senders to this node.
        out_degree: Number of distinct receivers from this node.
        total_inflow_gbp: Total GBP received.
        total_outflow_gbp: Total GBP sent.
        net_position_gbp: Inflow minus outflow.
        pass_through_ratio: Outflow / inflow (1.0 = perfect pass-through,
            characteristic of mule accounts).
        betweenness_centrality: Fraction of shortest paths passing through
            this node. High values indicate bottleneck/bridge accounts.
        pagerank: PageRank score reflecting importance in the fund flow network.
        in_txn_count: Number of inbound transactions.
        out_txn_count: Number of outbound transactions.
        is_mule: Whether this node is labelled as a mule account (if labels
            are available in the input data).
    """

    node_id: str
    in_degree: int
    out_degree: int
    total_inflow_gbp: float
    total_outflow_gbp: float
    net_position_gbp: float
    pass_through_ratio: float
    betweenness_centrality: float
    pagerank: float
    in_txn_count: int
    out_txn_count: int
    is_mule: bool = False


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class TransactionGraphBuilder:
    """Build and analyse a directed transaction network graph for AML monitoring.

    Constructs a NetworkX DiGraph from a transaction DataFrame. Each unique
    (sender, receiver) pair becomes a directed edge; multiple transactions
    between the same pair are aggregated into edge attributes. Node-level
    centrality metrics (betweenness, PageRank) are computed and cached on
    first access.

    Example::

        from fincrime_ml.aml.graph import TransactionGraphBuilder
        from fincrime_ml.core.data.synth_aml import SyntheticAMLGenerator

        gen = SyntheticAMLGenerator(seed=42)
        df = gen.generate(n_transactions=5_000, suspicious_rate=0.05)

        builder = TransactionGraphBuilder()
        builder.build(df)

        stats = builder.graph_stats()
        features = builder.node_features()
        edge_df = builder.edge_dataframe()

    Attributes:
        sender_col: Column name for the sending account.
        receiver_col: Column name for the receiving account.
        amount_col: Column name for transaction amount in GBP.
        timestamp_col: Column name for transaction timestamp.
        txn_id_col: Column name for transaction identifier.
        mule_sender_col: Optional column name for mule-sender label.
        mule_receiver_col: Optional column name for mule-receiver label.
        graph: The underlying NetworkX DiGraph (None before build()).
    """

    def __init__(
        self,
        sender_col: str = DEFAULT_SENDER_COL,
        receiver_col: str = DEFAULT_RECEIVER_COL,
        amount_col: str = DEFAULT_AMOUNT_COL,
        timestamp_col: str = DEFAULT_TIMESTAMP_COL,
        txn_id_col: str = DEFAULT_TXN_ID_COL,
        mule_sender_col: str | None = "is_mule_sender",
        mule_receiver_col: str | None = "is_mule_receiver",
    ) -> None:
        self.sender_col = sender_col
        self.receiver_col = receiver_col
        self.amount_col = amount_col
        self.timestamp_col = timestamp_col
        self.txn_id_col = txn_id_col
        self.mule_sender_col = mule_sender_col
        self.mule_receiver_col = mule_receiver_col

        self.graph: nx.DiGraph | None = None
        self._betweenness: dict[str, float] | None = None
        self._pagerank: dict[str, float] | None = None
        self._df_ref: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        df: pd.DataFrame,
        min_edge_weight: float = 0.0,
    ) -> "TransactionGraphBuilder":
        """Build the directed transaction graph from a DataFrame.

        Each unique (sender_account, receiver_account) pair becomes one
        directed edge. Parallel transactions are aggregated: the edge weight
        is the total GBP transferred, and supplementary attributes store
        the transaction count, mean amount, and time range.

        Node attributes are set from the aggregated edge data: total inflow
        (sum of incoming edge weights), total outflow (sum of outgoing edge
        weights), and mule labels if present.

        Args:
            df: Transaction DataFrame. Must contain sender_col, receiver_col,
                amount_col, timestamp_col, and txn_id_col.
            min_edge_weight: Edges with total weight below this value are
                excluded. Use to filter noise from micro-transactions.

        Returns:
            Self (for method chaining).

        Raises:
            KeyError: If required columns are absent from df.
        """
        required = [
            self.sender_col,
            self.receiver_col,
            self.amount_col,
            self.timestamp_col,
            self.txn_id_col,
        ]
        self._check_columns(df, required)

        df_work = df.copy()
        df_work[self.timestamp_col] = pd.to_datetime(df_work[self.timestamp_col])

        self._df_ref = df_work
        self._betweenness = None
        self._pagerank = None

        G = nx.DiGraph()

        # Collect mule labels per account (sender or receiver)
        mule_accounts: set[str] = set()
        if self.mule_sender_col and self.mule_sender_col in df_work.columns:
            mule_accounts.update(
                df_work.loc[df_work[self.mule_sender_col] == 1, self.sender_col].astype(str)
            )
        if self.mule_receiver_col and self.mule_receiver_col in df_work.columns:
            mule_accounts.update(
                df_work.loc[df_work[self.mule_receiver_col] == 1, self.receiver_col].astype(str)
            )

        # Aggregate edges: group by (sender, receiver)
        edge_agg = (
            df_work.groupby([self.sender_col, self.receiver_col])
            .agg(
                total_amount=(self.amount_col, "sum"),
                txn_count=(self.txn_id_col, "count"),
                mean_amount=(self.amount_col, "mean"),
                first_txn=(self.timestamp_col, "min"),
                last_txn=(self.timestamp_col, "max"),
            )
            .reset_index()
        )

        # Apply minimum edge weight filter
        if min_edge_weight > 0.0:
            edge_agg = edge_agg[edge_agg["total_amount"] >= min_edge_weight]

        # Add nodes first (to capture isolated or self-loop nodes)
        all_accounts = set(df_work[self.sender_col].astype(str).unique()) | set(
            df_work[self.receiver_col].astype(str).unique()
        )
        for account in all_accounts:
            G.add_node(str(account), is_mule=account in mule_accounts)

        # Add edges with aggregated attributes
        for _, row in edge_agg.iterrows():
            src = str(row[self.sender_col])
            dst = str(row[self.receiver_col])
            G.add_edge(
                src,
                dst,
                weight=float(row["total_amount"]),
                txn_count=int(row["txn_count"]),
                mean_amount=float(row["mean_amount"]),
                first_txn=row["first_txn"],
                last_txn=row["last_txn"],
            )

        # Compute per-node flow aggregates and attach as node attributes
        inflow = (
            df_work.groupby(self.receiver_col)[self.amount_col]
            .agg(total_inflow="sum", in_txn_count="count")
            .reset_index()
            .rename(columns={self.receiver_col: "node"})
        )
        outflow = (
            df_work.groupby(self.sender_col)[self.amount_col]
            .agg(total_outflow="sum", out_txn_count="count")
            .reset_index()
            .rename(columns={self.sender_col: "node"})
        )

        flow = inflow.merge(outflow, on="node", how="outer").fillna(0.0)
        for _, row in flow.iterrows():
            node = str(row["node"])
            if node in G:
                G.nodes[node]["total_inflow"] = float(row["total_inflow"])
                G.nodes[node]["total_outflow"] = float(row["total_outflow"])
                G.nodes[node]["in_txn_count"] = int(row["in_txn_count"])
                G.nodes[node]["out_txn_count"] = int(row["out_txn_count"])

        # Ensure nodes without flow entries have defaults
        for node in G.nodes:
            G.nodes[node].setdefault("total_inflow", 0.0)
            G.nodes[node].setdefault("total_outflow", 0.0)
            G.nodes[node].setdefault("in_txn_count", 0)
            G.nodes[node].setdefault("out_txn_count", 0)

        self.graph = G
        logger.info(
            "TransactionGraphBuilder.build: %d nodes, %d edges, £%.0f total volume",
            G.number_of_nodes(),
            G.number_of_edges(),
            sum(d["weight"] for _, _, d in G.edges(data=True)),
        )
        return self

    def graph_stats(self) -> GraphStats:
        """Return summary statistics for the built graph.

        Returns:
            GraphStats dataclass with node/edge counts, density, connectivity,
            and top-centrality nodes.

        Raises:
            RuntimeError: If build() has not been called.
        """
        self._check_built()
        G = self.graph

        betweenness = self._get_betweenness()
        pagerank = self._get_pagerank()

        total_volume = sum(d["weight"] for _, _, d in G.edges(data=True))
        top_between = max(betweenness, key=lambda n: betweenness[n]) if betweenness else ""
        top_pr = max(pagerank, key=lambda n: pagerank[n]) if pagerank else ""

        return GraphStats(
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            total_volume_gbp=round(total_volume, 2),
            density=round(nx.density(G), 6),
            n_weakly_connected_components=nx.number_weakly_connected_components(G),
            n_strongly_connected_components=nx.number_strongly_connected_components(G),
            max_in_degree=max((d for _, d in G.in_degree()), default=0),
            max_out_degree=max((d for _, d in G.out_degree()), default=0),
            top_betweenness_node=str(top_between),
            top_pagerank_node=str(top_pr),
        )

    def node_features(self) -> pd.DataFrame:
        """Return a DataFrame of per-node AML features.

        Includes flow metrics (inflow, outflow, pass-through ratio),
        topological metrics (degree, betweenness, PageRank), and mule labels.

        Returns:
            DataFrame with one row per node. Columns: node_id, in_degree,
            out_degree, total_inflow_gbp, total_outflow_gbp, net_position_gbp,
            pass_through_ratio, betweenness_centrality, pagerank, in_txn_count,
            out_txn_count, is_mule.

        Raises:
            RuntimeError: If build() has not been called.
        """
        self._check_built()
        G = self.graph
        betweenness = self._get_betweenness()
        pagerank = self._get_pagerank()

        rows = []
        for node in G.nodes:
            nd = G.nodes[node]
            inflow = nd.get("total_inflow", 0.0)
            outflow = nd.get("total_outflow", 0.0)
            pass_through = outflow / inflow if inflow > 0 else 0.0

            rows.append(
                {
                    "node_id": node,
                    "in_degree": G.in_degree(node),
                    "out_degree": G.out_degree(node),
                    "total_inflow_gbp": round(inflow, 2),
                    "total_outflow_gbp": round(outflow, 2),
                    "net_position_gbp": round(inflow - outflow, 2),
                    "pass_through_ratio": round(pass_through, 4),
                    "betweenness_centrality": round(betweenness.get(node, 0.0), 6),
                    "pagerank": round(pagerank.get(node, 0.0), 6),
                    "in_txn_count": nd.get("in_txn_count", 0),
                    "out_txn_count": nd.get("out_txn_count", 0),
                    "is_mule": bool(nd.get("is_mule", False)),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("betweenness_centrality", ascending=False)
            .reset_index(drop=True)
        )

    def edge_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of all edges with aggregated attributes.

        Returns:
            DataFrame with columns: source, target, weight (total GBP),
            txn_count, mean_amount, first_txn, last_txn.

        Raises:
            RuntimeError: If build() has not been called.
        """
        self._check_built()
        rows = []
        for src, dst, data in self.graph.edges(data=True):
            rows.append(
                {
                    "source": src,
                    "target": dst,
                    "weight": round(data.get("weight", 0.0), 2),
                    "txn_count": data.get("txn_count", 0),
                    "mean_amount": round(data.get("mean_amount", 0.0), 2),
                    "first_txn": data.get("first_txn"),
                    "last_txn": data.get("last_txn"),
                }
            )
        return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)

    def subgraph(self, node_id: str, radius: int = 1) -> nx.DiGraph:
        """Extract the ego-network subgraph around a given node.

        Returns the induced subgraph consisting of all nodes within
        ``radius`` hops of ``node_id`` in the underlying undirected graph,
        preserving directionality of edges.

        Args:
            node_id: The focal node identifier.
            radius: Number of hops to include (default 1 = immediate neighbours).

        Returns:
            A NetworkX DiGraph subgraph view.

        Raises:
            RuntimeError: If build() has not been called.
            KeyError: If node_id is not in the graph.
        """
        self._check_built()
        if node_id not in self.graph:
            raise KeyError(f"subgraph: node '{node_id}' not found in graph.")
        undirected = self.graph.to_undirected()
        ego = nx.ego_graph(undirected, node_id, radius=radius)
        return self.graph.subgraph(ego.nodes).copy()

    def flag_high_risk_nodes(
        self,
        betweenness_pct: float = 90.0,
        pagerank_pct: float = 90.0,
        pass_through_threshold: float = 0.80,
    ) -> pd.DataFrame:
        """Flag nodes that exceed risk thresholds on centrality or flow metrics.

        Combines three independent risk signals into a composite flag:

        1. **High betweenness**: the node is a structural bridge in the
           network, consistent with a layering intermediary.
        2. **High PageRank**: the node receives disproportionate fund flow
           relative to the broader network.
        3. **High pass-through ratio**: the node forwards most of what it
           receives, consistent with a mule account.

        Args:
            betweenness_pct: Percentile above which betweenness centrality
                is flagged (default: 90th percentile).
            pagerank_pct: Percentile above which PageRank is flagged.
            pass_through_threshold: Pass-through ratio above which the node
                is flagged (default 0.80 = forwards at least 80% of inflow).

        Returns:
            DataFrame of flagged nodes with columns: node_id, flag_reason,
            betweenness_centrality, pagerank, pass_through_ratio, is_mule.

        Raises:
            RuntimeError: If build() has not been called.
        """
        self._check_built()
        node_df = self.node_features()
        if len(node_df) == 0:
            return pd.DataFrame(
                columns=[
                    "node_id",
                    "flag_reason",
                    "betweenness_centrality",
                    "pagerank",
                    "pass_through_ratio",
                    "is_mule",
                ]
            )

        bt_thresh = float(np.percentile(node_df["betweenness_centrality"].values, betweenness_pct))
        pr_thresh = float(np.percentile(node_df["pagerank"].values, pagerank_pct))

        flagged = []
        for _, row in node_df.iterrows():
            reasons = []
            if row["betweenness_centrality"] >= bt_thresh and bt_thresh > 0:
                reasons.append("high_betweenness")
            if row["pagerank"] >= pr_thresh and pr_thresh > 0:
                reasons.append("high_pagerank")
            if row["pass_through_ratio"] >= pass_through_threshold and row["total_inflow_gbp"] > 0:
                reasons.append("high_pass_through")

            if reasons:
                flagged.append(
                    {
                        "node_id": row["node_id"],
                        "flag_reason": "|".join(reasons),
                        "betweenness_centrality": row["betweenness_centrality"],
                        "pagerank": row["pagerank"],
                        "pass_through_ratio": row["pass_through_ratio"],
                        "is_mule": row["is_mule"],
                    }
                )

        logger.info(
            "flag_high_risk_nodes: %d nodes flagged from %d total",
            len(flagged),
            len(node_df),
        )
        return pd.DataFrame(flagged).reset_index(drop=True)

    def find_mule_chains(self, min_chain_length: int = 2) -> list[list[str]]:
        """Find directed paths in the graph that constitute mule account chains.

        A mule chain is a directed path where at least two consecutive nodes
        are mule accounts. This corresponds to the layering stage of the FATF
        money laundering lifecycle, where funds move through multiple
        controlled accounts to obscure their origin.

        This method uses simple path enumeration on the graph. For large
        graphs with many mule accounts, this can be computationally intensive;
        the method is designed for subgraph analysis rather than full-graph
        traversal.

        Args:
            min_chain_length: Minimum number of hops (edges) in a chain.
                A chain of 2 means at least 3 nodes.

        Returns:
            List of node-id lists, each representing one mule chain path.

        Raises:
            RuntimeError: If build() has not been called.
        """
        self._check_built()
        G = self.graph
        mule_nodes = {n for n, d in G.nodes(data=True) if d.get("is_mule", False)}

        if len(mule_nodes) < 2:
            return []

        chains: list[list[str]] = []

        for source in mule_nodes:
            for target in mule_nodes:
                if source == target:
                    continue
                try:
                    path = nx.shortest_path(G, source, target)
                except nx.NetworkXNoPath:
                    continue
                if len(path) >= min_chain_length + 1:
                    chains.append(path)

        # Deduplicate (same path in different orderings)
        seen: set[tuple[str, ...]] = set()
        unique_chains = []
        for chain in chains:
            key = tuple(chain)
            if key not in seen:
                seen.add(key)
                unique_chains.append(chain)

        return unique_chains

    def apply_filter(
        self,
        predicate: Callable[[dict], bool],
    ) -> nx.DiGraph:
        """Return a subgraph containing only nodes satisfying a predicate.

        Args:
            predicate: A function that takes a node attribute dict and returns
                True to include the node, False to exclude it.

        Returns:
            A new DiGraph with only the filtered nodes and their induced edges.

        Raises:
            RuntimeError: If build() has not been called.
        """
        self._check_built()
        keep = [n for n, d in self.graph.nodes(data=True) if predicate(d)]
        return self.graph.subgraph(keep).copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_betweenness(self) -> dict[str, float]:
        if self._betweenness is None:
            self._betweenness = nx.betweenness_centrality(
                self.graph, normalized=True, weight="weight"
            )
        return self._betweenness

    def _get_pagerank(self) -> dict[str, float]:
        if self._pagerank is None:
            try:
                self._pagerank = nx.pagerank(self.graph, weight="weight")
            except nx.PowerIterationFailedConvergence:
                logger.warning("PageRank failed to converge; returning uniform scores.")
                n = self.graph.number_of_nodes()
                self._pagerank = {node: 1.0 / n for node in self.graph.nodes}
        return self._pagerank

    def _check_built(self) -> None:
        if self.graph is None:
            raise RuntimeError(
                "TransactionGraphBuilder: call build() before accessing graph properties."
            )

    @staticmethod
    def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"TransactionGraphBuilder: required columns missing: {missing}")
