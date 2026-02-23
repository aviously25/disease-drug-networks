"""
Interactive Knowledge Graph Visualization with Streamlit + Pyvis

Run with: streamlit run visualize_kg.py

Features:
- Search nodes by name
- Filter by node type and relation type
- Explore N-hop neighborhoods
- Interactive graph visualization
- Node details panel
- Export subgraphs
"""

import streamlit as st
import pandas as pd
import networkx as nx
import pickle
from pyvis.network import Network
import tempfile
import os
from collections import Counter

# Page config
st.set_page_config(
    page_title="Knowledge Graph Explorer",
    page_icon="🔬",
    layout="wide"
)

# Color mapping for node types
NODE_COLORS = {
    'gene/protein': '#4CAF50',      # Green
    'drug': '#2196F3',              # Blue
    'disease': '#F44336',           # Red
    'variant': '#9C27B0',           # Purple
    'haplotype': '#E91E63',         # Pink
    'pathway': '#FF9800',           # Orange
    'biological_process': '#00BCD4', # Cyan
    'molecular_function': '#CDDC39', # Lime
    'cellular_component': '#795548', # Brown
    'anatomy': '#607D8B',           # Blue Grey
    'effect/phenotype': '#FF5722',  # Deep Orange
    'exposure': '#9E9E9E',          # Grey
}

# Default color for unknown types
DEFAULT_COLOR = '#757575'


@st.cache_resource
def load_graph():
    """Load the combined knowledge graph."""
    pkl_path = 'data/combined/combined_graph.pkl'

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            G = pickle.load(f)
        return G
    else:
        st.error(f"Graph file not found: {pkl_path}")
        st.info("Please run `python combine_knowledge_graphs.py` first to generate the combined graph.")
        return None


@st.cache_data
def get_node_index(_G):
    """Build a searchable index of node names."""
    index = {}
    for node in _G.nodes():
        attrs = _G.nodes[node]
        name = attrs.get('name', str(node))
        # Index by lowercase name for case-insensitive search
        if isinstance(name, str):
            index[name.lower()] = node
        else:
            index[str(name).lower()] = node
        # Also index by unified_id (convert to string if needed)
        index[str(node).lower()] = node
    return index


@st.cache_data
def get_graph_stats(_G):
    """Compute basic graph statistics."""
    node_types = Counter()
    for node in _G.nodes():
        node_type = _G.nodes[node].get('node_type', 'unknown')
        node_types[node_type] += 1

    relation_types = Counter()
    for u, v, data in _G.edges(data=True):
        relation = data.get('relation', 'unknown')
        relation_types[relation] += 1

    return {
        'num_nodes': _G.number_of_nodes(),
        'num_edges': _G.number_of_edges(),
        'node_types': dict(node_types),
        'relation_types': dict(relation_types)
    }


def extract_subgraph(G, center_node, hops=2, max_nodes=500, node_types=None, relation_types=None):
    """Extract a subgraph around a center node with filtering."""
    if center_node not in G:
        return None

    # BFS to find nodes within N hops
    visited = {center_node}
    current_level = {center_node}

    for _ in range(hops):
        next_level = set()
        for node in current_level:
            # Get neighbors (works for both directed and undirected graphs)
            if G.is_directed():
                neighbors = set(G.successors(node)) | set(G.predecessors(node))
            else:
                neighbors = set(G.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    # Check node type filter
                    if node_types:
                        neighbor_type = G.nodes[neighbor].get('node_type', 'unknown')
                        if neighbor_type not in node_types:
                            continue
                    next_level.add(neighbor)
                    visited.add(neighbor)

                    # Stop if we've reached max nodes
                    if len(visited) >= max_nodes:
                        break
            if len(visited) >= max_nodes:
                break
        current_level = next_level
        if len(visited) >= max_nodes:
            break

    # Create subgraph
    subgraph = G.subgraph(visited).copy()

    # Filter edges by relation type if specified
    if relation_types:
        edges_to_remove = []
        for u, v, key, data in subgraph.edges(keys=True, data=True):
            relation = data.get('relation', 'unknown')
            if relation not in relation_types:
                edges_to_remove.append((u, v, key))
        for u, v, key in edges_to_remove:
            subgraph.remove_edge(u, v, key)

    return subgraph


def build_pyvis_network(subgraph, center_node=None, height="600px"):
    """Convert NetworkX subgraph to Pyvis network."""
    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=subgraph.is_directed(),
        select_menu=True,
        filter_menu=True
    )

    # Physics settings for better layout
    net.barnes_hut(
        gravity=-5000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.9
    )

    # Add nodes (convert IDs to strings for Pyvis compatibility)
    node_id_map = {}  # Map original IDs to string IDs
    for node in subgraph.nodes():
        node_str = str(node)
        node_id_map[node] = node_str

        attrs = subgraph.nodes[node]
        node_type = attrs.get('node_type', 'unknown')
        name = str(attrs.get('name', node))

        # Color based on node type
        color = NODE_COLORS.get(node_type, DEFAULT_COLOR)

        # Highlight center node
        if node == center_node:
            color = '#FFD700'  # Gold
            border_width = 3
        else:
            border_width = 1

        # Size based on degree
        degree = subgraph.degree(node)
        size = min(10 + degree * 2, 50)

        # Build title (hover text)
        title_parts = [
            f"<b>{name}</b>",
            f"Type: {node_type}",
            f"Connections: {degree}"
        ]
        if attrs.get('source'):
            title_parts.append(f"Source: {attrs.get('source')}")
        title = "<br>".join(title_parts)

        net.add_node(
            node_str,
            label=name[:20] + "..." if len(name) > 20 else name,
            title=title,
            color=color,
            size=size,
            borderWidth=border_width,
            group=node_type
        )

    # Add edges (use string IDs)
    is_directed = subgraph.is_directed()
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', 'unknown')
        source_kg = data.get('source_kg', 'unknown')

        # Edge color based on source
        if source_kg == 'PharmGKB':
            edge_color = '#9C27B0'  # Purple for PharmGKB
        else:
            edge_color = '#888888'  # Grey for PrimeKG

        title = f"{relation}<br>Source: {source_kg}"

        edge_args = {
            'title': title,
            'color': edge_color,
        }
        if is_directed:
            edge_args['arrows'] = 'to'

        net.add_edge(node_id_map[u], node_id_map[v], **edge_args)

    return net


def render_graph(net):
    """Render Pyvis network in Streamlit."""
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        net.save_graph(f.name)
        temp_path = f.name

    # Read and display
    with open(temp_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Clean up
    os.unlink(temp_path)

    # Display in Streamlit
    st.components.v1.html(html, height=650, scrolling=True)


def main():
    st.title("Knowledge Graph Explorer")
    st.markdown("*Interactive visualization of the combined PrimeKG + PharmGKB knowledge graph*")

    # Load graph
    G = load_graph()
    if G is None:
        return

    # Get stats and index
    stats = get_graph_stats(G)
    node_index = get_node_index(G)

    # Sidebar
    with st.sidebar:
        st.header("Graph Statistics")
        st.metric("Total Nodes", f"{stats['num_nodes']:,}")
        st.metric("Total Edges", f"{stats['num_edges']:,}")

        st.divider()

        st.header("Search & Filter")

        # Search box
        search_query = st.text_input(
            "Search node by name",
            placeholder="e.g., BRCA1, Aspirin, Diabetes"
        )

        # Find matching nodes
        matching_nodes = []
        if search_query:
            query_lower = search_query.lower()
            for name, node_id in node_index.items():
                if query_lower in name:
                    matching_nodes.append((name, node_id))
            matching_nodes = matching_nodes[:50]  # Limit results

        # Node selection
        selected_node = None
        if matching_nodes:
            options = [f"{name} ({G.nodes[nid].get('node_type', 'unknown')})"
                      for name, nid in matching_nodes]
            selection = st.selectbox(
                f"Found {len(matching_nodes)} matches",
                options=options,
                index=0
            )
            if selection:
                idx = options.index(selection)
                selected_node = matching_nodes[idx][1]

        st.divider()

        # Node type filter
        st.subheader("Filter by Node Type")
        available_types = sorted(stats['node_types'].keys())
        selected_types = st.multiselect(
            "Include node types",
            options=available_types,
            default=available_types[:5] if len(available_types) > 5 else available_types,
            help="Select which node types to include in the visualization"
        )

        st.divider()

        # Relation type filter
        st.subheader("Filter by Relation")
        available_relations = sorted(stats['relation_types'].keys())
        selected_relations = st.multiselect(
            "Include relations",
            options=available_relations,
            default=None,
            help="Leave empty to include all relations"
        )

        st.divider()

        # Hop distance
        hops = st.slider(
            "Neighborhood hops",
            min_value=1,
            max_value=3,
            value=2,
            help="Number of hops from the center node to explore"
        )

        # Max nodes
        max_nodes = st.slider(
            "Maximum nodes",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Limit the number of nodes to display for performance"
        )

    # Main content
    col1, col2 = st.columns([3, 1])

    with col1:
        if selected_node:
            # Extract subgraph
            node_type_filter = set(selected_types) if selected_types else None
            relation_filter = set(selected_relations) if selected_relations else None

            subgraph = extract_subgraph(
                G,
                selected_node,
                hops=hops,
                max_nodes=max_nodes,
                node_types=node_type_filter,
                relation_types=relation_filter
            )

            if subgraph and subgraph.number_of_nodes() > 0:
                node_name = G.nodes[selected_node].get('name', selected_node)
                st.subheader(f"Neighborhood of: {node_name}")
                st.caption(f"Showing {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges ({hops}-hop neighborhood)")

                # Build and render visualization
                net = build_pyvis_network(subgraph, center_node=selected_node)
                render_graph(net)
            else:
                st.warning("No nodes found with current filters. Try adjusting the filters.")
        else:
            st.info("Enter a search term in the sidebar to explore the knowledge graph.")

            # Show node type distribution
            st.subheader("Node Type Distribution")
            type_df = pd.DataFrame([
                {"Type": k, "Count": v, "Color": NODE_COLORS.get(k, DEFAULT_COLOR)}
                for k, v in sorted(stats['node_types'].items(), key=lambda x: -x[1])
            ])
            st.dataframe(type_df[["Type", "Count"]], use_container_width=True)

    with col2:
        st.subheader("Node Details")

        if selected_node:
            attrs = G.nodes[selected_node]

            st.markdown(f"**Name:** {attrs.get('name', selected_node)}")
            st.markdown(f"**Type:** {attrs.get('node_type', 'unknown')}")
            st.markdown(f"**Source:** {attrs.get('source', 'unknown')}")

            # Degree information
            if G.is_directed():
                in_degree = G.in_degree(selected_node)
                out_degree = G.out_degree(selected_node)
                st.markdown(f"**In-degree:** {in_degree}")
                st.markdown(f"**Out-degree:** {out_degree}")
            else:
                degree = G.degree(selected_node)
                st.markdown(f"**Degree:** {degree}")

            # Original IDs
            if attrs.get('primekg_id'):
                st.markdown(f"**PrimeKG ID:** {attrs.get('primekg_id')}")
            if attrs.get('pharmgkb_id'):
                st.markdown(f"**PharmGKB ID:** {attrs.get('pharmgkb_id')}")

            st.divider()

            # Export options
            st.subheader("Export Subgraph")
            if st.button("Download as CSV"):
                if subgraph:
                    # Create edge list CSV
                    edges = []
                    for u, v, data in subgraph.edges(data=True):
                        edges.append({
                            'source': u,
                            'target': v,
                            'relation': data.get('relation', ''),
                            'source_kg': data.get('source_kg', '')
                        })
                    edge_df = pd.DataFrame(edges)
                    csv = edge_df.to_csv(index=False)
                    st.download_button(
                        "Download Edge List",
                        csv,
                        file_name=f"subgraph_{attrs.get('name', selected_node)}.csv",
                        mime="text/csv"
                    )
        else:
            st.caption("Select a node to see details")

        # Legend
        st.divider()
        st.subheader("Legend")
        for node_type, color in sorted(NODE_COLORS.items()):
            st.markdown(
                f'<span style="color:{color};">&#9679;</span> {node_type}',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
