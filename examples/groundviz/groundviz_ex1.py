from ground.client import GroundClient

g = GroundClient()
id = g.create_node("action", "action")
x = g.create_node("x", "x")
y = g.create_node("y", "y")

#create first subgraph
id0 = g.create_node_version(node_id = id.get_id(), tags={"value": {"action": "action"}})
x0 = g.create_node_version(node_id=x.get_id(), tags={"value": {"input": 0}})
y0 = g.create_node_version(node_id=y.get_id(), tags={"result": {"result": 1}})

le1 = g.create_lineage_edge("produces0", "produces0")
le2 = g.create_lineage_edge("consumes0", "consumes0")
g.create_lineage_edge_version(le1.get_id(), y0.get_id(), id0.get_id())
g.create_lineage_edge_version(le2.get_id(), x0.get_id(), id0.get_id())

#create second subgraph
id1 = g.create_node_version(node_id = id.get_id(), tags={"value": {"action": "action"}})
x1 = g.create_node_version(node_id=x.get_id(), tags={"value": {"input": 1}})
y1 = g.create_node_version(node_id=y.get_id(), tags={"result": {"result": 2}})

le1 = g.create_lineage_edge("produces1", "produces1")
le2 = g.create_lineage_edge("consumes1", "consumes1")
g.create_lineage_edge_version(le1.get_id(), y1.get_id(), id1.get_id())
g.create_lineage_edge_version(le2.get_id(), x1.get_id(), id1.get_id())

#link first subgraph to second
le1 = g.create_lineage_edge("id0_1", "id0_1")
le2 = g.create_lineage_edge("x0_1", "x0_1")
le3 = g.create_lineage_edge("y0_1", "y0_1")
g.create_lineage_edge_version(le1.get_id(), id1.get_id(), id0.get_id())
g.create_lineage_edge_version(le2.get_id(), x1.get_id(), x0.get_id())
g.create_lineage_edge_version(le3.get_id(), y1.get_id(), y0.get_id())

#create third subgraph
id2 = g.create_node_version(node_id = id.get_id(), tags={"value": {"action": "action"}})
x2 = g.create_node_version(node_id=x1.get_node_id(), tags={"value": {"input": 2}})
y2 = g.create_node_version(node_id=y1.get_node_id(), tags ={"result": {"result": 3}})

le1 = g.create_lineage_edge("produces2", "produces2")
le2 = g.create_lineage_edge("consumes2", "consumes2")
g.create_lineage_edge_version(le1.get_id(), y2.get_id(), id2.get_id())
g.create_lineage_edge_version(le2.get_id(), x2.get_id(), id2.get_id())

#link second subgraph to third
le1 = g.create_lineage_edge("id1_2", "id1_2")
le2 = g.create_lineage_edge("x1_2", "x1_2")
le3 = g.create_lineage_edge("y1_2", "y1_2")
g.create_lineage_edge_version(le1.get_id(), id2.get_id(), id1.get_id())
g.create_lineage_edge_version(le2.get_id(), x2.get_id(), x1.get_id())
g.create_lineage_edge_version(le3.get_id(), y2.get_id(), y1.get_id())

