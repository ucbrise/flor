from ground.client import GroundClient

g = GroundClient()
id = g.create_node("action", "action")
x = g.create_node("x", "x")
y = g.create_node("y", "y")
n = g.create_node("n", "n")
acc = g.create_node("acc", "acc")

#create first subgraph
id0 = g.create_node_version(node_id = id.get_id(), tags={"value": {"action": "action"}})
x0 = g.create_node_version(node_id=x.get_id(), tags={"value": {"x": 0}})
y0 = g.create_node_version(node_id=y.get_id(), tags={"value": {"y": 1}})
n0 = g.create_node_version(node_id=n.get_id(), tags={"value": {"num_est": 2}})
acc0 = g.create_node_version(node_id=acc.get_id(), tags={"value": {"accuracy": 0.3}})

le1 = g.create_lineage_edge("consumes_x0", "consumes_x0")
le2 = g.create_lineage_edge("consumes_y0", "consumes_y0")
le3 = g.create_lineage_edge("consumes_n0", "consumes_n0")
le4 = g.create_lineage_edge("produces_acc0", "produces_acc0")
g.create_lineage_edge_version(le1.get_id(), y0.get_id(), id0.get_id())
g.create_lineage_edge_version(le2.get_id(), x0.get_id(), id0.get_id())
g.create_lineage_edge_version(le3.get_id(), n0.get_id(), id0.get_id())
g.create_lineage_edge_version(le4.get_id(), acc0.get_id(), id0.get_id())

#create second subgraph
id1 = g.create_node_version(node_id = id.get_id(), tags={"value": {"action": "action"}})
y1 = g.create_node_version(node_id=y.get_id(), tags={"value": {"y": 2}})
acc1 = g.create_node_version(node_id=acc.get_id(), tags={"value": {"accuracy": 0.4}})

le1 = g.create_lineage_edge("consumes_x1", "consumes_x1")
le2 = g.create_lineage_edge("consumes_y1", "consumes_y1")
le3 = g.create_lineage_edge("consumes_n1", "consumes_n1")
le4 = g.create_lineage_edge("produces_acc1", "produces_acc1")
g.create_lineage_edge_version(le1.get_id(), y1.get_id(), id1.get_id())
g.create_lineage_edge_version(le2.get_id(), x0.get_id(), id1.get_id())
g.create_lineage_edge_version(le3.get_id(), n0.get_id(), id1.get_id())
g.create_lineage_edge_version(le4.get_id(), acc1.get_id(), id1.get_id())

#link logged/linked edges
le1 = g.create_lineage_edge("xy0", "xy0")
le2 = g.create_lineage_edge("xy1", "xy1")
le3 = g.create_lineage_edge("nacc0", "nacc0")
le4 = g.create_lineage_edge("nacc1", "nacc1")
g.create_lineage_edge_version(le1.get_id(), y0.get_id(), x0.get_id())
g.create_lineage_edge_version(le2.get_id(), y1.get_id(), x0.get_id())
g.create_lineage_edge_version(le3.get_id(), acc0.get_id(), n0.get_id())
g.create_lineage_edge_version(le4.get_id(), acc1.get_id(), n0.get_id())

#link first subgraph to second
le1 = g.create_lineage_edge("id0_1", "id0_1")
le2 = g.create_lineage_edge("y0_1", "y0_1")
le3 = g.create_lineage_edge("acc0_1", "acc0_1")
g.create_lineage_edge_version(le1.get_id(), id1.get_id(), id0.get_id())
g.create_lineage_edge_version(le2.get_id(), y1.get_id(), y0.get_id())
g.create_lineage_edge_version(le3.get_id(), acc1.get_id(), acc0.get_id())


