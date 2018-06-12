class VersionHistoryDag:

    def __init__(self, item_id, edges):
        self._item_id = item_id
        self._edge_ids = [edge.get_id() for edge in edges]
        self._parent_child_map = {}
        for edge in edges:
            self.add_to_parent_child_map(edge.get_from_id(), edge.get_to_id())

    def get_item_id(self):
        return self._item_id

    def get_edge_ids(self):
        return self._edge_ids

    def check_item_in_dag(self, id):
        return id in self._parent_child_map or id in self.get_leaves()

    def add_edge(self, parent_id, child_id, successor_id):
        self._edge_ids.add(successor_id)
        self.add_to_parent_child_map(parent_id, child_id)

    def get_parent(self, child_id):
        return [key for key in self._parent_child_map if child_id in self._parent_child_map[key]]

    def get_parent_child_pairs(self):
        result = {}
        for parent in self._parent_child_map.keys():
            children = self._parent_child_map[parent]
            for child in children:
                result[parent] = child
        return result

    def get_leaves(self):
        return set(self._parent_child_map.values()) - set(self._parent_child_map.keys())

    def add_to_parent_child_map(self, parent, child):
        if parent not in self._parent_child_map:
            self._parent_child_map[parent] = []
        self._parent_child_map[parent].append(child)
