"""
Pipeline data model.

Edges are stored as (from_id, to_id, to_port_index) triples, which
allows fan-out (one output → many inputs) and multi-input nodes
(metric steps with ports A and B).
"""
from __future__ import annotations
import json
import uuid
import numpy as np
from base_step import STEP_REGISTRY


class PipelineNode:
    def __init__(self, step, node_id: str | None = None, pos=(0, 0)):
        self.id = node_id or str(uuid.uuid4())
        self.step = step
        self.pos = pos

    @property
    def class_name(self):
        return self.step.__class__.__name__


class Pipeline:
    def __init__(self):
        self.nodes: dict[str, PipelineNode] = {}
        # edges: list of (from_node_id, to_node_id, to_port_index)
        self.edges: list[tuple[str, str, int]] = []

    # ------------------------------------------------------------------
    # Graph editing
    # ------------------------------------------------------------------

    def add_node(self, step, pos=(0, 0)) -> PipelineNode:
        node = PipelineNode(step, pos=pos)
        self.nodes[node.id] = node
        return node

    def remove_node(self, node_id: str):
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges
                      if e[0] != node_id and e[1] != node_id]

    def add_edge(self, from_id: str, to_id: str, to_port: int = 0):
        if from_id == to_id:
            return
        # Only one source per input port
        self.edges = [e for e in self.edges
                      if not (e[1] == to_id and e[2] == to_port)]
        self.edges.append((from_id, to_id, to_port))

    def remove_edge(self, from_id: str, to_id: str, to_port: int = 0):
        self.edges = [e for e in self.edges
                      if not (e[0] == from_id and e[1] == to_id
                              and e[2] == to_port)]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _topological_order(self) -> list[str]:
        incoming = {nid: 0 for nid in self.nodes}
        for f, t, _ in self.edges:
            incoming[t] += 1
        queue = [nid for nid, c in incoming.items() if c == 0]
        order = []
        remaining = list(self.edges)
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            outgoing = [e for e in remaining if e[0] == nid]
            for edge in outgoing:
                remaining.remove(edge)
                incoming[edge[1]] -= 1
                if incoming[edge[1]] == 0:
                    queue.append(edge[1])
        if len(order) != len(self.nodes):
            raise ValueError("Pipeline contains a cycle — not allowed.")
        return order

    def run(self, image: np.ndarray,
            on_step_done=None) -> dict[str, "np.ndarray | float | str"]:
        """
        Execute the pipeline. Returns a dict mapping node_id to its output.
        For metric nodes the output is a float/str; for all others it is
        a numpy array.
        on_step_done(node_id, result) is called after each step if provided.
        """
        order = self._topological_order()

        # predecessors[to_id][port_index] = from_id
        predecessors: dict[str, dict[int, str]] = {nid: {} for nid in self.nodes}
        for f, t, p in self.edges:
            predecessors[t][p] = f

        results: dict[str, object] = {}

        for nid in order:
            node = self.nodes[nid]
            is_source = getattr(node.step, "IS_SOURCE", False)
            n_inputs = getattr(node.step, "N_INPUTS", 1)
            pred_map = predecessors[nid]   # {port_index: from_id}

            if is_source:
                inputs = [np.zeros((1, 1, 3), dtype=np.uint8)]
            else:
                inputs = []
                for i in range(n_inputs):
                    from_id = pred_map.get(i)
                    if from_id is not None and from_id in results:
                        val = results[from_id]
                        inp = val if isinstance(val, np.ndarray) else image
                    else:
                        inp = image  # fallback
                    inputs.append(inp)

            out = node.step.run(inputs)
            results[nid] = out
            if on_step_done:
                on_step_done(nid, out)

        return results

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "class_name": n.class_name,
                    "params": n.step.get_param_values(),
                    "pos": list(n.pos),
                }
                for n in self.nodes.values()
            ],
            "edges": [list(e) for e in self.edges],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline":
        pipeline = cls()
        for n in data["nodes"]:
            step_cls = STEP_REGISTRY.get(n["class_name"])
            if step_cls is None:
                raise ValueError(
                    f"Unknown step type '{n['class_name']}' — "
                    "is the module imported?"
                )
            step = step_cls()
            step.set_param_values(n.get("params", {}))
            node = PipelineNode(step, node_id=n["id"],
                                pos=tuple(n.get("pos", (0, 0))))
            pipeline.nodes[node.id] = node
        for e in data.get("edges", []):
            # support old 2-element edges from previous versions
            if len(e) == 2:
                e = [e[0], e[1], 0]
            pipeline.edges.append(tuple(e))
        return pipeline

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
