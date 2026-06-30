"""
Pipeline = Graph aus Node-Instanzen (ProcessingStep) + gerichteten Kanten.
Unabhängig von der GUI, damit sie auch headless (Skript, Batch) nutzbar ist.
"""
from __future__ import annotations
import json
import uuid
import numpy as np
from base_step import STEP_REGISTRY


class PipelineNode:
    def __init__(self, step, node_id: str | None = None, pos=(0, 0)):
        self.id = node_id or str(uuid.uuid4())
        self.step = step          # ProcessingStep-Instanz
        self.pos = pos            # (x, y) im Editor

    @property
    def class_name(self):
        return self.step.__class__.__name__


class Pipeline:
    def __init__(self):
        self.nodes: dict[str, PipelineNode] = {}
        # edges: list of (from_node_id, to_node_id)
        self.edges: list[tuple[str, str]] = []

    # ---------- Graph-Bearbeitung -------------------------------------
    def add_node(self, step, pos=(0, 0)) -> PipelineNode:
        node = PipelineNode(step, pos=pos)
        self.nodes[node.id] = node
        return node

    def remove_node(self, node_id: str):
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if node_id not in e]

    def add_edge(self, from_id: str, to_id: str):
        if from_id == to_id:
            return
        if (from_id, to_id) in self.edges:
            return
        # Inputs auf max. eine eingehende Kante begrenzen (lineare Kette
        # pro Eingang) - mehrere Outputs von einer Box sind erlaubt.
        self.edges = [e for e in self.edges if e[1] != to_id]
        self.edges.append((from_id, to_id))

    def remove_edge(self, from_id: str, to_id: str):
        self.edges = [e for e in self.edges if e != (from_id, to_id)]

    # ---------- Ausführung ----------------------------------------------
    def _topological_order(self) -> list[str]:
        incoming = {nid: 0 for nid in self.nodes}
        for f, t in self.edges:
            incoming[t] += 1
        queue = [nid for nid, c in incoming.items() if c == 0]
        order = []
        remaining_edges = list(self.edges)
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            outgoing = [e for e in remaining_edges if e[0] == nid]
            for f, t in outgoing:
                remaining_edges.remove((f, t))
                incoming[t] -= 1
                if incoming[t] == 0:
                    queue.append(t)
        if len(order) != len(self.nodes):
            raise ValueError("Pipeline enthält einen Zyklus - das ist nicht erlaubt.")
        return order

    def run(self, image: np.ndarray, on_step_done=None) -> dict[str, np.ndarray]:
        """
        Führt die gesamte Pipeline auf einem Bild aus.
        Gibt für jeden Node-Id das resultierende Bild zurück (so kann man
        auch Zwischenergebnisse in der UI anzeigen).
        on_step_done(node_id, image) wird optional nach jedem Schritt aufgerufen.
        """
        order = self._topological_order()
        results: dict[str, np.ndarray] = {}
        predecessors: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for f, t in self.edges:
            predecessors[t].append(f)

        for nid in order:
            node = self.nodes[nid]
            preds = predecessors[nid]
            if not preds:
                input_image = image
            else:
                # Bei mehreren Vorgängern (sollte durch add_edge-Logik nicht
                # vorkommen, aber zur Sicherheit): nimm den ersten.
                input_image = results[preds[0]]
            out = node.step.run(input_image)
            results[nid] = out
            if on_step_done:
                on_step_done(nid, out)
        return results

    def run_stack(self, images: list[np.ndarray], on_image_done=None) -> list[dict]:
        """Führt die Pipeline auf jedem Bild eines Stacks aus."""
        all_results = []
        for i, img in enumerate(images):
            res = self.run(img)
            all_results.append(res)
            if on_image_done:
                on_image_done(i, res)
        return all_results

    # ---------- Serialisierung ------------------------------------------
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
                    f"Unbekannter Step-Typ '{n['class_name']}' - ist das Modul importiert?"
                )
            step = step_cls()
            step.set_param_values(n.get("params", {}))
            node = PipelineNode(step, node_id=n["id"], pos=tuple(n.get("pos", (0, 0))))
            pipeline.nodes[node.id] = node
        pipeline.edges = [tuple(e) for e in data.get("edges", [])]
        return pipeline

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
