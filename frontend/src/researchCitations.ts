type Node = { type: string; value?: string; url?: string; children?: Node[] };

export function researchCitations(validIndices: number[]) {
  const valid = new Set(validIndices);
  return () => (tree: Node) => {
    function visit(parent: Node) {
      if (parent.type === "link" && parent.children?.length === 1 && parent.children[0].type === "text") {
        const match = parent.children[0].value?.match(/^\[SOURCE (\d+)\]$/);
        if (match && valid.has(Number(match[1]))) parent.url = `#atlas-evidence-${match[1]}`;
        return;
      }
      if (!parent.children || ["link", "code", "inlineCode"].includes(parent.type)) return;
      parent.children = parent.children.flatMap((child) => {
        if (child.type !== "text" || !child.value) { visit(child); return [child]; }
        const nodes: Node[] = [];
        let start = 0;
        for (const match of child.value.matchAll(/\[SOURCE (\d+)\]/g)) {
          if (!valid.has(Number(match[1]))) continue;
          if (match.index > start) nodes.push({ type: "text", value: child.value.slice(start, match.index) });
          nodes.push({ type: "link", url: `#atlas-evidence-${match[1]}`,
            children: [{ type: "text", value: match[0] }] });
          start = match.index + match[0].length;
        }
        if (!nodes.length) return [child];
        if (start < child.value.length) nodes.push({ type: "text", value: child.value.slice(start) });
        return nodes;
      });
    }
    visit(tree);
  };
}
