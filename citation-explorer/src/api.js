// src/api.js
export async function fetchGraphForDOI(doi) {
    const res = await fetch('/api/graph', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doi }),
    });
    if (!res.ok) throw new Error(`API error ${res.status}`);
    return res.json(); // { nodes, links }
  }
  