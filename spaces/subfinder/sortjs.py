"""Client-side sorting for the results table.

Injected into <head> rather than inlined in the HTML payload, because Gradio
re-renders the HTML component on every prediction and an inline <script> would
either be stripped or re-executed on each render.
"""

SORT_JS = """
<script>
window.sfTail = function (btn) {
  const card = btn.closest('.sf-card');
  const tail = card.querySelectorAll('tr.sf-tail');
  const open = btn.dataset.open === '1';
  tail.forEach(tr => tr.hidden = open);
  btn.dataset.open = open ? '0' : '1';
  btn.textContent = open ? btn.dataset.more : btn.dataset.less;
};

window.sfSort = function (th) {
  const table = th.closest('table');
  const tbody = table.tBodies[0];
  const idx   = Array.from(th.parentNode.children).indexOf(th);
  const type  = th.dataset.type || 'text';
  const asc   = th.dataset.dir !== 'asc';

  Array.from(th.parentNode.children).forEach(h => {
    if (h !== th) { delete h.dataset.dir; h.classList.remove('asc','desc'); }
  });
  th.dataset.dir = asc ? 'asc' : 'desc';
  th.classList.toggle('asc',  asc);
  th.classList.toggle('desc', !asc);

  const key = (tr) => {
    const cell = tr.children[idx];
    const raw  = cell.dataset.v !== undefined ? cell.dataset.v : cell.textContent.trim();
    return type === 'num' ? parseFloat(raw) : raw.toLowerCase();
  };
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {
    const x = key(a), y = key(b);
    if (type === 'num') {
      const xa = isNaN(x) ? -Infinity : x, ya = isNaN(y) ? -Infinity : y;
      return asc ? xa - ya : ya - xa;
    }
    return asc ? x.localeCompare(y) : y.localeCompare(x);
  });
  rows.forEach(r => tbody.appendChild(r));
};

window.sfFilter = function (inp) {
  const table = document.getElementById(inp.dataset.target);
  if (!table) return;
  const q = inp.value.trim().toLowerCase();
  let shown = 0;
  Array.from(table.tBodies[0].rows).forEach(tr => {
    const hit = !q || tr.textContent.toLowerCase().includes(q);
    tr.style.display = hit ? '' : 'none';
    if (hit) shown++;
  });
  const c = document.getElementById(inp.dataset.count);
  if (c) c.textContent = shown + ' of ' + table.tBodies[0].rows.length + ' rows';
};

window.sfOnlySig = function (cb) {
  const table = document.getElementById(cb.dataset.target);
  if (!table) return;
  let shown = 0;
  Array.from(table.tBodies[0].rows).forEach(tr => {
    const hit = !cb.checked || tr.dataset.sig === '1';
    tr.style.display = hit ? '' : 'none';
    if (hit) shown++;
  });
  const c = document.getElementById(cb.dataset.count);
  if (c) c.textContent = shown + ' of ' + table.tBodies[0].rows.length + ' rows';
};
</script>
"""
