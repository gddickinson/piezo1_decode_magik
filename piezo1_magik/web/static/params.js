// Parameters: configs/*.yaml and every script's command-line options, read from the repo, with units.
import { api, app, esc, list, route } from './core.js';

const val = v => (v === null || v === undefined ? '<span class="mute">none</span>' : esc(Array.isArray(v) ? v.join(' × ') : String(v)));
const umCell = u => (u === null || u === undefined ? '' : `${esc(Array.isArray(u) ? u.join(' × ') : u)} µm`);

route('params', async (args, q) => {
  const p = await api('/params');
  app().innerHTML = `
  <h1>Parameters</h1>
  <p class="lead">Read from the repository when this page was built: ${p.n_config_values} values in <code>configs/*.yaml</code> and
    ${p.n_options} options of ${p.scripts.length} scripts. Units come from the code's comments and help text;
    pixel values are also given in µm at ${p.pixel_size_nm} nm/px (the PSF-model default). Times are in frames: no frame interval is set anywhere in the code.</p>
  <label for="pf">Filter by name or text</label><input id="pf" value="${esc(q.q || '')}" placeholder="e.g. threshold, distance, roi">
  <div id="ptabs"></div>`;
  const render = () => {
    const f = document.getElementById('pf').value.toLowerCase();
    const has = s => !f || s.toLowerCase().includes(f);
    document.getElementById('ptabs').innerHTML = `
    <h2>Config files</h2>${list(p.configs, c => {
      const rows = c.rows.filter(r => has(`${r.key} ${r.comment} ${r.unit}`));
      return rows.length ? `<h3>${esc(c.file)}</h3><p class="small mute">${esc(c.read_by)}</p>
      <div class="scroll"><table class="tbl"><thead><tr><th>key</th><th>value</th><th>unit</th><th>in µm</th><th>comment in the file</th></tr></thead><tbody>
      ${list(rows, r => `<tr><td>${esc(r.key)}</td><td>${val(r.value)}</td><td class="unit">${esc(r.unit)}</td><td>${umCell(r.um)}</td><td class="wrap">${esc(r.comment)}</td></tr>`)}
      </tbody></table></div>` : '';
    })}
    <h2>Script options</h2>${list(p.scripts, s => {
      const rows = s.options.filter(o => has(`${o.name} ${o.help} ${o.unit} ${s.script}`));
      return rows.length ? `<h3>${esc(s.script)}</h3><p class="small mute">${esc(s.summary)}</p>
      <div class="scroll"><table class="tbl"><thead><tr><th>option</th><th>type</th><th>default</th><th>unit</th><th>in µm</th><th>help</th></tr></thead><tbody>
      ${list(rows, o => `<tr><td>${esc(o.name)}${o.required ? ' <span class="badge local">required</span>' : ''}</td><td>${esc(o.type)}</td>
        <td>${o.required ? '' : val(o.default)}</td><td class="unit">${esc(o.unit)}</td><td>${umCell(o.um)}</td><td class="wrap">${esc(o.help)}</td></tr>`)}
      </tbody></table></div>` : '';
    })}`;
  };
  document.getElementById('pf').addEventListener('input', render);
  render();
});
