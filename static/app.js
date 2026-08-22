/* SLM Foundry — single-page console.
 *
 * No framework, no build step: one script, hand-rolled DOM helpers, and a state
 * object. The whole surface is about a dozen views over a REST API, and a
 * toolchain would cost more to maintain than it saves.
 *
 * Two conventions worth knowing before editing:
 *   - `h()` builds elements; anything user-supplied goes in as a text node, never
 *     as innerHTML. Model outputs are shown verbatim all over this UI, and they
 *     are the least trustworthy strings in the product.
 *   - Views are pure re-renders from `state`. Nothing patches the DOM in place,
 *     so a stale fragment cannot survive a refresh.
 */
'use strict';

// ------------------------------------------------------------------ utilities

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'html') el.innerHTML = v;             // only for our own markup
    else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
const set = (el, ...kids) => { clear(el); kids.flat().forEach(k => k && el.append(k)); return el; };

const fmtNum = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v)) ? '—'
  : (typeof v === 'number' ? (Math.abs(v) >= 1000 ? v.toLocaleString()
      : Number(v.toFixed(d)).toString()) : String(v));
const fmtPct = (v) => v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`;
const fmtBytes = (b) => {
  if (!b) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB']; let i = 0; let n = b;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${u[i]}`;
};
function ago(ts) {
  if (!ts) return '—';
  const s = (Date.now() - new Date(ts).getTime()) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
function duration(a, b) {
  if (!a) return '—';
  const s = ((b ? new Date(b) : new Date()).getTime() - new Date(a).getTime()) / 1000;
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}
const titleCase = (s) => String(s || '').replace(/[_-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

const STATUS_TONE = {
  queued: '', running: 'run', succeeded: 'good', failed: 'bad', cancelled: '',
  draft: '', evaluated: 'accent', staging: 'run', production: 'good', archived: '',
  ready: 'good', validating: 'run', invalid: 'bad',
  open: 'run', closed: '', generating: 'run', complete: 'good', disputed: 'bad',
};
const statusChip = (s) => h('span', { class: `chip ${STATUS_TONE[s] || ''}` },
  (s === 'running' || s === 'generating') ? h('span', { class: 'dot' }) : null, titleCase(s));

/** The tiny-backend marker. Shown everywhere a tiny-backend number appears so the
 *  distinction between "this ran" and "this is a result" survives a screenshot. */
const tinyChip = () => h('span', {
  class: 'chip warn',
  title: 'Trained on the tiny backend: real optimisation maths, a randomly-initialised ' +
         'two-layer model. Loss curves are genuine; quality numbers are not.',
}, 'tiny backend');

// ---------------------------------------------------------------------- state

const state = {
  token: localStorage.getItem('foundry_token') || null,
  user: null, catalog: null, view: 'overview',
  datasets: [], runs: [], models: [], batches: [], queue: null,
  reviewMode: 'queue', reviewItem: null, reviewStart: 0,
  form: { method: null, params: {}, advanced: false },
  timers: {}, chartOff: new Set(['lr', 'grad_norm', 'tokens', 'step', 'total', 'iteration']),
};

// ------------------------------------------------------------------------ api

async function api(path, { method = 'GET', body, raw, query } = {}) {
  const url = new URL(path, location.origin);
  for (const [k, v] of Object.entries(query || {})) {
    if (v !== null && v !== undefined && v !== '') url.searchParams.set(k, v);
  }
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (body && !raw) headers['Content-Type'] = 'application/json';

  const res = await fetch(url, {
    method, headers, body: raw ? body : (body ? JSON.stringify(body) : undefined),
  });
  if (res.status === 401) { logout(); throw new Error('session expired'); }
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(data?.detail
    ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
    : `request failed (${res.status})`);
  return data;
}

function toast(message, tone = '') {
  const el = h('div', { class: `toast ${tone}`, text: message });
  $('#toasts').append(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 4200);
}

// ----------------------------------------------------------------------- auth

$('#login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const err = $('#login-error');
  err.textContent = '';
  try {
    const out = await api('/api/login', {
      method: 'POST',
      body: { email: $('#login-email').value.trim(), password: $('#login-password').value },
    });
    state.token = out.token;
    localStorage.setItem('foundry_token', out.token);
    state.user = out.user;
    await boot();
  } catch (e2) { err.textContent = e2.message; }
});

function logout() {
  if (state.token) api('/api/logout', { method: 'POST' }).catch(() => {});
  state.token = null; state.user = null;
  localStorage.removeItem('foundry_token');
  Object.values(state.timers).forEach(clearInterval);
  state.timers = {};
  $('#app-view').hidden = true;
  $('#login-view').hidden = false;
}
$('#logout').addEventListener('click', logout);

const ROLE_RANK = { viewer: 0, member: 1, reviewer: 2, lead: 3, admin: 4 };
const can = (min) => ROLE_RANK[state.user?.role] >= ROLE_RANK[min];

// ----------------------------------------------------------------------- boot

async function boot() {
  state.user = state.user || await api('/api/me');
  state.catalog = await api('/api/catalog');

  $('#login-view').hidden = true;
  $('#app-view').hidden = false;
  $('#user-name').textContent = state.user.name;
  $('#user-role').textContent = state.user.role;
  $('#user-avatar').textContent = state.user.name.split(/\s+/).map(w => w[0]).join('').slice(0, 2).toUpperCase();
  $('#team-name').textContent = `· ${state.user.team_name}`;
  $('#ov-team').textContent = state.user.team_name;

  $$('[data-min-role]').forEach(el => { el.hidden = !can(el.dataset.minRole); });

  initUpload();
  initTrainForm();
  await refreshAll();
  go(state.view);

  clearInterval(state.timers.poll);
  // 5s: fast enough that a queued job appears to start promptly, slow enough that
  // a room full of open tabs is not a load source of its own.
  state.timers.poll = setInterval(pollLight, 5000);
}

async function refreshAll() {
  const [datasets, runs, models, batches, queue] = await Promise.all([
    api('/api/datasets'), api('/api/runs'), api('/api/models'),
    api('/api/review/batches'), api('/api/queue'),
  ]);
  Object.assign(state, { datasets, runs, models, batches, queue });
  updateCounts();
}

async function pollLight() {
  try {
    const [runs, queue] = await Promise.all([api('/api/runs'), api('/api/queue')]);
    state.runs = runs; state.queue = queue;
    updateCounts();
    if (state.view === 'queue') renderQueue();
    if (state.view === 'train') renderRunList();
    if (state.view === 'overview') renderOverview();
    if (state.openRunId) refreshRunDrawer(state.openRunId);
  } catch { /* transient — the next tick retries */ }
}

function updateCounts() {
  const active = state.runs.filter(r => r.status === 'running' || r.status === 'queued').length;
  badge('#count-runs', active);
  badge('#count-queue', (state.queue?.active || []).length);
  const pending = state.batches.reduce((n, b) => n + (b.items - b.reviewed), 0);
  badge('#count-review', pending, pending > 0 && can('reviewer'));
}
function badge(sel, n, show = null) {
  const el = $(sel); if (!el) return;
  el.textContent = n > 99 ? '99+' : String(n);
  el.hidden = show === null ? !n : !show;
}

// --------------------------------------------------------------------- router

$$('.tab').forEach(tab => tab.addEventListener('click', () => go(tab.dataset.view)));
$$('[data-goto]').forEach(el => el.addEventListener('click', () => go(el.dataset.goto)));

function go(view) {
  state.view = view;
  $$('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
  $$('main.view').forEach(m => { m.hidden = m.id !== `view-${view}`; });
  ({ overview: renderOverview, data: renderData, train: renderTrain,
     review: renderReview, models: renderModels, queue: renderQueue }[view] || (() => {}))();
}

// ------------------------------------------------------------------- overview

function renderOverview() {
  const runs = state.runs;
  const prod = state.models.filter(m => m.status === 'production');
  const active = runs.filter(r => ['queued', 'running'].includes(r.status));
  const pairs = state.batches.reduce((n, b) => n + b.pairs, 0);

  const banner = $('#ov-banner');
  clear(banner);
  const workers = state.queue?.workers || [];
  if (!workers.some(w => w.alive)) {
    banner.append(h('div', { class: 'banner bad' },
      h('strong', { text: 'No live workers.' }),
      ' Jobs will queue but nothing will run. Start one with ',
      h('code', { text: 'python -m foundry.worker' }), '.'));
  } else if (runs.some(r => r.backend === 'tiny' && r.status === 'succeeded')) {
    banner.append(h('div', { class: 'banner' },
      h('strong', { text: 'Some runs used the tiny backend.' }),
      ' The requested base model was not available locally, so training ran against a ' +
      'randomly-initialised two-layer model. The optimisation is real; the quality ' +
      'numbers are not. Download the base model, or set the backend explicitly.'));
  }

  set($('#ov-stats'),
    statCard('Datasets', state.datasets.length,
      `${state.datasets.reduce((n, d) => n + d.num_rows, 0).toLocaleString()} rows`),
    statCard('Runs in flight', active.length,
      `${runs.filter(r => r.status === 'succeeded').length} succeeded all time`),
    statCard('Preference pairs', pairs.toLocaleString(), 'collected from review'),
    statCard('In production', prod.length,
      prod.length ? prod.map(m => `${m.name} v${m.version}`).join(', ') : 'nothing promoted yet'));

  set($('#ov-runs'), runs.length
    ? runTable(runs.slice(0, 6))
    : emptyState('◇', 'No runs yet',
        'Upload a demonstration dataset, then run supervised fine-tuning on it.',
        h('button', { class: 'btn primary', onClick: () => go('data') }, 'Upload data')));

  set($('#ov-production'), prod.length
    ? h('div', {}, prod.map(m => h('div', {
        class: 'card-body tight', style: 'border-bottom:1px solid var(--border);cursor:pointer',
        onClick: () => openModel(m.id),
      },
      h('div', { class: 'row auto', style: 'justify-content:space-between' },
        h('span', { class: 't-main', text: `${m.name} v${m.version}` }),
        h('span', { class: 'chip good' }, 'production')),
      h('div', { class: 't-sub' }, `${titleCase(m.method)} · `,
        m.headline?.value != null ? `${m.headline.label} ${fmtNum(m.headline.value)}` : 'not benchmarked'))))
    : h('div', { class: 'card-body' }, h('p', { class: 'muted small', text:
        'Nothing in production. Benchmark a model version, then promote it.' })));

  const q = state.queue;
  set($('#ov-cluster'),
    h('dl', { class: 'kv' },
      h('dt', { text: 'Workers' }),
      h('dd', { text: `${workers.filter(w => w.alive).length} live / ${workers.length}` }),
      h('dt', { text: 'Cluster queued' }), h('dd', { text: String(q?.cluster_queued ?? 0) }),
      h('dt', { text: 'Your concurrency' }),
      h('dd', { text: `${q?.team?.running ?? 0} / ${q?.team?.concurrency ?? 0}` }),
      h('dt', { text: 'Judge' }),
      h('dd', { text: workers[0]?.capabilities?.judge || '—' })));
}

const statCard = (label, value, foot) => h('div', { class: 'card stat' },
  h('div', { class: 'label', text: label }),
  h('div', { class: 'value', text: String(value) }),
  h('div', { class: 'foot', text: foot }));

const emptyState = (icon, title, body, action) => h('div', { class: 'empty' },
  h('div', { class: 'icon', text: icon }), h('h3', { text: title }),
  h('p', { text: body }), action);

// ----------------------------------------------------------------------- data

function initUpload() {
  const kindSel = $('#up-kind');
  const filter = $('#data-filter');
  clear(kindSel); 
  state.catalog.dataset_kinds.forEach(k => {
    kindSel.append(h('option', { value: k.kind, text: titleCase(k.kind) }));
    if (!$(`option[value="${k.kind}"]`, filter)) {
      filter.append(h('option', { value: k.kind, text: titleCase(k.kind) }));
    }
  });
  const help = () => {
    const k = state.catalog.dataset_kinds.find(x => x.kind === kindSel.value);
    $('#up-kind-help').textContent = k ? k.help : '';
  };
  kindSel.addEventListener('change', help); help();
  filter.addEventListener('change', renderData);

  const dz = $('#dropzone'), file = $('#up-file');
  dz.addEventListener('click', () => file.click());
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('over');
    if (e.dataTransfer.files[0]) { file.files = e.dataTransfer.files; onPick(); }
  });
  file.addEventListener('change', onPick);
  $('#up-name').addEventListener('input', onPick);

  function onPick() {
    const f = file.files[0];
    $('#drop-title').textContent = f ? `${f.name} · ${fmtBytes(f.size)}` : 'Drop a file, or click to choose';
    if (f && !$('#up-name').value) $('#up-name').value = f.name.replace(/\.[^.]+$/, '');
    $('#up-submit').disabled = !(f && $('#up-name').value.trim());
  }

  $('#up-submit').addEventListener('click', async () => {
    const f = file.files[0]; if (!f) return;
    const btn = $('#up-submit');
    btn.disabled = true; set(btn, h('span', { class: 'spinner' }), ' Validating…');
    $('#up-error').textContent = '';
    try {
      const ds = await api('/api/datasets/upload', {
        method: 'POST', raw: true, body: f,
        query: { name: $('#up-name').value.trim(), kind: kindSel.value,
                 description: $('#up-desc').value },
      });
      toast(ds.status === 'ready'
        ? `${ds.name} v${ds.version}: ${ds.num_rows.toLocaleString()} rows accepted` +
          (ds.num_bad_rows ? `, ${ds.num_bad_rows} dropped` : '')
        : `${ds.name}: no usable rows — see the validation errors`,
        ds.status === 'ready' ? 'good' : 'bad');
      $('#up-name').value = ''; $('#up-desc').value = ''; file.value = '';
      $('#drop-title').textContent = 'Drop a file, or click to choose';
      state.datasets = await api('/api/datasets');
      renderData();
      if (ds.num_bad_rows) openDataset(ds.id);
    } catch (e) { $('#up-error').textContent = e.message; }
    finally { btn.disabled = false; btn.textContent = 'Upload & validate'; }
  });
}

function renderData() {
  const kind = $('#data-filter').value;
  const rows = state.datasets.filter(d => !kind || d.kind === kind);
  set($('#data-list'), rows.length ? h('table', { class: 'data' },
    h('thead', {}, h('tr', {},
      ...['Name', 'Kind', 'Rows', 'Size', 'Status', 'Added'].map(t => h('th', { text: t })))),
    h('tbody', {}, rows.map(d => h('tr', { class: 'clickable', onClick: () => openDataset(d.id) },
      h('td', {}, h('div', { class: 't-main', text: `${d.name} v${d.version}` }),
        d.description ? h('div', { class: 't-sub', text: d.description }) : null),
      h('td', {}, h('span', { class: 'chip', text: d.kind })),
      h('td', { class: 'num', text: d.num_rows.toLocaleString() },
        d.num_bad_rows ? h('div', { class: 't-sub', text: `${d.num_bad_rows} dropped` }) : null),
      h('td', { class: 'num', text: fmtBytes(d.bytes) }),
      h('td', {}, statusChip(d.status)),
      h('td', { class: 't-sub', text: ago(d.created_at) })))))
    : emptyState('⇪', 'No datasets yet',
        'Upload demonstrations to fine-tune on, prompts to sample from, preference pairs, ' +
        'or a benchmark to score against.'));
}

async function openDataset(id) {
  const d = await api(`/api/datasets/${id}`);
  openDrawer(`${d.name} v${d.version}`, [
    h('div', { class: 'chips', style: 'margin-bottom:14px' },
      h('span', { class: 'chip', text: d.kind }), statusChip(d.status),
      h('span', { class: 'chip', text: `${d.num_rows.toLocaleString()} rows` }),
      h('span', { class: 'chip', text: fmtBytes(d.bytes) }),
      h('span', { class: 'chip mono', text: `sha ${d.sha256}` })),
    d.description ? h('p', { class: 'muted', text: d.description }) : null,

    d.errors?.length ? h('div', { class: 'card', style: 'margin-bottom:14px' },
      h('div', { class: 'card-head' }, h('h3', { text: `${d.num_bad_rows} rows dropped` })),
      h('div', { class: 'card-body tight' },
        h('table', { class: 'data' }, h('tbody', {}, d.errors.map(e =>
          h('tr', {}, h('td', { class: 'num', style: 'width:70px', text: `line ${e.line}` }),
            h('td', { class: 'small', text: e.error }))))))) : null,

    Object.keys(d.stats || {}).length ? h('div', { class: 'card', style: 'margin-bottom:14px' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Statistics' })),
      h('div', { class: 'card-body' }, h('dl', { class: 'kv' },
        ...Object.entries(d.stats).flatMap(([k, v]) => [
          h('dt', { text: titleCase(k) }),
          h('dd', { text: typeof v === 'object' && v
            ? Object.entries(v).map(([kk, vv]) => `${kk} ${fmtNum(vv, 1)}`).join(' · ')
            : fmtNum(v) })])))) : null,

    h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Preview' })),
      h('div', { class: 'card-body' }, d.preview.map((row, i) =>
        h('div', { style: i ? 'margin-top:14px;padding-top:14px;border-top:1px solid var(--border)' : '' },
          ...Object.entries(row).map(([k, v]) => h('div', { style: 'margin-bottom:6px' },
            h('div', { class: 'field-label', style: 'margin:0', text: titleCase(k) }),
            h('div', { class: 'small', style: 'white-space:pre-wrap',
                       text: Array.isArray(v) ? v.join(' | ') : String(v) })))))))
  ]);
}

// ---------------------------------------------------------------------- train

function initTrainForm() {
  const base = $('#run-base');
  clear(base);
  state.catalog.base_models.forEach(b => base.append(h('option', { value: b.id, text: b.label })));

  set($('#method-cards'), state.catalog.method_order.map(key => {
    const m = state.catalog.methods[key];
    return h('button', { class: 'method', dataset: { method: key }, type: 'button',
      onClick: () => selectMethod(key) },
      h('div', { class: 'method-head' },
        h('span', { class: 'method-name', text: m.label }),
        h('span', { class: `chip ${key === 'sft' ? 'accent' : ''}`, text: m.family })),
      h('div', { class: 'method-blurb', text: m.blurb }));
  }));

  $('#show-advanced').addEventListener('change', e => {
    state.form.advanced = e.target.checked; renderParams();
  });
  $('#train-toggle-form').addEventListener('click', () => {
    const w = $('#train-form-wrap'); w.hidden = !w.hidden;
    if (!w.hidden && !state.form.method) selectMethod('sft');
  });
  $('#train-cancel-form').addEventListener('click', () => { $('#train-form-wrap').hidden = true; });
  $('#run-submit').addEventListener('click', submitRun);
  $('#run-filter').addEventListener('change', renderRunList);
}

function selectMethod(key) {
  state.form.method = key;
  state.form.params = Object.fromEntries(
    state.catalog.methods[key].params.map(p => [p.name, p.default]));
  $$('.method').forEach(el => el.classList.toggle('selected', el.dataset.method === key));
  $('#method-detail').hidden = false;
  if (!$('#run-name').value) {
    $('#run-name').value = `${state.user.team_slug}-${key}-${state.runs.length + 1}`;
  }
  renderSources();
  renderParams();
}

function renderSources() {
  const m = state.catalog.methods[state.form.method];
  const needs = m.requires || {};
  const wrap = clear($('#run-sources'));

  const dsSelect = (id, kind, label, help, optional) => {
    const opts = state.datasets.filter(d => d.kind === kind && d.status === 'ready');
    return h('label', { class: 'field' },
      h('span', { class: 'field-label', text: label }),
      h('select', { id }, h('option', { value: '', text: optional ? '— none —' : '— choose —' }),
        ...opts.map(d => h('option', { value: d.id,
          text: `${d.name} v${d.version} · ${d.num_rows.toLocaleString()} rows` }))),
      h('span', { class: 'field-help', text: opts.length ? help
        : `No ready '${kind}' datasets. Upload one on the Data tab.` }));
  };

  if (needs.train) {
    const isPref = needs.train === 'preference';
    wrap.append(dsSelect('run-train-ds', needs.train, `Training data (${needs.train})`,
      isPref ? 'Leave empty to use the preference pairs collected in the review console.'
             : 'Rows are split 90/10 by prompt hash for validation.',
      isPref));
  }
  wrap.append(dsSelect('run-eval-ds', 'benchmark', 'Benchmark (optional)',
    'Scored automatically when the run finishes, and it is what enables promotion.', true));

  const policies = state.models.filter(m2 => m2.artifact_kind === 'adapter');
  wrap.append(h('label', { class: 'field' },
    h('span', { class: 'field-label', text: needs.policy ? 'Start from' : 'Start from (optional)' }),
    h('select', { id: 'run-parent' }, h('option', { value: '', text: '— base model —' }),
      ...policies.map(p => h('option', { value: p.id,
        text: `${p.name} v${p.version} · ${titleCase(p.method)}` }))),
    h('span', { class: 'field-help', text: needs.policy
      ? 'Preference methods improve an existing policy. Pick the SFT model you trained.'
      : 'Continue from an earlier adapter instead of the raw base model.' })));

  if (needs.reward_model) {
    const rms = state.models.filter(m2 => m2.artifact_kind === 'reward');
    wrap.append(h('label', { class: 'field' },
      h('span', { class: 'field-label', text: 'Reward model' }),
      h('select', { id: 'run-reward' }, h('option', { value: '', text: '— choose —' }),
        ...rms.map(p => h('option', { value: p.id, text: `${p.name} v${p.version}` }))),
      h('span', { class: 'field-help', text: rms.length
        ? 'PPO scores its rollouts with this.'
        : 'No reward models yet. Train one with the Reward model method first.' })));
  }
}

function renderParams() {
  const m = state.catalog.methods[state.form.method];
  const wrap = clear($('#run-params'));
  m.params.filter(p => state.form.advanced || !p.advanced).forEach(p => {
    let input;
    if (p.type === 'bool') {
      input = h('label', { class: 'switch' },
        h('input', { type: 'checkbox', checked: !!state.form.params[p.name],
          onChange: e => { state.form.params[p.name] = e.target.checked; } }),
        h('span', { text: p.label }));
      wrap.append(h('div', { class: 'field' }, input,
        p.help ? h('span', { class: 'field-help', text: p.help }) : null));
      return;
    }
    if (p.type === 'enum') {
      input = h('select', { onChange: e => {
        state.form.params[p.name] = e.target.value;
        if (p.name === 'optimizer') renderParams();
      } }, ...p.options.map(o => h('option', { value: o.value, text: o.label,
        selected: state.form.params[p.name] === o.value })));
    } else {
      input = h('input', { type: 'number', value: state.form.params[p.name],
        step: p.step || (p.type === 'int' ? 1 : 'any'), min: p.min, max: p.max,
        onInput: e => { state.form.params[p.name] = e.target.value; } });
    }
    wrap.append(h('label', { class: 'field' },
      h('span', { class: 'field-label', text: p.label },
        p.advanced ? h('span', { class: 'chip', text: 'adv' }) : null),
      input, p.help ? h('span', { class: 'field-help', text: p.help }) : null));
  });
}

async function submitRun() {
  const err = $('#run-error'); err.textContent = '';
  const btn = $('#run-submit');
  const val = (id) => { const el = $(id); return el && el.value ? Number(el.value) : null; };
  const body = {
    name: $('#run-name').value.trim(),
    method: state.form.method,
    base_model: $('#run-base').value,
    params: state.form.params,
    lora: { r: Number($('#lora-r').value), alpha: Number($('#lora-alpha').value),
            dropout: Number($('#lora-dropout').value) },
    train_dataset_id: val('#run-train-ds'),
    eval_dataset_id: val('#run-eval-ds'),
    parent_model_version_id: val('#run-parent'),
    reward_model_version_id: val('#run-reward'),
    backend: $('#run-backend').value,
  };
  if (!body.name) { err.textContent = 'Give the run a name.'; return; }

  btn.disabled = true; set(btn, h('span', { class: 'spinner' }), ' Queueing…');
  try {
    const run = await api('/api/runs', { method: 'POST', body });
    (run.warnings || []).forEach(w => toast(`Adjusted: ${w}`, 'warn'));
    toast(`Run #${run.id} queued`, 'good');
    $('#train-form-wrap').hidden = true;
    await refreshAll(); renderTrain(); openRun(run.id);
  } catch (e) { err.textContent = e.message; }
  finally { btn.disabled = false; btn.textContent = 'Queue run'; }
}

const RAIL = [
  { key: 'data', title: 'Data', sub: 'demonstrations, prompts, benchmarks' },
  { key: 'sft', title: 'Fine-tune', sub: 'SFT with LoRA' },
  { key: 'review', title: 'Collect feedback', sub: 'humans or an AI judge' },
  { key: 'align', title: 'Align', sub: 'DPO · PPO · GRPO · GSPO · RLAIF' },
  { key: 'evaluate', title: 'Benchmark', sub: 'score on held-out data' },
  { key: 'promote', title: 'Promote', sub: 'ship to production' },
];

function renderTrain() {
  const done = {
    data: state.datasets.some(d => d.status === 'ready'),
    sft: state.runs.some(r => r.method === 'sft' && r.status === 'succeeded'),
    review: state.batches.some(b => b.pairs > 0),
    align: state.runs.some(r => ['dpo', 'ppo', 'grpo', 'gspo', 'rlaif'].includes(r.method)
                                && r.status === 'succeeded'),
    evaluate: state.models.some(m => m.headline?.value != null),
    promote: state.models.some(m => m.status === 'production'),
  };
  set($('#pipeline-rail'), RAIL.map((s, i) => h('button', {
    class: `rail-step ${done[s.key] ? 'done' : ''}`, type: 'button',
    onClick: () => {
      if (s.key === 'data') return go('data');
      if (s.key === 'review') return go('review');
      if (s.key === 'promote' || s.key === 'evaluate') return go('models');
      $('#train-form-wrap').hidden = false;
      selectMethod(s.key === 'sft' ? 'sft' : 'dpo');
    },
  },
    h('span', { class: 'rail-num', text: done[s.key] ? '✓' : String(i + 1) }),
    h('span', { class: 'rail-text' },
      h('span', { class: 'rail-title', text: s.title }),
      h('span', { class: 'rail-sub', text: s.sub })))));
  renderRunList();
}

function renderRunList() {
  const f = $('#run-filter')?.value;
  const rows = state.runs.filter(r => !f || r.status === f);
  set($('#run-list'), rows.length ? runTable(rows)
    : emptyState('◈', 'No runs', 'Start with supervised fine-tuning on a demonstration dataset.',
        h('button', { class: 'btn primary', onClick: () => { $('#train-form-wrap').hidden = false;
          selectMethod('sft'); } }, 'New run')));
}

function runTable(runs) {
  return h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ...['Run', 'Method', 'Status', 'Progress', 'Headline', 'Started']
      .map(t => h('th', { text: t })))),
    h('tbody', {}, runs.map(r => {
      const pct = r.job?.progress?.pct ?? (r.status === 'succeeded' ? 100 : 0);
      const head = r.metrics?.headline;
      const key = r.metrics ? Object.keys(r.metrics).find(k =>
        ['heldout_pref_acc', 'best_val_loss', 'final_reward', 'heldout_pair_acc',
         'final_loss'].includes(k)) : null;
      return h('tr', { class: 'clickable', onClick: () => openRun(r.id) },
        h('td', {}, h('div', { class: 't-main', text: r.name }),
          h('div', { class: 't-sub', text: `#${r.id} · ${r.base_model}` })),
        h('td', {}, h('span', { class: 'chip', text: r.method.toUpperCase() })),
        h('td', {}, h('div', { class: 'chips' }, statusChip(r.status),
          r.backend === 'tiny' ? tinyChip() : null)),
        h('td', { style: 'width:130px' },
          h('div', { class: 'rail-bar' }, h('div', {
            class: `rail-fill ${r.status === 'succeeded' ? 'good' : r.status === 'failed' ? 'bad' : ''}`,
            style: `width:${Math.min(100, pct)}%` })),
          h('div', { class: 't-sub', text: r.job?.progress?.step
            ? `step ${r.job.progress.step}/${r.job.progress.total}` : `${Math.round(pct)}%` })),
        h('td', { class: 'num', text: head?.value != null ? fmtNum(head.value)
          : (key ? fmtNum(r.metrics[key]) : '—') },
          key ? h('div', { class: 't-sub', text: titleCase(key) }) : null),
        h('td', { class: 't-sub', text: r.started_at ? ago(r.started_at) : ago(r.created_at) }));
    })));
}

// ------------------------------------------------------------- run drawer

async function openRun(id) {
  state.openRunId = id;
  openDrawer('Run', [h('div', { class: 'empty' }, h('span', { class: 'spinner' }))],
    () => { state.openRunId = null; state.runCursor = 0; });
  state.runCursor = 0;
  await refreshRunDrawer(id, true);
}

async function refreshRunDrawer(id, full = false) {
  const body = $('#drawer-body'); if (!body) return;
  const run = await api(`/api/runs/${id}`);
  const { events, cursor } = await api(`/api/runs/${id}/events`,
    { query: { after: full ? 0 : (state.runCursor || 0) } });
  if (full) state.runLog = [];
  state.runLog = (state.runLog || []).concat(events).slice(-400);
  state.runCursor = cursor;

  $('#drawer-title').textContent = `${run.name} · #${run.id}`;
  const pct = run.job?.progress?.pct ?? (run.status === 'succeeded' ? 100 : 0);
  const latest = run.job?.progress?.latest || {};

  set(body,
    h('div', { class: 'chips', style: 'margin-bottom:12px' },
      statusChip(run.status), h('span', { class: 'chip accent', text: run.method.toUpperCase() }),
      h('span', { class: 'chip', text: run.base_model }),
      run.backend === 'tiny' ? tinyChip() : null,
      h('span', { class: 'chip', text: duration(run.started_at, run.finished_at) })),

    run.error ? h('div', { class: 'banner bad' }, h('strong', { text: 'Failed. ' }), run.error) : null,

    h('div', { class: 'rail-bar', style: 'margin-bottom:6px' }, h('div', {
      class: `rail-fill ${run.status === 'succeeded' ? 'good' : run.status === 'failed' ? 'bad' : ''}`,
      style: `width:${Math.min(100, pct)}%` })),
    h('div', { class: 'row auto', style: 'justify-content:space-between;margin-bottom:14px' },
      h('span', { class: 't-sub', text: run.job?.progress?.step
        ? `step ${run.job.progress.step} of ${run.job.progress.total}` : `${Math.round(pct)}%` }),
      h('div', { class: 'row auto' },
        ['queued', 'running'].includes(run.status)
          ? h('button', { class: 'btn sm danger', onClick: async () => {
              await api(`/api/runs/${run.id}/cancel`, { method: 'POST' });
              toast('Cancellation requested — the trainer stops at the next step boundary');
              refreshRunDrawer(run.id, true);
            } }, 'Cancel') : null,
        run.model_version ? h('button', { class: 'btn sm', onClick: () =>
          openModel(run.model_version.id) }, 'Open model') : null)),

    Object.keys(latest).length ? h('div', { class: 'grid c4', style: 'margin-bottom:14px' },
      ...Object.entries(latest).filter(([k]) => !['pct', 'elapsed_s'].includes(k)).slice(0, 4)
        .map(([k, v]) => h('div', { class: 'card stat' },
          h('div', { class: 'label', text: titleCase(k) }),
          h('div', { class: 'value sm', text: fmtNum(v) })))) : null,

    run.series.length ? h('div', { class: 'card', style: 'margin-bottom:14px' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Training curves' })),
      h('div', { class: 'card-body' }, chart(run.series))) : null,

    Object.keys(run.metrics || {}).length ? h('div', { class: 'card', style: 'margin-bottom:14px' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Final metrics' })),
      h('div', { class: 'card-body' }, h('dl', { class: 'kv' },
        ...Object.entries(run.metrics).filter(([, v]) => typeof v !== 'object')
          .map(([k, v]) => [h('dt', { text: titleCase(k) }), h('dd', { text: fmtNum(v) })]).flat()))) : null,

    run.samples?.length ? h('div', { class: 'card', style: 'margin-bottom:14px' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Sample generations' }),
        h('span', { class: 'chip', text: `${run.samples.length}` })),
      h('div', { class: 'card-body' }, run.samples.slice(0, 5).map((s, i) =>
        h('div', { style: i ? 'margin-top:12px;padding-top:12px;border-top:1px solid var(--border)' : '' },
          h('div', { class: 'field-label', style: 'margin:0', text: 'Prompt' }),
          h('div', { class: 'small muted', style: 'white-space:pre-wrap;margin-bottom:6px',
                     text: String(s.prompt).slice(0, 400) }),
          h('div', { class: 'field-label', style: 'margin:0', text: 'Generated' }),
          h('div', { class: 'small', style: 'white-space:pre-wrap', text: String(s.generated).slice(0, 600) }),
          s.reward !== undefined ? h('div', { class: 'chips', style: 'margin-top:6px' },
            h('span', { class: 'chip', text: `reward ${fmtNum(s.reward)}` }),
            s.advantage !== undefined ? h('span', { class: 'chip', text: `advantage ${fmtNum(s.advantage)}` }) : null) : null)))) : null,

    h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Log' })),
      h('div', { class: 'card-body tight' }, logView(state.runLog))),

    h('details', { style: 'margin-top:14px' },
      h('summary', { class: 'field-label', style: 'cursor:pointer', text: 'Resolved configuration' }),
      h('pre', { class: 'log', style: 'margin-top:8px',
        text: JSON.stringify({ params: run.params, lora: run.lora }, null, 2) })));
}

function logView(events) {
  const box = h('div', { class: 'log' }, ...events.map(e => h('div', {},
    h('span', { class: 'ts', text: new Date(e.ts).toLocaleTimeString() + '  ' }),
    h('span', { class: e.level, text: e.message }),
    Object.keys(e.data || {}).length
      ? h('span', { class: 'metric', text: '  ' + Object.entries(e.data)
          .map(([k, v]) => `${k}=${fmtNum(v, 3)}`).join(' ') }) : null)));
  requestAnimationFrame(() => { box.scrollTop = box.scrollHeight; });
  return box;
}

// ---------------------------------------------------------------------- chart

const SERIES_COLOURS = ['#4f46e5', '#1d6fd0', '#0f7a4d', '#a3620a', '#b42318',
                        '#7c3aed', '#0891b2', '#c2410c'];

function chart(series) {
  const keys = [...new Set(series.flatMap(Object.keys))]
    .filter(k => !['step', 'total', 'elapsed_s', 'phase'].includes(k))
    .filter(k => series.some(r => typeof r[k] === 'number'));
  const active = keys.filter(k => !state.chartOff.has(k)).slice(0, 6);

  const W = 640, H = 190, PAD = { l: 46, r: 12, t: 10, b: 24 };
  // createElementNS, not createElement: an <svg> built by document.createElement is
  // an HTML element that happens to be called "svg" and renders nothing at all.
  const NS = 'http://www.w3.org/2000/svg';
  const node = (tag, a) => { const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); return e; };
  const svg = node('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`,
                            preserveAspectRatio: 'none' });

  const xs = series.map((r, i) => r.step ?? i);
  const xMin = Math.min(...xs), xMax = Math.max(...xs, xMin + 1);
  const px = (x) => PAD.l + (x - xMin) / (xMax - xMin) * (W - PAD.l - PAD.r);

  // Each series is scaled to its own range. Loss and accuracy on one axis would
  // flatten whichever has the smaller span into a straight line.
  for (let i = 0; i < 4; i++) {
    const y = PAD.t + i * (H - PAD.t - PAD.b) / 3;
    // Themed colours go through `style`, since a presentation *attribute* is not
    // parsed as CSS and would leave var(--border) as an unresolved literal.
    svg.append(node('line', { x1: PAD.l, x2: W - PAD.r, y1: y, y2: y,
      style: 'stroke:var(--border)', 'stroke-width': 1 }));
  }

  active.forEach((key, ki) => {
    const pts = series.map((r, i) => [xs[i], r[key]]).filter(p => typeof p[1] === 'number');
    if (pts.length < 2) return;
    const vals = pts.map(p => p[1]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const span = (hi - lo) || 1;
    const py = (v) => PAD.t + (1 - (v - lo) / span) * (H - PAD.t - PAD.b);
    svg.append(node('path', {
      d: pts.map((p, i) => `${i ? 'L' : 'M'}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join(''),
      fill: 'none', stroke: SERIES_COLOURS[ki % SERIES_COLOURS.length],
      'stroke-width': 1.8, 'stroke-linejoin': 'round',
    }));
    // Element.append() returns undefined — build the label, set its text, then attach.
    const label = node('text', { x: 4, y: PAD.t + 9 + ki * 12, 'font-size': 9,
      fill: SERIES_COLOURS[ki % SERIES_COLOURS.length] });
    label.textContent = fmtNum(hi, 2);
    svg.append(label);
  });

  const legend = h('div', { class: 'chart-legend' }, ...keys.map((k, i) => {
    const off = state.chartOff.has(k);
    const idx = active.indexOf(k);
    return h('button', { class: off ? 'off' : '', type: 'button', onClick: () => {
      off ? state.chartOff.delete(k) : state.chartOff.add(k);
      if (state.openRunId) refreshRunDrawer(state.openRunId, false);
    } },
      h('span', { class: 'swatch', style:
        `background:${idx >= 0 ? SERIES_COLOURS[idx % SERIES_COLOURS.length] : 'var(--n400)'}` }),
      titleCase(k));
  }));

  return h('div', {}, svg, legend);
}

// --------------------------------------------------------------------- review

['queue', 'batches', 'stats', 'disputed'].forEach(mode => {
  const btn = $(`#review-mode-${mode}`);
  if (btn) btn.addEventListener('click', () => { state.reviewMode = mode; renderReview(); });
});

function renderReview() {
  $$('#view-review .view-head .btn').forEach(b => {
    b.classList.toggle('primary', b.id === `review-mode-${state.reviewMode}`);
    b.classList.toggle('quiet', b.id !== `review-mode-${state.reviewMode}`);
  });
  ({ queue: renderReviewQueue, batches: renderBatches, stats: renderReviewStats,
     disputed: renderDisputed }[state.reviewMode])();
}

async function renderReviewQueue() {
  const panel = set($('#review-panel'), h('div', { class: 'empty' }, h('span', { class: 'spinner' })));
  const { item, queue_remaining } = await api('/api/review/next');
  state.reviewItem = item;
  state.reviewStart = Date.now();

  if (!item) {
    set(panel, emptyState('✓', 'Queue is clear',
      queue_remaining ? `${queue_remaining} items are waiting on other reviewers.`
        : 'Create a review batch to sample candidate responses from a model.',
      h('button', { class: 'btn primary', onClick: () => { state.reviewMode = 'batches';
        renderReview(); } }, 'Review batches')));
    return;
  }

  let picked = null;
  const cards = h('div', { class: `candidates ${item.candidates.length === 2 ? 'two' : ''}` });
  item.candidates.forEach((c, i) => {
    const card = h('div', { class: 'candidate' },
      h('div', { class: 'candidate-head' },
        h('span', { class: 'candidate-key', text: String(i + 1) }),
        h('span', { class: 'small muted', text: `Response ${String.fromCharCode(65 + i)}` }),
        h('span', { class: 'small muted', style: 'margin-left:auto',
                    text: `${c.text.length} chars` })),
      h('div', { class: 'candidate-body', text: c.text }),
      h('div', { class: 'candidate-foot' },
        h('button', { class: 'btn block', onClick: () => choose(c.id) },
          'Pick this ', h('span', { class: 'kbd', text: String(i + 1) }))));
    card._cid = c.id;
    cards.append(card);
  });

  function choose(id) {
    picked = id;
    $$('.candidate', cards).forEach(el => el.classList.toggle('picked', el._cid === id));
    $('#submit-verdict').disabled = false;
  }

  set(panel, h('div', { class: 'grid review' },
    h('div', {},
      h('div', { class: 'card prompt-card' },
        h('div', { class: 'card-head' },
          h('h3', { text: 'Prompt' }),
          h('span', { class: 'chip', text: item.batch_name }),
          item.uncertainty > 0.5 ? h('span', { class: 'chip warn',
            title: 'The candidates differ a lot — your judgement is worth more here',
            text: 'high signal' }) : null),
        h('div', { class: 'card-body' }, h('blockquote', { text: item.prompt }))),
      cards,
      h('div', { class: 'verdict-row' },
        h('button', { class: 'btn', onClick: () => choose('tie') },
          'Equally good ', h('span', { class: 'kbd', text: 'T' })),
        h('button', { class: 'btn', onClick: () => choose('both_bad') },
          'Both bad ', h('span', { class: 'kbd', text: 'B' })),
        h('button', { class: 'btn quiet', onClick: skip },
          'Skip ', h('span', { class: 'kbd', text: 'S' }))),
      item.ai_suggestion ? h('details', { class: 'ai-disclose' },
        h('summary', { text: `An AI judge has already reviewed this — reveal its answer` }),
        h('div', { class: 'inner' },
          h('p', { text: `${item.ai_suggestion.model} chose: ${item.ai_suggestion.choice}` }),
          h('p', { class: 'small', text: item.ai_suggestion.rationale }),
          h('p', { class: 'small muted', text:
            'Collapsed by default on purpose. A reviewer who reads the judge first is ' +
            'confirming it, not judging — and this batch exists to measure the judge.' }))) : null),

    h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Your verdict' }),
        h('span', { class: 'chip', text: `${queue_remaining} left` })),
      h('div', { class: 'card-body' },
        h('label', { class: 'field' },
          h('span', { class: 'field-label', text: 'Why? (optional but valuable)' }),
          h('textarea', { id: 'verdict-why',
            placeholder: 'What decided it? A rationale is what makes a disputed item resolvable.' })),
        h('label', { class: 'field' },
          h('span', { class: 'field-label', text: 'Confidence' }),
          h('input', { type: 'range', id: 'verdict-conf', min: '0.3', max: '1', step: '0.1',
                       value: '1', style: 'accent-color:var(--accent)' }),
          h('span', { class: 'field-help', text:
            'Low confidence shrinks this item\'s weight in the training loss rather than ' +
            'discarding it.' })),
        h('button', { class: 'btn primary block lg', id: 'submit-verdict', disabled: true,
          onClick: submit }, 'Submit & next'),
        h('div', { class: 'sep' }),
        h('dl', { class: 'kv' },
          h('dt', { text: 'Item' }), h('dd', { text: `#${item.id}` }),
          h('dt', { text: 'Protocol' }), h('dd', { text: item.protocol }),
          h('dt', { text: 'Sampling p' }), h('dd', { text: String(item.sampling_prob) })),
        h('p', { class: 'field-help', text:
          'This item was sampled with the probability shown, and that number is stored ' +
          'with your answer. It is what lets a win-rate measured on reviewed items say ' +
          'anything about the ones nobody reviewed.' })))));

  async function submit() {
    if (!picked) return;
    await api('/api/review/annotate', { method: 'POST', body: {
      item_id: item.id, choice: picked,
      rationale: $('#verdict-why').value,
      confidence: Number($('#verdict-conf').value),
      latency_ms: Date.now() - state.reviewStart,
    } });
    state.batches = await api('/api/review/batches');
    updateCounts();
    renderReviewQueue();
  }
  async function skip() {
    await api('/api/review/skip', { method: 'POST', body: { item_id: item.id } });
    renderReviewQueue();
  }

  panel._keys = (e) => {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    const n = Number(e.key);
    if (n >= 1 && n <= item.candidates.length) choose(item.candidates[n - 1].id);
    else if (e.key.toLowerCase() === 't') choose('tie');
    else if (e.key.toLowerCase() === 'b') choose('both_bad');
    else if (e.key.toLowerCase() === 's') skip();
    else if (e.key === 'Enter' && picked) submit();
  };
}

document.addEventListener('keydown', (e) => {
  const panel = $('#review-panel');
  if (state.view === 'review' && state.reviewMode === 'queue' && panel?._keys) panel._keys(e);
});

async function renderBatches() {
  const panel = $('#review-panel');
  const prompts = state.datasets.filter(d => d.kind === 'prompts' && d.status === 'ready');
  const policies = state.models.filter(m => m.artifact_kind === 'adapter');

  set(panel, h('div', { class: 'grid side' },
    h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Review batches' })),
      state.batches.length ? h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, ...['Batch', 'Protocol', 'Progress', 'Pairs', 'Status', '']
          .map(t => h('th', { text: t })))),
        h('tbody', {}, state.batches.map(b => h('tr', {},
          h('td', {}, h('div', { class: 't-main', text: b.name }),
            h('div', { class: 't-sub', text: `${b.items} items · ${b.annotations_per_item} annotation(s) each` })),
          h('td', {}, h('span', { class: 'chip', text: b.protocol }),
            b.ai_assist_fraction > 0 ? h('span', { class: 'chip warn',
              text: `AI ${Math.round(b.ai_assist_fraction * 100)}%` }) : null),
          h('td', { style: 'width:120px' },
            h('div', { class: 'rail-bar' }, h('div', { class: 'rail-fill',
              style: `width:${b.items ? (b.reviewed / b.items * 100) : 0}%` })),
            h('div', { class: 't-sub', text: `${b.reviewed}/${b.items}` })),
          h('td', { class: 'num', text: String(b.pairs) }),
          h('td', {}, statusChip(b.status)),
          h('td', {}, h('button', { class: 'btn sm', onClick: async () => {
            await api(`/api/review/batches/${b.id}/assemble`, { method: 'POST' });
            toast('Assembling preference pairs…', 'good');
          } }, 'Build pairs')))))) 
        : emptyState('◑', 'No review batches',
            'A batch samples prompts, generates candidate responses from a model, and puts ' +
            'them in front of reviewers.')),

    h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', { text: 'New batch' })),
      h('div', { class: 'card-body' },
        h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Name' }),
          h('input', { type: 'text', id: 'nb-name', placeholder: 'round-2-helpfulness' })),
        h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Prompts' }),
          h('select', { id: 'nb-prompts' }, ...prompts.map(d => h('option', { value: d.id,
            text: `${d.name} v${d.version} · ${d.num_rows} prompts` }))),
          prompts.length ? null : h('span', { class: 'field-help', text:
            'No prompts datasets. Upload one on the Data tab.' })),
        h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Sample from' }),
          h('select', { id: 'nb-policy' }, ...policies.map(m => h('option', { value: m.id,
            text: `${m.name} v${m.version} · ${titleCase(m.method)}` }))),
          policies.length ? null : h('span', { class: 'field-help', text:
            'No trained models yet — run SFT first.' })),
        h('div', { class: 'row' },
          h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Candidates' }),
            h('input', { type: 'number', id: 'nb-k', value: '2', min: '2', max: '8' })),
          h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Reviewers/item' }),
            h('input', { type: 'number', id: 'nb-n', value: '1', min: '1', max: '7' })),
          h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Prompts' }),
            h('input', { type: 'number', id: 'nb-limit', value: '50', min: '1', max: '5000' }))),
        h('label', { class: 'field' },
          h('span', { class: 'field-label', text: 'AI pre-labelling (RLAIF)' }),
          h('input', { type: 'range', id: 'nb-ai', min: '0', max: '1', step: '0.1', value: '0',
                       style: 'accent-color:var(--accent)' }),
          h('span', { class: 'field-help', id: 'nb-ai-help', text:
            'Off. Set above zero to have a judge model label a share of the batch — at 0.1 ' +
            'you are auditing the judge against humans, at 1.0 the humans are auditing it.' })),
        h('div', { class: 'inline-error', id: 'nb-error' }),
        h('button', { class: 'btn primary block', disabled: !prompts.length || !policies.length,
          onClick: createBatch }, 'Create & generate')))));

  const ai = $('#nb-ai');
  ai?.addEventListener('input', () => {
    const v = Number(ai.value);
    $('#nb-ai-help').textContent = v === 0
      ? 'Off. Humans label everything.'
      : v === 1 ? 'Every item is judged by AI. Humans still review whatever they pick up, ' +
                  'and the agreement report compares the two.'
        : `${Math.round(v * 100)}% of items get an AI opinion alongside the human one — ` +
          'that overlap is what measures the judge.';
  });

  async function createBatch() {
    const err = $('#nb-error'); err.textContent = '';
    try {
      await api('/api/review/batches', { method: 'POST', body: {
        name: $('#nb-name').value.trim() || `batch-${state.batches.length + 1}`,
        protocol: 'pairwise',
        prompt_dataset_id: Number($('#nb-prompts').value),
        policy_model_version_id: Number($('#nb-policy').value),
        candidates_per_prompt: Number($('#nb-k').value),
        annotations_per_item: Number($('#nb-n').value),
        ai_assist_fraction: Number($('#nb-ai').value),
        limit: Number($('#nb-limit').value),
      } });
      toast('Batch queued — candidates are being generated', 'good');
      state.batches = await api('/api/review/batches');
      renderBatches();
    } catch (e) { err.textContent = e.message; }
  }
}

async function renderReviewStats() {
  const s = await api('/api/review/stats');
  const kappa = s.human_vs_ai_kappa;
  set($('#review-panel'),
    h('div', { class: 'grid c4', style: 'margin-bottom:14px' },
      statCard('Items reviewed', s.items, `${s.disputed} disputed`),
      statCard('Inter-annotator', fmtPct(s.inter_annotator_agreement),
        `${s.comparisons} overlapping comparisons`),
      statCard('Human vs AI', fmtPct(s.human_vs_ai_agreement),
        kappa === null || kappa === undefined ? 'no overlap yet' : `kappa ${fmtNum(kappa, 2)}`),
      statCard('Preference pairs', s.pairs_total, 'available to train on')),

    kappa !== null && kappa !== undefined && kappa < 0.4 ? h('div', { class: 'banner' },
      h('strong', { text: s.human_vs_ai_agreement < 0.5
        ? 'The judge disagrees with your reviewers more often than it agrees. '
        : 'The judge barely beats chance on this task. ' }),
      `Cohen's kappa is ${fmtNum(kappa, 2)} against raw agreement of ` +
      `${fmtPct(s.human_vs_ai_agreement)}` +
      (s.human_vs_ai_agreement >= 0.5
        ? ' — which looks respectable only because a two-way choice starts at 50%. '
        : ', which is worse than a coin flip on a two-way choice. ') +
      'Treat RLAIF pairs from this batch as weak evidence, and check whether the judge ' +
      'is being asked to grade something it cannot see.') : null,

    h('div', { class: 'grid c2' },
      h('div', { class: 'card' },
        h('div', { class: 'card-head' }, h('h3', { text: 'Pairs by source' })),
        h('div', { class: 'card-body tight' }, h('table', { class: 'data' },
          h('thead', {}, h('tr', {}, h('th', { text: 'Source' }), h('th', { text: 'Pairs' }),
            h('th', { text: 'Mean margin' }))),
          h('tbody', {}, s.pairs.map(p => h('tr', {},
            h('td', {}, h('span', { class: `chip ${p.source === 'ai' ? 'warn' : 'good'}`,
              text: p.source })),
            h('td', { class: 'num', text: String(p.count) }),
            h('td', { class: 'num', text: fmtNum(p.mean_margin, 2) }))))))),
      h('div', { class: 'card' },
        h('div', { class: 'card-head' }, h('h3', { text: 'Reviewers' })),
        h('div', { class: 'card-body tight' }, h('table', { class: 'data' },
          h('thead', {}, h('tr', {}, h('th', { text: 'Who' }), h('th', { text: 'Annotations' }),
            h('th', { text: 'Median time' }))),
          h('tbody', {}, s.annotator_throughput.map(a => h('tr', {},
            h('td', { text: a.name }), h('td', { class: 'num', text: String(a.annotations) }),
            h('td', { class: 'num', text: `${a.median_seconds}s` })))))))));
}

async function renderDisputed() {
  const items = await api('/api/review/disputed');
  set($('#review-panel'), items.length
    ? h('div', { class: 'grid' }, items.map(item => h('div', { class: 'card' },
        h('div', { class: 'card-head' }, h('h3', { text: `Item #${item.id}` }),
          h('span', { class: 'chip bad', text: 'disputed' })),
        h('div', { class: 'card-body' },
          h('blockquote', { style: 'margin:0 0 12px', text: item.prompt.slice(0, 600) }),
          h('div', { class: 'candidates two' }, item.candidates.map((c, i) =>
            h('div', { class: 'candidate' },
              h('div', { class: 'candidate-head' },
                h('span', { class: 'candidate-key', text: String(i + 1) }),
                h('span', { class: 'small muted', text: `Response ${c.id.toUpperCase()}` })),
              h('div', { class: 'candidate-body', text: c.text }),
              h('div', { class: 'candidate-foot' },
                h('button', { class: 'btn block', onClick: async () => {
                  await api('/api/review/adjudicate', { method: 'POST', body: {
                    item_id: item.id, choice: c.id,
                    rationale: $(`#adj-${item.id}`).value } });
                  toast('Adjudicated', 'good'); renderDisputed();
                } }, 'This one wins'))))),
          h('div', { class: 'sep' }),
          h('div', { class: 'field-label', text: 'What the reviewers said' }),
          h('table', { class: 'data' }, h('tbody', {}, item.annotations.map(a => h('tr', {},
            h('td', {}, h('span', { class: `chip ${a.type === 'ai' ? 'warn' : ''}`, text: a.who })),
            h('td', { class: 't-main', text: a.choice }),
            h('td', { class: 'small muted', text: a.rationale || '—' }))))),
          h('label', { class: 'field', style: 'margin-top:12px' },
            h('span', { class: 'field-label', text: 'Your rationale' }),
            h('input', { type: 'text', id: `adj-${item.id}`,
              placeholder: 'Why does this one win? This is the record that settles it.' }))))))
    : emptyState('⚖', 'Nothing to adjudicate',
        'Items land here when reviewers disagree past the agreement threshold.'));
}

// --------------------------------------------------------------------- models

$('#model-filter').addEventListener('change', renderModels);
$('#lb-dataset').addEventListener('change', renderLeaderboard);
$('#open-playground').addEventListener('click', openPlayground);
$('#open-keys').addEventListener('click', openKeys);

function renderModels() {
  const f = $('#model-filter').value;
  const rows = state.models.filter(m => !f || m.status === f);
  set($('#model-list'), rows.length ? h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ...['Model', 'Method', 'Headline', 'Status', 'Created']
      .map(t => h('th', { text: t })))),
    h('tbody', {}, rows.map(m => h('tr', { class: 'clickable', onClick: () => openModel(m.id) },
      h('td', {}, h('div', { class: 't-main', text: `${m.name} v${m.version}` }),
        h('div', { class: 't-sub', text: m.base_model })),
      h('td', {}, h('span', { class: 'chip', text: m.method.toUpperCase() }),
        m.artifact_kind === 'reward' ? h('span', { class: 'chip accent', text: 'RM' }) : null),
      h('td', { class: 'num', text: m.headline?.value != null ? fmtNum(m.headline.value) : '—' },
        m.headline?.label ? h('div', { class: 't-sub', text: m.headline.label }) : null),
      h('td', {}, h('div', { class: 'chips' }, statusChip(m.status),
        m.backend === 'tiny' ? tinyChip() : null)),
      h('td', { class: 't-sub', text: ago(m.created_at) })))))
    : emptyState('◱', 'No models yet', 'Finish a training run and its artifact lands here.'));

  const sel = $('#lb-dataset');
  const benches = state.datasets.filter(d => d.kind === 'benchmark' && d.status === 'ready');
  const prev = sel.value;
  set(sel, ...benches.map(d => h('option', { value: d.id, text: `${d.name} v${d.version}` })));
  if (prev) sel.value = prev;
  renderLeaderboard();
}

async function renderLeaderboard() {
  const id = $('#lb-dataset').value;
  const body = $('#lb-body');
  if (!id) { set(body, h('div', { class: 'card-body' },
    h('p', { class: 'muted small', text: 'Upload a benchmark dataset to rank models against it.' }))); return; }
  const rows = await api('/api/leaderboard', { query: { dataset_id: id } });
  set(body, rows.length ? h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, h('th', { text: '#' }), h('th', { text: 'Model' }),
      h('th', { text: 'Score' }))),
    h('tbody', {}, rows.map((r, i) => h('tr', { class: 'clickable',
      onClick: () => openModel(r.model_version_id) },
      h('td', { class: 'num', text: String(i + 1) }),
      h('td', {}, h('div', { class: 't-main', text: `${r.name} v${r.version}` }),
        h('div', { class: 't-sub', text: titleCase(r.method) })),
      h('td', { class: 'num', text: fmtNum(r.value) },
        h('div', { class: 't-sub', text: r.label || '' }))))))
    : h('div', { class: 'card-body' }, h('p', { class: 'muted small', text:
        'Nothing scored on this benchmark yet. Open a model and run an evaluation.' })));
}

async function openModel(id) {
  const m = await api(`/api/models/${id}`);
  const benches = state.datasets.filter(d => d.kind === 'benchmark' && d.status === 'ready');
  const nextStatus = { draft: 'staging', evaluated: 'staging', staging: 'production' }[m.status];

  openDrawer(`${m.name} v${m.version}`, [
    h('div', { class: 'chips', style: 'margin-bottom:14px' },
      statusChip(m.status), h('span', { class: 'chip accent', text: m.method.toUpperCase() }),
      h('span', { class: 'chip', text: m.base_model }),
      h('span', { class: 'chip', text: m.artifact_kind }),
      m.backend === 'tiny' ? tinyChip() : null),

    m.status === 'draft' ? h('div', { class: 'banner info' },
      'Not benchmarked. Score it against a benchmark dataset before promoting — the ' +
      'promotion endpoint will refuse a draft.') : null,

    h('div', { class: 'row auto', style: 'margin-bottom:14px' },
      nextStatus && can('lead') ? h('button', { class: 'btn primary', onClick: async () => {
        try {
          await api(`/api/models/${id}/promote`, { method: 'POST', body: { to: nextStatus } });
          toast(`Promoted to ${nextStatus}`, 'good');
          state.models = await api('/api/models'); renderModels(); openModel(id);
        } catch (e) { toast(e.message, 'bad'); }
      } }, `Promote to ${nextStatus}`) : null,
      benches.length ? h('select', { id: 'ev-ds', style: 'width:auto' },
        ...benches.map(d => h('option', { value: d.id, text: d.name }))) : null,
      benches.length ? h('button', { class: 'btn', onClick: async () => {
        const out = await api(`/api/models/${id}/evaluate`, { method: 'POST',
          body: { dataset_id: Number($('#ev-ds').value) } });
        toast(`Evaluation queued (job #${out.job_id || '—'})`, 'good');
      } }, 'Run evaluation') : null,
      h('button', { class: 'btn quiet', onClick: () => openPlayground(id) }, 'Try it')),

    deployCard(m),

    h('div', { class: 'card', style: 'margin-bottom:14px' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Lineage' })),
      h('div', { class: 'card-body tight' }, h('table', { class: 'data' },
        h('tbody', {}, m.lineage.map(n => h('tr', {},
          h('td', {}, h('span', { class: 'chip', text: n.method === 'base' ? 'base' : n.method.toUpperCase() })),
          h('td', { class: 't-main', text: n.version ? `${n.name} v${n.version}` : n.name }),
          h('td', { class: 'num', text: n.headline?.value != null ? fmtNum(n.headline.value) : '' }))))))),

    m.evals.length ? h('div', { class: 'card', style: 'margin-bottom:14px' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Benchmarks' })),
      h('div', { class: 'card-body tight' }, h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, h('th', { text: 'Dataset' }), h('th', { text: 'Status' }),
          h('th', { text: 'Metrics' }))),
        h('tbody', {}, m.evals.map(e => h('tr', {},
          h('td', { class: 't-main', text: e.dataset }),
          h('td', {}, statusChip(e.status)),
          h('td', { class: 'small mono', text: Object.entries(e.metrics || {})
            .filter(([k, v]) => typeof v === 'number' && k !== 'n')
            .map(([k, v]) => `${k} ${fmtNum(v, 3)}`).join('  ') || '—' }))))))) : null,

    h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Metrics' })),
      h('div', { class: 'card-body' }, h('dl', { class: 'kv' },
        ...Object.entries(m.metrics || {}).filter(([, v]) => typeof v !== 'object')
          .map(([k, v]) => [h('dt', { text: titleCase(k) }), h('dd', { text: fmtNum(v) })]).flat()))),
  ]);
  refreshExports(id);
}

/** Everything needed to call this model from outside the console, plus export.
 *  Lives on the model drawer because "how do I use this?" is the question people
 *  have while looking at a model, not on a separate page they have to find. */
function deployCard(m) {
  const alias = m.status === 'production' ? m.name : `${m.name}@${m.version}`;
  const tiny = m.backend === 'tiny';
  const curl = `curl ${location.origin}/v1/chat/completions \\
  -H "Authorization: Bearer $FOUNDRY_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${alias}",
    "messages": [{"role": "user", "content": "Hello"}]
  }'`;

  return h('div', { class: 'card', style: 'margin-bottom:14px' },
    h('div', { class: 'card-head' }, h('h3', { text: 'Use this model' }),
      m.status === 'production'
        ? h('span', { class: 'chip good', text: 'served at ' + m.name })
        : h('span', { class: 'chip', text: 'pin with @' + m.version })),
    h('div', { class: 'card-body' },
      tiny ? h('div', { class: 'banner' },
        h('strong', { text: 'Not servable. ' }),
        'This version was trained on the tiny backend, so its weights are a ' +
        'randomly-initialised stand-in. The inference endpoint refuses it rather than ' +
        'returning noise. The adapter still exports.') : null,

      h('p', { class: 'small muted' },
        m.status === 'production'
          ? 'Callers can use the bare name and always get whichever version is promoted.'
          : 'Only production versions answer to the bare name. Promote it, or pin the version.'),
      h('pre', { class: 'log', style: 'max-height:none', text: curl }),
      h('div', { class: 'row auto', style: 'margin-top:10px' },
        h('button', { class: 'btn sm', onClick: () => copy(curl) }, 'Copy curl'),
        h('button', { class: 'btn sm quiet', onClick: openKeys }, 'API keys'),
        h('span', { style: 'flex:1' }),
        h('button', { class: 'btn sm', onClick: () => runExport(m.id, 'adapter') },
          'Export adapter'),
        h('button', { class: 'btn sm', title: tiny
            ? 'Needs the real base model — unavailable for a tiny-backend version'
            : 'Folds the adapter into the base weights: a self-contained model',
          onClick: () => runExport(m.id, 'merged') }, 'Export merged')),
      h('div', { id: 'export-list', style: 'margin-top:12px' })));
}

async function refreshExports(modelId) {
  const box = $('#export-list'); if (!box) return;
  const rows = await api(`/api/models/${modelId}/exports`);
  if (!rows.length) return set(box, h('p', { class: 'small muted', text:
    'No exports yet. Adapter is a few MB and needs the base model; merged is a ' +
    'self-contained model directory.' }));
  set(box, h('table', { class: 'data' },
    h('tbody', {}, rows.map(e => h('tr', {},
      h('td', {}, h('span', { class: 'chip', text: e.format })),
      h('td', {}, statusChip(e.status),
        e.error ? h('div', { class: 't-sub', style: 'color:var(--bad)',
                             text: e.error.slice(0, 120) }) : null),
      h('td', { class: 'num', text: e.bytes ? fmtBytes(e.bytes) : '—' }),
      h('td', { class: 'num mono t-sub', text: e.sha256 || '' }),
      h('td', {}, e.status === 'succeeded'
        ? h('a', { class: 'btn sm', href: `/api/exports/${e.id}/download` }, 'Download')
        : null))))));
}

async function runExport(modelId, fmt) {
  try {
    await api(`/api/models/${modelId}/export`, { method: 'POST', body: { fmt } });
    toast(`${fmt} export queued`, 'good');
    refreshExports(modelId);
    // Exports take seconds, so poll briefly rather than making the user refresh.
    let n = 0;
    const t = setInterval(() => { refreshExports(modelId); if (++n > 20) clearInterval(t); }, 3000);
  } catch (e) { toast(e.message, 'bad'); }
}

function copy(text) {
  navigator.clipboard.writeText(text).then(
    () => toast('Copied', 'good'), () => toast('Could not copy', 'bad'));
}

/** API key management. The secret is shown once, in the response that created it,
 *  and there is deliberately no way to retrieve it afterwards. */
async function openKeys() {
  const keys = await api('/api/keys');
  openDrawer('API keys', [
    h('p', { class: 'muted small' },
      'Session tokens expire in twelve hours. These do not — use them for services, ' +
      'scripts and CI. A ', h('code', { text: 'serve' }), ' key can only call ',
      h('code', { text: '/v1' }), '; it cannot touch your data.'),
    h('div', { class: 'card', style: 'margin-bottom:14px' },
      h('div', { class: 'card-head' }, h('h3', { text: 'New key' })),
      h('div', { class: 'card-body' },
        h('div', { class: 'row' },
          h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Name' }),
            h('input', { type: 'text', id: 'key-name', placeholder: 'prod-support-service' })),
          h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Scope' }),
            h('select', { id: 'key-scope' },
              h('option', { value: 'serve', text: 'serve — /v1 only' }),
              h('option', { value: 'full', text: 'full — everything you can do' })))),
        h('button', { class: 'btn primary', onClick: createKey }, 'Create key'),
        h('div', { id: 'key-out', style: 'margin-top:12px' }))),
    h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Existing keys' })),
      keys.length ? h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, ...['Name', 'Key', 'Scope', 'Calls', 'Last used', '']
          .map(x => h('th', { text: x })))),
        h('tbody', {}, keys.map(k => h('tr', {},
          h('td', { class: 't-main', text: k.name }),
          h('td', { class: 'mono t-sub', text: k.prefix + '…' }),
          h('td', {}, h('span', { class: 'chip', text: k.scope })),
          h('td', { class: 'num', text: String(k.calls) }),
          h('td', { class: 't-sub', text: k.last_used_at ? ago(k.last_used_at) : 'never' }),
          h('td', {}, k.revoked
            ? h('span', { class: 'chip bad', text: 'revoked' })
            : h('button', { class: 'btn sm danger', onClick: async () => {
                await api(`/api/keys/${k.id}`, { method: 'DELETE' });
                toast('Key revoked'); openKeys();
              } }, 'Revoke')))))) 
        : h('div', { class: 'card-body' },
            h('p', { class: 'muted small', text: 'No keys yet.' }))),
  ]);

  async function createKey() {
    const name = $('#key-name').value.trim();
    if (!name) return toast('Give the key a name', 'warn');
    try {
      const k = await api('/api/keys', { method: 'POST',
        body: { name, scope: $('#key-scope').value } });
      set($('#key-out'),
        h('div', { class: 'banner' }, h('strong', { text: k.warning })),
        h('pre', { class: 'log', style: 'max-height:none', text: k.secret }),
        h('button', { class: 'btn sm', onClick: () => copy(k.secret) }, 'Copy key'));
    } catch (e) { toast(e.message, 'bad'); }
  }
}

function openPlayground(modelId = null) {
  const models = state.models.filter(m => m.artifact_kind === 'adapter');
  openDrawer('Playground', [
    h('p', { class: 'muted small' },
      'One generation, run synchronously — the only place in the product that does not ' +
      'go through the queue. Comparing against base runs the same model with the LoRA ' +
      'adapters switched off, so nothing but the fine-tune differs.'),
    h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Model' }),
      h('select', { id: 'pg-model' }, ...models.map(m => h('option', { value: m.id,
        text: `${m.name} v${m.version}`, selected: m.id === modelId })))),
    h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'System (optional)' }),
      h('input', { type: 'text', id: 'pg-system' })),
    h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Prompt' }),
      h('textarea', { id: 'pg-prompt', placeholder: 'Ask it something…' })),
    h('div', { class: 'row' },
      h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Max new tokens' }),
        h('input', { type: 'number', id: 'pg-max', value: '160', min: '1', max: '1024' })),
      h('label', { class: 'field' }, h('span', { class: 'field-label', text: 'Temperature' }),
        h('input', { type: 'number', id: 'pg-temp', value: '0.7', step: '0.1', min: '0', max: '2' }))),
    h('label', { class: 'switch' },
      h('input', { type: 'checkbox', id: 'pg-compare', checked: true }),
      h('span', { text: 'Compare against the base model' })),
    h('button', { class: 'btn primary block lg', id: 'pg-go' }, 'Generate'),
    h('div', { id: 'pg-out', style: 'margin-top:16px' }),
  ]);

  $('#pg-go').addEventListener('click', async () => {
    const btn = $('#pg-go'), out = $('#pg-out');
    btn.disabled = true; set(btn, h('span', { class: 'spinner' }), ' Generating…');
    try {
      const r = await api('/api/playground', { method: 'POST', body: {
        model_version_id: Number($('#pg-model').value) || null,
        prompt: $('#pg-prompt').value, system: $('#pg-system').value,
        max_new_tokens: Number($('#pg-max').value),
        temperature: Number($('#pg-temp').value),
        compare_to_base: $('#pg-compare').checked,
      } });
      set(out,
        // The tiny backend's vocabulary is derived from whatever rows it was given,
        // so a single playground prompt yields a one-word vocabulary and an
        // immediate EOS. Say that, rather than rendering an empty box that reads
        // as a bug.
        r.backend === 'tiny' ? h('div', { class: 'banner' },
          h('strong', { text: 'Tiny backend — generation here is not meaningful. ' }),
          'The requested base model is not available locally, so this is a ' +
          'randomly-initialised stand-in whose vocabulary comes from training data it ' +
          'cannot see from a single prompt. Download the base model to use the playground.') : null,
        h('div', { class: `candidates ${r.base_response !== undefined ? 'two' : ''}` },
        h('div', { class: 'candidate picked' },
          h('div', { class: 'candidate-head' }, h('span', { class: 'candidate-key', text: '★' }),
            h('span', { class: 'small', text: r.model }),
            r.backend === 'tiny' ? tinyChip() : null),
          h('div', { class: 'candidate-body', text: r.response || '(empty)' })),
        r.base_response !== undefined ? h('div', { class: 'candidate' },
          h('div', { class: 'candidate-head' }, h('span', { class: 'candidate-key', text: '○' }),
            h('span', { class: 'small muted', text: `${r.base_model} (adapters off)` })),
          h('div', { class: 'candidate-body', text: r.base_response || '(empty)' })) : null));
    } catch (e) { set(out, h('div', { class: 'banner bad', text: e.message })); }
    finally { btn.disabled = false; btn.textContent = 'Generate'; }
  });
}

// ---------------------------------------------------------------------- queue

$('#queue-auto').addEventListener('change', e => {
  clearInterval(state.timers.queue);
  if (e.target.checked) state.timers.queue = setInterval(() => {
    if (state.view === 'queue') pollLight();
  }, 4000);
});

function renderQueue() {
  const q = state.queue; if (!q) return;
  const alive = q.workers.filter(w => w.alive);

  set($('#queue-panel'),
    !alive.length ? h('div', { class: 'banner bad' },
      h('strong', { text: 'No live workers. ' }),
      'Jobs will sit in the queue. Start one with ',
      h('code', { text: 'python -m foundry.worker --kinds train,eval,generate,judge,assemble' }),
      '.') : null,

    h('div', { class: 'grid c4', style: 'margin-bottom:14px' },
      statCard('Your running', `${q.team.running}/${q.team.concurrency}`, 'team concurrency cap'),
      statCard('Your queued', q.active.filter(j => j.status === 'queued').length, 'waiting'),
      statCard('Cluster queued', q.cluster_queued, 'across all teams'),
      statCard('Workers', `${alive.length}/${q.workers.length}`, 'alive / registered')),

    h('div', { class: 'grid c2' },
      h('div', { class: 'card' },
        h('div', { class: 'card-head' }, h('h3', { text: 'Workers' })),
        q.workers.length ? h('table', { class: 'data' },
          h('thead', {}, h('tr', {}, ...['Worker', 'Kinds', 'Job', 'Hardware', 'Seen']
            .map(t => h('th', { text: t })))),
          h('tbody', {}, q.workers.map(w => h('tr', {},
            h('td', {}, h('div', { class: 't-main', text: w.worker_id.slice(0, 26) }),
              h('div', { class: 't-sub', text: w.hostname })),
            h('td', { class: 'small', text: (w.kinds || []).join(', ') }),
            h('td', { class: 'num', text: w.current_job_id ? `#${w.current_job_id}` : '—' }),
            h('td', { class: 'small' }, h('span', { class: 'chip', text:
              w.capabilities?.gpu || (w.capabilities?.mps ? 'mps' : 'cpu') }),
              w.capabilities?.judge === 'heuristic-v1'
                ? h('span', { class: 'chip warn', title:
                    'No ANTHROPIC_API_KEY on this worker — RLAIF falls back to a mechanical scorer',
                    text: 'no judge' }) : null),
            h('td', {}, h('span', { class: `chip ${w.alive ? 'good' : 'bad'}` },
              w.alive ? h('span', { class: 'dot' }) : null, ago(w.last_seen_at)))))))
          : h('div', { class: 'card-body' }, h('p', { class: 'muted small', text: 'None registered.' }))),

      h('div', { class: 'card' },
        h('div', { class: 'card-head' }, h('h3', { text: 'Fair share' }),
          h('span', { class: 'chip', title:
            'Candidates are ordered by how many jobs a team already has running, then by ' +
            'priority. A team with nothing running always gets the next free worker.',
            text: 'how this works' })),
        q.fair_share.length ? h('table', { class: 'data' },
          h('thead', {}, h('tr', {}, h('th', { text: 'Team' }), h('th', { text: 'Status' }),
            h('th', { text: 'Jobs' }))),
          h('tbody', {}, q.fair_share.map(r => h('tr', {},
            h('td', { class: 't-main', text: r.team }),
            h('td', {}, statusChip(r.status)),
            h('td', { class: 'num', text: String(r.count) })))))
          : h('div', { class: 'card-body' }, h('p', { class: 'muted small', text: 'Queue is empty.' })))),

    h('div', { class: 'card', style: 'margin-top:14px' },
      h('div', { class: 'card-head' }, h('h3', { text: 'Your jobs' })),
      (q.active.length || q.recent.length) ? h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, ...['Job', 'Kind', 'Status', 'Progress', 'Attempts', 'Worker', '']
          .map(t => h('th', { text: t })))),
        h('tbody', {}, [...q.active, ...q.recent].map(j => h('tr', {},
          h('td', {}, h('div', { class: 't-main', text: `#${j.id}` }),
            j.run_id ? h('div', { class: 't-sub', text: `run #${j.run_id}` }) : null),
          h('td', {}, h('span', { class: 'chip', text: j.kind })),
          h('td', {}, statusChip(j.status),
            j.error ? h('div', { class: 't-sub', style: 'color:var(--bad)',
              text: j.error.slice(0, 90) }) : null),
          h('td', { style: 'width:120px' },
            h('div', { class: 'rail-bar' }, h('div', {
              class: `rail-fill ${j.status === 'succeeded' ? 'good' : j.status === 'failed' ? 'bad' : ''}`,
              style: `width:${j.progress?.pct || (j.status === 'succeeded' ? 100 : 0)}%` }))),
          h('td', { class: 'num', text: `${j.attempts}/${j.max_attempts}` }),
          h('td', { class: 't-sub', text: j.worker_id ? j.worker_id.slice(0, 18) : '—' }),
          h('td', {}, ['queued', 'running'].includes(j.status)
            ? h('button', { class: 'btn sm danger', onClick: async () => {
                await api(`/api/jobs/${j.id}/cancel`, { method: 'POST' });
                toast('Cancellation requested'); pollLight();
              } }, 'Cancel')
            : (j.run_id ? h('button', { class: 'btn sm quiet',
                onClick: () => openRun(j.run_id) }, 'Run') : null))))))
        : h('div', { class: 'card-body' }, h('p', { class: 'muted small', text: 'No jobs yet.' }))));
}

// --------------------------------------------------------------------- drawer

function openDrawer(title, content, onClose) {
  closeDrawer();
  const root = $('#drawer-root');
  const backdrop = h('div', { class: 'drawer-backdrop', onClick: closeDrawer });
  const drawer = h('aside', { class: 'drawer' },
    h('div', { class: 'drawer-head' },
      h('h2', { id: 'drawer-title', text: title }),
      h('button', { class: 'btn quiet', onClick: closeDrawer }, 'Close')),
    h('div', { class: 'drawer-body', id: 'drawer-body' }, ...[content].flat()));
  root.append(backdrop, drawer);
  state.onDrawerClose = onClose;
}
function closeDrawer() {
  clear($('#drawer-root'));
  if (state.onDrawerClose) { state.onDrawerClose(); state.onDrawerClose = null; }
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

$('#help-btn').addEventListener('click', () => openDrawer('Keyboard & concepts', [
  h('div', { class: 'card', style: 'margin-bottom:14px' },
    h('div', { class: 'card-head' }, h('h3', { text: 'Review console' })),
    h('div', { class: 'card-body' }, h('dl', { class: 'kv' },
      h('dt', {}, h('span', { class: 'kbd', text: '1' }), ' / ', h('span', { class: 'kbd', text: '2' })),
      h('dd', { text: 'Pick that response' }),
      h('dt', {}, h('span', { class: 'kbd', text: 'T' })), h('dd', { text: 'Equally good' }),
      h('dt', {}, h('span', { class: 'kbd', text: 'B' })), h('dd', { text: 'Both bad' }),
      h('dt', {}, h('span', { class: 'kbd', text: 'S' })), h('dd', { text: 'Skip' }),
      h('dt', {}, h('span', { class: 'kbd', text: '↵' })), h('dd', { text: 'Submit and load the next' }),
      h('dt', {}, h('span', { class: 'kbd', text: 'Esc' })), h('dd', { text: 'Close a panel' })))),
  h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', { text: 'Which method?' })),
    h('div', { class: 'card-body' },
      h('p', { class: 'small' }, h('strong', { text: 'SFT' }),
        ' — always first. Every preference method assumes a policy that is already competent.'),
      h('p', { class: 'small' }, h('strong', { text: 'DPO' }),
        ' — the default way to spend preference pairs. No reward model, no sampling loop, ' +
        'hardest to destabilise.'),
      h('p', { class: 'small' }, h('strong', { text: 'GRPO' }),
        ' — when you want on-policy RL without a critic. Needs a reward signal and a ' +
        'group of samples per prompt.'),
      h('p', { class: 'small' }, h('strong', { text: 'GSPO' }),
        ' — GRPO for long generations. Clips at sequence level, so one freak token cannot ' +
        'dominate an update.'),
      h('p', { class: 'small' }, h('strong', { text: 'PPO' }),
        ' — most control, most moving parts, and the only one needing a trained reward model.'),
      h('p', { class: 'small' }, h('strong', { text: 'RLAIF' }),
        ' — not an optimiser. It replaces the human labelling step with a judge model and ' +
        'then hands the pairs to DPO, GRPO or GSPO.'))),
]));

// ----------------------------------------------------------------------- init

(async function init() {
  if (!state.token) {
    try {
      const probe = await fetch('/api/health').then(r => r.json());
      if (probe.warning) $('#login-hint').textContent = probe.warning;
    } catch { /* server may still be starting */ }
    return;
  }
  try { await boot(); } catch { logout(); }
})();
