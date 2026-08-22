/* Shared behaviour for the SLM Foundry manuals: search, filters, TOC tracking.
 *
 * Search walks the DOM once on load and records the text of every block element
 * against the section that contains it. Filtering then hides whole sections and
 * highlights matches inside the survivors. No index file, no build step, and it
 * stays correct when someone edits the prose — the index is the page.
 */
'use strict';

(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  const search = $('#search');
  const sections = $$('main section[id]');
  const tocLinks = $$('.toc a[href^="#"]');
  const status = $('#search-status');
  const empty = $('#no-results');

  // ------------------------------------------------------------------ index
  // One entry per section: its heading, its plain text, and the blocks we may
  // highlight inside it. Captured before any <mark> is ever inserted, so
  // repeated searches never index their own highlights.
  const index = sections.map(sec => ({
    el: sec,
    id: sec.id,
    level: sec.dataset.level || '',
    title: (sec.querySelector('h2, h3') || {}).textContent || sec.id,
    text: sec.textContent.toLowerCase(),
    blocks: $$('p, li, td, th, .math, .def dt, .def dd, h2, h3, h4', sec),
  }));
  index.forEach(entry => {
    entry.blocks.forEach(b => { b.dataset.raw = b.innerHTML; });
  });

  let activeFilter = 'all';
  let query = '';

  // --------------------------------------------------------------- highlight
  function clearMarks(entry) {
    entry.blocks.forEach(b => {
      if (b.innerHTML !== b.dataset.raw) b.innerHTML = b.dataset.raw;
    });
  }

  const escapeRe = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  function highlight(entry, q) {
    const re = new RegExp('(' + escapeRe(q) + ')', 'gi');
    entry.blocks.forEach(b => {
      const raw = b.dataset.raw;
      if (!raw.toLowerCase().includes(q)) { if (b.innerHTML !== raw) b.innerHTML = raw; return; }
      // Only rewrite text nodes — splicing <mark> into an attribute or a tag
      // name would corrupt the markup.
      let out = '', depth = 0, buf = '';
      for (const ch of raw) {
        if (ch === '<') { out += flush(buf, re); buf = ''; depth++; out += ch; }
        else if (ch === '>') { depth--; out += ch; }
        else if (depth > 0) out += ch;
        else buf += ch;
      }
      out += flush(buf, re);
      b.innerHTML = out;
    });
  }
  const flush = (text, re) => text ? text.replace(re, '<mark>$1</mark>') : '';

  // ------------------------------------------------------------------ apply
  function apply() {
    const q = query.trim().toLowerCase();
    let shown = 0;

    index.forEach(entry => {
      const passesFilter = activeFilter === 'all' || entry.level === activeFilter ||
                           entry.level === 'always';
      const passesQuery = !q || entry.text.includes(q);
      const visible = passesFilter && passesQuery;

      entry.el.classList.toggle('hidden', !visible);
      if (!visible) { clearMarks(entry); return; }
      shown++;
      q ? highlight(entry, q) : clearMarks(entry);
    });

    // TOC follows the same filter, so it never points at something hidden.
    tocLinks.forEach(a => {
      const entry = index.find(e => '#' + e.id === a.getAttribute('href'));
      if (entry) a.style.display = entry.el.classList.contains('hidden') ? 'none' : '';
    });

    if (empty) empty.hidden = shown > 0;
    if (status) {
      if (q) {
        status.hidden = false;
        status.textContent = shown
          ? `${shown === 1 ? '1 section matches' : shown + ' sections match'} ` +
            `“${query.trim()}”. Press Esc to clear.`
          : `Nothing matches “${query.trim()}”.`;
      } else {
        status.hidden = true;
      }
    }
  }

  // ----------------------------------------------------------------- events
  if (search) {
    let t;
    search.addEventListener('input', e => {
      query = e.target.value;
      clearTimeout(t);
      t = setTimeout(apply, 120);      // debounce: highlighting rewrites a lot of DOM
    });
    search.addEventListener('keydown', e => {
      if (e.key === 'Escape') { search.value = ''; query = ''; apply(); search.blur(); }
    });
  }

  document.addEventListener('keydown', e => {
    if (e.key === '/' && document.activeElement !== search &&
        !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
      e.preventDefault();
      search && search.focus();
    }
  });

  $$('.filters button').forEach(btn => {
    btn.addEventListener('click', () => {
      activeFilter = btn.dataset.filter;
      $$('.filters button').forEach(b => b.classList.toggle('on', b === btn));
      apply();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });

  // --------------------------------------------------- active TOC on scroll
  const byId = new Map(tocLinks.map(a => [a.getAttribute('href').slice(1), a]));
  const seen = new Set();
  const io = new IntersectionObserver(entries => {
    entries.forEach(en => {
      en.isIntersecting ? seen.add(en.target.id) : seen.delete(en.target.id);
    });
    // Topmost visible section wins, so the highlight does not flicker between
    // two sections that are both partly on screen.
    const first = sections.find(s => seen.has(s.id));
    tocLinks.forEach(a => a.classList.remove('active'));
    if (first && byId.has(first.id)) byId.get(first.id).classList.add('active');
  }, { rootMargin: '-70px 0px -70% 0px', threshold: 0 });
  sections.forEach(s => io.observe(s));

  // ------------------------------------------------------------ theme toggle
  const toggle = $('#theme-toggle');
  if (toggle) {
    const saved = localStorage.getItem('foundry_docs_theme');
    if (saved) document.documentElement.dataset.theme = saved;
    const label = () => {
      const t = document.documentElement.dataset.theme;
      toggle.textContent = t === 'dark' ? 'Light' : t === 'light' ? 'Dark' : 'Theme';
    };
    label();
    toggle.addEventListener('click', () => {
      const cur = document.documentElement.dataset.theme;
      const next = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      localStorage.setItem('foundry_docs_theme', next);
      label();
    });
  }

  // Deep link with a search term: manual.html?q=lora
  const params = new URLSearchParams(location.search);
  if (params.get('q') && search) {
    search.value = params.get('q');
    query = params.get('q');
    apply();
  }
})();
