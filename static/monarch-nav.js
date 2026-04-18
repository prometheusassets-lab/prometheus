/* monarch-nav.js — shared navigation injected into every page */
(function () {
  const pages = [
    { label: "SCREENER", href: "/" },
    { label: "OPTIONS", href: "/options" },
    { label: "FUNDAMENTALS", href: "/fundamentals" },
    { label: "ML PREDICTOR", href: "/ml" },
  ];
  const current = location.pathname.replace(/\/$/, "") || "/";

  const nav = document.createElement("nav");
  nav.id = "monarch-nav";
  nav.innerHTML = `
    <div class="nav-brand">◼ MONARCH <span>PRO</span></div>
    <div class="nav-links">
      ${pages.map(p => {
        const active = (current === p.href) || (p.href !== "/" && current.startsWith(p.href));
        return `<a href="${p.href}" class="nav-link${active ? " active" : ""}">${p.label}</a>`;
      }).join("")}
    </div>
    <div class="nav-right">
      <span id="nav-status-dot" class="nav-dot"></span>
      <span id="nav-status-txt" style="font-size:.60rem;color:#888">—</span>
      <a href="/login" class="nav-auth">AUTH</a>
    </div>`;

  const style = document.createElement("style");
  style.textContent = `
    #monarch-nav {
      display: flex; align-items: center; gap: 0;
      background: #080808; border-bottom: 1px solid #2a2a2a;
      padding: 0 20px; height: 40px; position: sticky; top: 0; z-index: 100;
      font-family: 'IBM Plex Mono', monospace;
    }
    .nav-brand { color: #ff8c00; font-size: .72rem; font-weight: 700;
                 letter-spacing: .20em; text-transform: uppercase; margin-right: 32px;
                 white-space: nowrap; }
    .nav-brand span { color: #ffb347; }
    .nav-links { display: flex; flex: 1; }
    .nav-link { color: #666; font-size: .62rem; font-weight: 600; letter-spacing: .12em;
                text-transform: uppercase; text-decoration: none; padding: 0 16px;
                height: 40px; display: flex; align-items: center; border-right: 1px solid #1a1a1a;
                transition: color .15s, background .15s; white-space: nowrap; }
    .nav-link:hover { color: #ffb347; background: #110900; }
    .nav-link.active { color: #ff8c00; background: #140e00; border-bottom: 2px solid #ff8c00; }
    .nav-right { display: flex; align-items: center; gap: 10px; margin-left: auto; }
    .nav-dot { width: 7px; height: 7px; border-radius: 50%; background: #333; }
    .nav-dot.ok { background: #00d084; box-shadow: 0 0 5px #00d084; }
    .nav-auth { color: #666; font-size: .58rem; font-weight: 600; letter-spacing: .1em;
                text-transform: uppercase; text-decoration: none; border: 1px solid #2a2a2a;
                padding: 3px 10px; }
    .nav-auth:hover { color: #ff8c00; border-color: #ff8c00; }
  `;

  document.head.appendChild(style);
  document.body.insertBefore(nav, document.body.firstChild);

  // Poll auth status
  async function pollStatus() {
    try {
      const d = await fetch("/auth/status").then(r => r.json());
      const dot = document.getElementById("nav-status-dot");
      const txt = document.getElementById("nav-status-txt");
      if (d.connected) {
        dot.classList.add("ok");
        txt.textContent = d.prefix + "…";
        txt.style.color = "#00d084";
      } else {
        dot.classList.remove("ok");
        txt.textContent = "NO TOKEN";
        txt.style.color = "#ff3b3b";
      }
    } catch (e) {}
  }
  pollStatus();
  setInterval(pollStatus, 30000);
})();
